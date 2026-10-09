#!/usr/bin/env python3
"""ROS 2 node: SO-101 (parallel two-finger gripper) pick-and-place in Gazebo - second model.

The block is a rigid body and is held ONLY by friction: both fingers close on it and
squeeze it (force-limited position control). There is no attach/detach "grasp assist".

Two modes
  collect : a scripted IK expert performs the task; every control step
            (image, joint state, commanded action) is saved as an episode .npz
  eval    : actions come from the learned ACT policy (served over TCP by
            policy_server.py, which runs in the Colab Python with LeRobot/torch)

State / action (6-d, same layout as the first model)
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
  gripper = opening of each finger from the centre line in metres
            (0 = fully closed, 0.03 = open, block half-width = 0.0125).
  Both fingers always receive the same command -> they open/close symmetrically.

The control loop is driven by camera frames (10 Hz in *simulation* time).
Runs under the ROS 2 system Python (rclpy); only numpy is needed besides ROS.
"""
import argparse, os, pickle, re, socket, struct, subprocess, sys, threading, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_kin import SO101Kinematics, ARM_JOINTS  # noqa: E402

FINGERS = ["finger_left", "finger_right"]
CMD_JOINTS = ARM_JOINTS + FINGERS
WORLD = "pick_place"
BLOCK_HALF = 0.0125
TRAY_XY = np.array([0.10, 0.20])
TRAY_HALF = 0.045 - 0.008
HOME = np.array([0.0, -1.5, 0.6, 1.6, 0.0, 0.02])   # rad x5, finger opening (m)
GRIP_OPEN, GRIP_CLOSED = 0.030, 0.0                  # closed target is inside the block -> squeeze
GRASP_Z = 0.012          # TCP height when grasping (fingertips ~4 mm above the table)
CARRY_Z = 0.080
PLACE_Z = 0.050
BLOCK_X = (0.17, 0.25)
BLOCK_Y = (-0.08, 0.05)


# --------------------------------------------------------------------------- helpers
def gz_set_pose(name, x, y, z, yaw=0.0):
    req = (f'name: "{name}", position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}, '
           f'orientation: {{w: {np.cos(yaw / 2):.6f}, z: {np.sin(yaw / 2):.6f}}}')
    subprocess.run(["gz", "service", "-s", f"/world/{WORLD}/set_pose", "--reqtype", "gz.msgs.Pose",
                    "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", req],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def gz_get_pose(name):
    """Actual model position in Gazebo (used to verify resets and to judge success)."""
    r = subprocess.run(["gz", "model", "-m", name, "-p"], capture_output=True, text=True, timeout=20)
    nums = re.findall(r"Pose \[ XYZ \(m\) \][^\n]*\n\s*\[([-0-9.e ]+)\]", r.stdout)
    if not nums:
        return None
    return np.array([float(v) for v in nums[0].split()])


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_msg(sock):
    hdr = b""
    while len(hdr) < 4:
        chunk = sock.recv(4 - len(hdr))
        if not chunk:
            raise ConnectionError("policy server closed")
        hdr += chunk
    n = struct.unpack("!I", hdr)[0]
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("policy server closed")
        buf += chunk
    return pickle.loads(bytes(buf))


def lerp_traj(waypoints, steps):
    """waypoints: list of 6-d vectors; steps: list of step counts per segment."""
    out = []
    for a, b, n in zip(waypoints[:-1], waypoints[1:], steps):
        for i in range(1, n + 1):
            s = i / n
            s = 0.5 - 0.5 * np.cos(np.pi * s)  # smooth (cosine) profile
            out.append((1 - s) * a + s * b)
    return out


# --------------------------------------------------------------------------- node
class TaskNode(Node):
    def __init__(self, args):
        super().__init__("so101_task_node")
        self.args = args
        self.kin = SO101Kinematics(args.urdf)
        self.cmd_pubs = {j: self.create_publisher(Float64, f"/so101/{j}/cmd", 10) for j in CMD_JOINTS}
        self.joint_pos = None
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.video_frame = None
        self.create_subscription(JointState, "/so101/joint_states", self.on_joints, 10)
        self.create_subscription(Image, "/top_cam/image", self.on_image, qos_profile_sensor_data)
        self.create_subscription(Image, "/video_cam/image", self.on_video, qos_profile_sensor_data)
        self.block_xyz = np.array([0.2, 0.0, BLOCK_HALF])

    # ---- callbacks
    def on_joints(self, msg):
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in CMD_JOINTS):
            grip = 0.5 * (d["finger_left"] + d["finger_right"])
            self.joint_pos = np.array([d[j] for j in ARM_JOINTS] + [grip], dtype=np.float32)

    @staticmethod
    def _to_np(msg):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[..., :3].copy()

    def on_image(self, msg):
        with self.lock:
            self.frame = self._to_np(msg)
            self.frame_id += 1

    def on_video(self, msg):
        self.video_frame = self._to_np(msg)

    # ---- low level
    def command(self, a):
        for j, v in zip(ARM_JOINTS, a[:5]):
            self.cmd_pubs[j].publish(Float64(data=float(v)))
        g = float(np.clip(a[5], 0.0, 0.035))
        for j in FINGERS:                      # both fingers: same command -> symmetric open/close
            self.cmd_pubs[j].publish(Float64(data=g))

    def wait_frame(self, last_id, timeout=10.0):
        t0 = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            with self.lock:
                if self.frame_id != last_id and self.frame is not None and self.joint_pos is not None:
                    return self.frame_id, self.frame.copy(), self.joint_pos.copy()
            if time.time() - t0 > timeout:
                raise TimeoutError("no camera frame / joint state received - is the simulator running?")

    def tool_xyz(self, q):
        return self.kin.fk(q[:5])[:3, 3]

    def wait_bridge(self, timeout=20.0):
        """Wait until the ros_gz_bridge subscribes to our command topics (messages sent earlier are lost)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if all(p.get_subscription_count() > 0 for p in self.cmd_pubs.values()):
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise TimeoutError("ros_gz_bridge is not subscribed to the command topics")

    def closing_axis_yaw(self, xyz):
        """Yaw of the fingers' closing axis when grasping top-down at xyz (block faces are aligned to it)."""
        q, _ = self.kin.ik(np.array([xyz[0], xyz[1], GRASP_Z]))
        x_axis = self.kin.fk(q)[:3, 0]
        return float(np.arctan2(x_axis[1], x_axis[0]))

    def reset(self, rng):
        self.wait_bridge()
        bx, by = rng.uniform(*BLOCK_X), rng.uniform(*BLOCK_Y)
        yaw = self.closing_axis_yaw((bx, by)) + rng.uniform(-0.15, 0.15)
        fid = self.frame_id
        for _ in range(8):                    # go home with the fingers open (drops anything held)
            self.command(HOME)
            fid, _, _ = self.wait_frame(fid)
        for attempt in range(4):
            gz_set_pose("block", bx, by, BLOCK_HALF + 0.001, yaw)
            for _ in range(15):               # ~1.5 s of sim time to settle
                fid, _, _ = self.wait_frame(fid)
                self.command(HOME)
            real = gz_get_pose("block")
            if real is not None and abs(real[2] - BLOCK_HALF) < 0.005 and np.linalg.norm(real[:2] - [bx, by]) < 0.01:
                break
            self.get_logger().warn(f"block reset not confirmed (pose={real}), retrying")
        else:
            raise RuntimeError("could not reset the block")
        self.block_xyz = np.array([bx, by, BLOCK_HALF])
        fid, _, _ = self.wait_frame(fid)
        return fid

    def real_success(self, q):
        """Success judged from the block's actual pose in Gazebo: resting inside the tray, fingers open."""
        real = gz_get_pose("block")
        ok = (real is not None and real[2] < 0.03 and float(q[5]) > 0.018
              and bool(np.all(np.abs(real[:2] - TRAY_XY) < TRAY_HALF)))
        return ok, real

    # ---- scripted expert
    def expert_plan(self, rng):
        k, b = self.kin, self.block_xyz
        jit = lambda: rng.uniform(-0.003, 0.003, 3) * np.array([1, 1, 0])
        def ik(p, q0=None):
            q, ok = k.ik(p, q0=q0)
            if not ok:
                raise RuntimeError(f"IK failed for {p}")
            return q
        above = ik(np.array([b[0], b[1], CARRY_Z]) + jit())
        grasp = ik(np.array([b[0], b[1], GRASP_Z]), q0=above.copy())
        lift = ik(np.array([b[0], b[1], CARRY_Z]), q0=grasp.copy())
        t = np.array([*(TRAY_XY + rng.uniform(-0.01, 0.01, 2)), CARRY_Z])
        over_tray = ik(t)
        place = ik(np.array([t[0], t[1], PLACE_Z]), q0=over_tray.copy())
        g = lambda q, gr: np.concatenate([q, [gr]])
        wps = [HOME.copy(), g(above, GRIP_OPEN), g(grasp, GRIP_OPEN), g(grasp, GRIP_CLOSED), g(grasp, GRIP_CLOSED),
               g(lift, GRIP_CLOSED), g(over_tray, GRIP_CLOSED), g(place, GRIP_CLOSED),
               g(place, GRIP_OPEN), g(over_tray, GRIP_OPEN), HOME.copy()]
        r = lambda n: int(n + rng.integers(-2, 3))
        #        home>above above>grasp close hold lift  >tray  down  open  up  home
        steps = [r(14),      r(10),      4,    4,   r(10), r(14), 6,    4,    5,  r(12)]
        traj = lerp_traj(wps, steps)
        traj += [HOME.copy()] * 5
        return traj

    # ---- episodes
    def run_collect(self):
        rng = np.random.default_rng(self.args.seed)
        os.makedirs(self.args.out, exist_ok=True)
        n_ok = 0
        ep = 0
        while n_ok < self.args.episodes and ep < self.args.max_tries:
            fid = self.reset(rng)
            try:
                traj = self.expert_plan(rng)
            except RuntimeError as e:
                self.get_logger().warn(str(e)); continue
            imgs, states, actions, vids = [], [], [], []
            block0 = self.block_xyz.copy()
            q = None
            for a in traj:
                fid, img, q = self.wait_frame(fid)
                imgs.append(img); states.append(q); actions.append(a.astype(np.float32))
                if n_ok < self.args.record_video and self.video_frame is not None:
                    vids.append(self.video_frame.copy())
                self.command(a)
            for _ in range(5):              # let the released block settle
                fid, _, q = self.wait_frame(fid)
            success, real = self.real_success(q)
            ep += 1
            self.get_logger().info(f"episode {ep}: steps={len(traj)} success={success} block={np.round(real, 3) if real is not None else None} ({n_ok + success}/{self.args.episodes})")
            if success:
                np.savez_compressed(os.path.join(self.args.out, f"ep_{n_ok:04d}.npz"),
                                    image=np.stack(imgs), state=np.stack(states),
                                    action=np.stack(actions), block0=block0)
                if vids:
                    np.savez_compressed(os.path.join(self.args.out, f"demo_video_{n_ok:04d}.npz"),
                                        video=np.stack(vids), image=np.stack(imgs))
                n_ok += 1
        self.get_logger().info(f"collect finished: {n_ok} successful episodes out of {ep} attempts")

    def run_eval(self):
        rng = np.random.default_rng(self.args.seed)
        os.makedirs(self.args.out, exist_ok=True)
        sock = socket.create_connection(("127.0.0.1", self.args.port))
        results = []
        for ep in range(self.args.episodes):
            fid = self.reset(rng)
            send_msg(sock, {"cmd": "reset"}); recv_msg(sock)
            block0 = self.block_xyz.copy()
            imgs, vids, states, actions = [], [], [], []
            closed_seen, done_at = False, None
            for t in range(self.args.max_steps):
                fid, img, q = self.wait_frame(fid)
                send_msg(sock, {"cmd": "act", "image": img, "state": q})
                a = np.asarray(recv_msg(sock)["action"], dtype=np.float32)
                self.command(a)
                imgs.append(img); states.append(q); actions.append(a)
                if self.video_frame is not None:
                    vids.append(self.video_frame.copy())
                # episode end: the policy closed the gripper, then re-opened it above the tray
                if a[5] < 0.008:
                    closed_seen = True
                tool = self.tool_xyz(q)
                if done_at is None and closed_seen and a[5] > 0.02 and np.all(np.abs(tool[:2] - TRAY_XY) < 0.06):
                    done_at = t
                if done_at is not None and t - done_at > 15:
                    break
            for _ in range(3):
                fid, _, q = self.wait_frame(fid)
            success, real = self.real_success(q)
            results.append(success)
            self.get_logger().info(f"[eval] episode {ep}: success={success} steps={t + 1} block={np.round(real, 3) if real is not None else None}")
            np.savez_compressed(os.path.join(self.args.out, f"eval_{ep:03d}.npz"),
                                image=np.stack(imgs), video=np.stack(vids) if vids else np.zeros((0,)),
                                state=np.stack(states), action=np.stack(actions),
                                block0=block0, success=success)
        self.get_logger().info(f"[eval] success rate {np.mean(results):.2f} ({sum(results)}/{len(results)})")
        sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["collect", "eval"], required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max_tries", type=int, default=1000, help="collect: give up after this many attempts")
    ap.add_argument("--max_steps", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--record_video", type=int, default=3, help="collect: store presentation video for the first N episodes")
    args = ap.parse_args()
    rclpy.init()
    node = TaskNode(args)
    try:
        node.run_collect() if args.mode == "collect" else node.run_eval()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
