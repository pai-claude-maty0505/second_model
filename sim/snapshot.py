#!/usr/bin/env python3
"""Sanity check: move the arm to HOME, then save one frame from each camera (as .ppm, numpy only)."""
import argparse, os, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "finger_left", "finger_right"]
HOME = [0.0, -1.5, 0.6, 1.6, 0.0, 0.03, 0.03]   # last two: finger opening (m), both open


def save_ppm(path, img):
    with open(path, "wb") as f:
        f.write(b"P6 %d %d 255\n" % (img.shape[1], img.shape[0]))
        f.write(np.ascontiguousarray(img[..., :3]).tobytes())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rclpy.init(); n = Node("snapshot")
    pubs = {j: n.create_publisher(Float64, f"/so101/{j}/cmd", 10) for j in JOINTS}
    imgs, js = {}, {}
    for cam in ["top_cam", "video_cam"]:
        n.create_subscription(Image, f"/{cam}/image",
                              lambda m, c=cam: imgs.__setitem__(c, np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)),
                              qos_profile_sensor_data)
    n.create_subscription(JointState, "/so101/joint_states", lambda m: js.update(zip(m.name, m.position)), 10)
    t0 = time.time()
    while time.time() - t0 < 6:
        for j, v in zip(JOINTS, HOME):
            pubs[j].publish(Float64(data=v))
        rclpy.spin_once(n, timeout_sec=0.1)
    imgs.clear()
    t0 = time.time()
    while len(imgs) < 2 and time.time() - t0 < 20:
        rclpy.spin_once(n, timeout_sec=0.1)
    print("joint states:", {k: round(v, 3) for k, v in js.items()})
    print("commanded   :", dict(zip(JOINTS, HOME)))
    for c, im in imgs.items():
        save_ppm(os.path.join(a.out, f"{c}.ppm"), im); print("saved", c, im.shape)
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
