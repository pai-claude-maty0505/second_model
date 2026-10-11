"""Serve a trained LeRobot ACT policy over TCP for the ROS 2 node.

Why a server? ROS 2 (rclpy) is bound to the system Python, while LeRobot/torch
live in the Colab Python. Splitting "robot side" and "AI side" over a socket also
mirrors how inference servers are deployed on real robots.
"""
import argparse, pickle, socket, struct
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

TASK = "Pick up the red block and place it in the blue tray"


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_msg(sock):
    hdr = b""
    while len(hdr) < 4:
        c = sock.recv(4 - len(hdr))
        if not c:
            return None
        hdr += c
    n = struct.unpack("!I", hdr)[0]
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(min(1 << 20, n - len(buf)))
        if not c:
            return None
        buf += c
    return pickle.loads(bytes(buf))


class PolicyRunner:
    def __init__(self, ckpt, device, n_action_steps=None, temporal_ensemble=None):
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(ckpt).to(self.device).eval()
        cfg = self.policy.config
        if temporal_ensemble is not None:       # predict every step and average the overlapping chunks
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            cfg.temporal_ensemble_coeff = temporal_ensemble
            cfg.n_action_steps = 1
            self.policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble, cfg.chunk_size)
        elif n_action_steps is not None:        # re-plan more often than in training
            cfg.n_action_steps = n_action_steps
        print("inference: n_action_steps =", cfg.n_action_steps, "| temporal_ensemble =", cfg.temporal_ensemble_coeff, flush=True)
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=ckpt,
                                                       preprocessor_overrides={"device_processor": {"device": device}})

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def act(self, image, state):
        img = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)  # 1x3xHxW in [0,1]
        obs = {"observation.images.top": img,
               "observation.state": torch.from_numpy(np.asarray(state, np.float32)).unsqueeze(0),
               "task": [TASK]}
        obs = self.pre(obs)
        action = self.policy.select_action(obs)
        action = self.post(action)
        return action.squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".../checkpoints/last/pretrained_model")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_action_steps", type=int, default=None, help="override: execute this many steps of each chunk")
    ap.add_argument("--temporal_ensemble", type=float, default=None, help="ACT temporal ensembling coefficient (e.g. 0.01)")
    a = ap.parse_args()
    runner = PolicyRunner(a.ckpt, a.device, a.n_action_steps, a.temporal_ensemble)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", a.port))
    srv.listen(1)
    print(f"policy server ready on :{a.port} ({a.device})", flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            msg = recv_msg(conn)
            if msg is None:
                break
            if msg["cmd"] == "reset":
                runner.reset(); send_msg(conn, {"ok": True})
            elif msg["cmd"] == "act":
                send_msg(conn, {"action": runner.act(msg["image"], msg["state"])})
        conn.close()


if __name__ == "__main__":
    main()
