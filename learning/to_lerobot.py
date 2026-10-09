"""Convert raw episodes (*.npz from so101_task_node.py --mode collect) to a LeRobotDataset (v3 format)."""
import argparse, glob, os, shutil
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper_opening_m"]
TASK = "Pick up the red block and place it in the blue tray"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--root", required=True, help="output dataset directory")
    ap.add_argument("--repo_id", default="local/so101_gz_parallel_grasp")
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.raw, "ep_*.npz")))
    assert files, f"no episodes in {a.raw}"
    h, w, _ = np.load(files[0])["image"].shape[1:]
    features = {
        "observation.images.top": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    }
    if os.path.exists(a.root):
        shutil.rmtree(a.root)
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=a.fps, features=features, root=a.root,
                               robot_type="so101_gazebo", use_videos=False, image_writer_threads=4)
    n_frames = 0
    for f in files:
        d = np.load(f)
        for img, s, act in zip(d["image"], d["state"], d["action"]):
            ds.add_frame({"observation.images.top": img, "observation.state": s.astype(np.float32),
                          "action": act.astype(np.float32), "task": TASK})
        ds.save_episode()
        n_frames += len(d["action"])
    ds.finalize()
    print(f"dataset written to {a.root}: {len(files)} episodes, {n_frames} frames")


if __name__ == "__main__":
    main()
