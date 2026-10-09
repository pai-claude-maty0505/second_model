"""Create the second-model files that are small edits of the first model's files.

Run in Colab:  python derive_second.py --first <PAI_final_code dir> --out <PAI_second_code dir>
  - copies so101_kin.py / policy_server.py unchanged
  - bridge.yaml   : gripper -> finger_left + finger_right, no attach/detach topics
  - snapshot.py   : finger joints
  - to_lerobot.py : dataset name / joint name of the 6th dimension
  - make_video.py : titles + --only_success option
  - PAI_second_SO101_parallel_gripper.ipynb : notebook derived from PAI_final_v3.ipynb
"""
import argparse, json, os, shutil

FIRST_REPO = "https://github.com/pai-claude-maty0505/second_model"


def rep(s, a, b):
    assert a in s, "pattern not found: " + a[:70]
    return s.replace(a, b)


def derive_code(first, out):
    for f in ["so101_kin.py", "policy_server.py"]:
        shutil.copy(os.path.join(first, f), os.path.join(out, f))
    b = open(os.path.join(first, "bridge.yaml")).read()
    blk = ('- ros_topic_name: "/so101/gripper/cmd"\n  gz_topic_name: "/so101/gripper/cmd"\n'
           '  ros_type_name: "std_msgs/msg/Float64"\n  gz_type_name: "gz.msgs.Double"\n  direction: ROS_TO_GZ\n')
    b = rep(b, blk, blk.replace("gripper", "finger_left") + blk.replace("gripper", "finger_right"))
    b = b[:b.index('- ros_topic_name: "/block/attach"')].rstrip() + "\n"
    open(os.path.join(out, "bridge.yaml"), "w").write(b)

    s = open(os.path.join(first, "snapshot.py")).read()
    s = rep(s, 'JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]\nHOME = [0.0, -1.5, 0.6, 1.6, 0.0, 1.0]',
            'JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "finger_left", "finger_right"]\n'
            'HOME = [0.0, -1.5, 0.6, 1.6, 0.0, 0.03, 0.03]   # last two: finger opening (m), both open')
    open(os.path.join(out, "snapshot.py"), "w").write(s)

    t = open(os.path.join(first, "to_lerobot.py")).read()
    t = rep(t, '"wrist_roll", "gripper"]', '"wrist_roll", "gripper_opening_m"]').replace("so101_gz_pick_place", "so101_gz_parallel_grasp")
    open(os.path.join(out, "to_lerobot.py"), "w").write(t)

    v = open(os.path.join(first, "make_video.py"), encoding="utf-8").read()
    v = rep(v, '("SO-101 ピック＆プレース：模倣学習の全工程", 40, (255, 255, 255))]',
            '("SO-101 ピック＆プレース（第2版）：2本指・摩擦把持", 40, (255, 255, 255))]')
    v = rep(v, '["ROS 2  ×  Gazebo  ×  LeRobot (ACT)  ×  Google Colab"]',
            '["ROS 2  ×  Gazebo  ×  LeRobot (ACT)  ×  Google Colab", "剛体ブロックを、両側が開閉する2本指で挟み、摩擦力だけで運ぶ"]')
    v = rep(v, '"行動 = IK軌道（関節角6次元）"', '"行動 = IK軌道（関節角5＋指の開き幅）"')
    v = rep(v, '"出力: 次の関節目標（チャンク）"', '"出力: 関節目標＋指の開き（チャンク）"')
    v = rep(v, '"物体位置はカメラ画像からポリシーが自ら推定"]',
            '"物体位置はカメラ画像からポリシーが自ら推定", "把持は摩擦のみ（固定・吸着などの補助なし）"]')
    old = ("    order = [f for f, s in zip(evals, succ_all) if s][: a.n_eval - 1] + [f for f, s in zip(evals, succ_all) if not s][:1]\n"
           "    order = order[: a.n_eval] if order else evals[: a.n_eval]\n")
    v = rep(v, old, old + "    if a.only_success:\n        order = [f for f, s in zip(evals, succ_all) if s][: a.n_eval]\n")
    v = rep(v, '    ap.add_argument("--n_eval", type=int, default=6)\n',
            '    ap.add_argument("--n_eval", type=int, default=6)\n'
            '    ap.add_argument("--only_success", action="store_true", help="show only successful trials")\n')
    open(os.path.join(out, "make_video.py"), "w", encoding="utf-8").write(v)


GRASP_MD = """### 2-5. 把持テスト：エキスパートで 3 回だけ実行（摩擦で持ち上がるかの確認）
両側の指でブロックを挟み、摩擦だけで持ち上げてトレイに入れられるかを確認します。`success=True` が出ればデータ収集に進めます。
下に「掴む直前・挟んだ直後・持ち上げ中・トレイ上・解放後」の記録用カメラ画像を並べます。"""

GRASP_CODE = r"""assert RENDER_MODE and gz_proc.poll() is None and bridge_proc.poll() is None, '2-3 を実行してください'
subprocess.run('rm -rf /content/data/grasptest', shell=True)
p = run_ros(f'cd {WS}/sim && timeout 900 {PY_ROS} -u so101_task_node.py --mode collect --urdf {URDF} '
            f'--out /content/data/grasptest --episodes 3 --max_tries 3 --record_video 3 --seed 7', show=False)
print('\n'.join(l for l in (p.stdout + p.stderr).splitlines() if 'episode' in l or 'rror' in l)[-3000:])
import numpy as np, glob
from PIL import Image; import matplotlib.pyplot as plt
vids = sorted(glob.glob('/content/data/grasptest/demo_video_*.npz'))
if vids:
    v = np.load(vids[0])['video']
    idx = [22, 28, 36, 50, 62, 70]
    fig, ax = plt.subplots(1, len(idx), figsize=(20, 3.5))
    for a, i in zip(ax, idx):
        a.imshow(v[min(i, len(v) - 1)]); a.set_title(f'step {i}'); a.axis('off')
    plt.show()
else:
    print('成功エピソードがありません（ログを確認）')"""


def _cell(kind, src):
    lines = src.split("\n")
    c = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def derive_notebook(v3_path, out_path):
    nb = json.load(open(v3_path, encoding="utf-8"))
    R = [
        ("""# Physical AI 応用1 最終課題：SO-101 ピック＆プレースの模倣学習（ROS 2 × Gazebo × LeRobot）v3

**タスク**：テーブル上のランダムな位置に置かれた赤いブロックを、SO-101 アームで掴んで青いトレイに入れる。
""", """# SO-101 ピック＆プレースの模倣学習 第2版：2本指グリッパ × 摩擦把持（ROS 2 × Gazebo × LeRobot）

**タスク**：テーブル上のランダムな位置に置かれた赤いブロック（剛体）を、SO-101 アームで掴んで青いトレイに入れる。

**第1版からの変更点**
- ハンド：SO-101 の片側だけが動くジョーを、**両側の指が開いて閉じる平行2本指グリッパ**に置き換え
- 把持：ブロックは指と衝突する**剛体**。固定（DetachableJoint）などの補助は使わず、**指で挟んだ摩擦力だけ**で持ち上げて運ぶ
"""),
        ("SAVE_DIR = '/content/drive/MyDrive/PAI_final'", "SAVE_DIR = '/content/drive/MyDrive/PAI_second'"),
        ("WS = '/content/pai_final'", "WS = '/content/pai_second'"),
        ("""### 2-1. ソースコード（Google Drive の `PAI_final_code` フォルダからコピー）
- `so101_kin.py`：URDF を直接読む順運動学・逆運動学（numpy のみ）
- `build_sim.py`：URDF→SDF 変換と後処理（関節位置コントローラ、関節状態パブリッシャ、把持アシスト用 DetachableJoint）、ワールド生成""",
         """### 2-1. ソースコード（Google Drive の `PAI_second_code` フォルダ、無ければ GitHub から取得）
- `so101_kin.py`：URDF を直接読む順運動学・逆運動学（numpy のみ）
- `build_sim.py`：URDF の改造（元のジョーを外し、掌＋左右の直動指を追加）、URDF→SDF 変換と後処理（衝突・摩擦の設定、関節コントローラ、関節状態パブリッシャ）、ワールド生成"""),
        ("""cands = (glob.glob('/content/drive/MyDrive/PAI_final_code') or glob.glob('/content/drive/MyDrive/*/PAI_final_code') or glob.glob('/content/drive/MyDrive/*/*/PAI_final_code') or glob.glob('/content/drive/MyDrive/*/*/*/PAI_final_code') or glob.glob('/content/drive/MyDrive/*/*/*/*/PAI_final_code'))
assert cands, 'Drive に PAI_final_code フォルダが見つかりません'
CODE_DIR = cands[0]; print('code from', CODE_DIR)""",
         f"""NAME = 'PAI_second_code'
cands = (glob.glob(f'/content/drive/MyDrive/{{NAME}}') or glob.glob(f'/content/drive/MyDrive/*/{{NAME}}') or glob.glob(f'/content/drive/MyDrive/*/*/{{NAME}}'))
if not cands:
    # Drive に無ければ GitHub の公開リポジトリから取得し、フラットなフォルダに並べる
    subprocess.run(['git', 'clone', '--depth', '1', '{FIRST_REPO}', '/content/second_model'], check=True)
    os.makedirs(f'/content/{{NAME}}', exist_ok=True)
    for sub in ['sim', 'learning', '.']:
        for f in glob.glob(f'/content/second_model/{{sub}}/*'):
            if os.path.isfile(f): shutil.copy(f, f'/content/{{NAME}}/')
    cands = [f'/content/{{NAME}}']
CODE_DIR = cands[0]; print('code from', CODE_DIR)"""),
        ("p = run_ros(f'cd {WS}/sim && {PY_ROS} build_sim.py --urdf {URDF} --out {WS}/sim/gen')",
         "p = run_ros(f'cd {WS}/sim && {PY_ROS} build_sim.py --urdf {URDF} --out {WS}/sim/gen')\n"
         "print(open(f'{WS}/sim/gen/so101_model/model.sdf').read().count('collide_bitmask'), 'collisions configured')"),
        ("--out /content/data/raw --episodes {N_EPISODES} --record_video 3'",
         "--out /content/data/raw --episodes {N_EPISODES} --max_tries 120 --record_video 3'"),
        ("plt.title('episode 0 actions (joint targets, rad)')", "plt.title('episode 0 actions (joint targets rad / finger opening m)')"),
        ("- 入力：ポリシーカメラ画像（160×120）＋ 関節角 6 次元\n- 出力：今後 20 ステップ分の関節目標（action chunk）",
         "- 入力：ポリシーカメラ画像（160×120）＋ 状態 6 次元（関節角 5 ＋ 指の開き幅）\n- 出力：今後 20 ステップ分の関節目標＋指の開き幅（action chunk）"),
        ("TRAIN_STEPS = 20000   # T4 で約 40〜60 分。まず試すなら 5000 でも可", "TRAIN_STEPS = 10000   # T4 で約 70 分（第1版と同じ設定）"),
        ("{SAVE_DIR}/act_last", "{SAVE_DIR}/act_second_last"),
        ("--out /content/data/eval --episodes {N_EVAL} --seed 12345 --port 5555'",
         "--out /content/data/eval --episodes {N_EVAL} --seed 12345 --port 5555 --max_steps 130'"),
        ("""!python {WS}/learning/make_video.py --raw /content/data/raw --eval /content/data/eval --train_log /content/train.log --out /content/PAI_final.mp4
!ffprobe -v error -show_entries format=duration -of csv=p=0 /content/PAI_final.mp4
!cp /content/PAI_final.mp4 {SAVE_DIR}/ && echo saved to Drive
from IPython.display import Video
Video('/content/PAI_final.mp4', embed=True, width=800)""",
         """# 全体版（成功例＋失敗例 1 つ）と、成功例だけの版の 2 本を作る
!python {WS}/learning/make_video.py --raw /content/data/raw --eval /content/data/eval --train_log /content/train.log --out /content/PAI_second.mp4
!python {WS}/learning/make_video.py --raw /content/data/raw --eval /content/data/eval --train_log /content/train.log --out /content/PAI_second_success.mp4 --only_success --n_eval 8
!ffprobe -v error -show_entries format=duration -of csv=p=0 /content/PAI_second.mp4
!ffprobe -v error -show_entries format=duration -of csv=p=0 /content/PAI_second_success.mp4
!cp /content/PAI_second.mp4 /content/PAI_second_success.mp4 /content/train.log /content/eval.log /content/collect.log {SAVE_DIR}/ && echo saved to Drive
from IPython.display import Video
Video('/content/PAI_second_success.mp4', embed=True, width=800)"""),
        ("`MyDrive/PAI_final/pai_final_code.zip`", "`MyDrive/PAI_second/pai_second_code.zip`"),
        ('zip -qr {SAVE_DIR}/pai_final_code.zip pai_final -x "pai_final/sim/gen/*"',
         'zip -qr {SAVE_DIR}/pai_second_code.zip pai_second -x "pai_second/sim/gen/*"'),
    ]
    cells, used = [], set()
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and src.lstrip().startswith(("# 確認用", "# 成果物", "# 成功例", "# 診断")):
            continue                                     # temporary cells added during the first run
        for i, (a, b) in enumerate(R):
            if a in src:
                src = src.replace(a, b); used.add(i)
        src = src.replace("so101_gz_pick_place", "so101_gz_parallel_grasp").replace("act_so101", "act_second")
        cells.append(_cell(c["cell_type"], src))
        if c["cell_type"] == "code" and "snapshot.py --out /content/snap" in src:
            cells.append(_cell("markdown", GRASP_MD))
            cells.append(_cell("code", GRASP_CODE))
    missing = [R[i][0][:60] for i in range(len(R)) if i not in used]
    assert not missing, f"notebook patterns not found: {missing}"
    nb["cells"] = cells
    json.dump(nb, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("notebook:", out_path, len(cells), "cells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True, help="first model code folder (PAI_final_code)")
    ap.add_argument("--v3", required=True, help="PAI_final_v3.ipynb")
    ap.add_argument("--out", required=True, help="second model code folder (PAI_second_code)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    derive_code(a.first, a.out)
    derive_notebook(a.v3, os.path.join(a.out, "PAI_second_SO101_parallel_gripper.ipynb"))
    print(sorted(os.listdir(a.out)))
