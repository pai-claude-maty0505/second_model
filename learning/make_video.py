"""Compose the 1-3 minute submission video (1x speed, 10 fps simulation frames upsampled to 30 fps).

Sections: title -> pipeline -> (1) expert data collection -> (2) ACT training curve
          -> (3) autonomous execution by the learned policy -> results.
"""
import argparse, glob, io, os, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

W, H, FPS = 1280, 720, 30
SIM_FPS = 10
BG = (18, 24, 38)
ACCENT = (255, 196, 64)
FONT_CANDIDATES = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def card(lines, sub=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = H // 2 - 60 * len(lines) // 2 - (40 if sub else 0)
    for i, (txt, size, col) in enumerate(lines):
        f = font(size)
        w = d.textlength(txt, font=f)
        d.text(((W - w) / 2, y), txt, font=f, fill=col)
        y += int(size * 1.5)
    if sub:
        f = font(26)
        for s in sub:
            w = d.textlength(s, font=f)
            d.text(((W - w) / 2, y + 10), s, font=f, fill=(200, 210, 230))
            y += 40
    return np.asarray(im)


def scene_frame(video, policy_img, title, info, badge=None):
    im = Image.new("RGB", (W, H), BG)
    v = Image.fromarray(video).resize((880, 660))
    im.paste(v, (20, 40))
    p = Image.fromarray(policy_img).resize((340, 255), Image.NEAREST)
    im.paste(p, (920, 80))
    d = ImageDraw.Draw(im)
    d.text((20, 4), title, font=font(28), fill=ACCENT)
    d.text((920, 44), "ポリシー入力カメラ (160x120)", font=font(20), fill=(200, 210, 230))
    y = 360
    for line in info:
        d.text((920, y), line, font=font(22), fill=(230, 230, 230)); y += 34
    if badge:
        txt, col = badge
        d.rounded_rectangle((920, 620, 1260, 690), radius=12, fill=col)
        d.text((940, 632), txt, font=font(32), fill=(255, 255, 255))
    d.text((W - 230, H - 30), "1x speed (sim time)", font=font(18), fill=(150, 160, 180))
    return np.asarray(im)


def hold(writer, frame, seconds):
    for _ in range(int(seconds * FPS)):
        writer.append_data(frame)


def sim_clip(writer, video, policy, title, info_fn, badge_fn=lambda t: None):
    rep = FPS // SIM_FPS
    for t in range(len(video)):
        f = scene_frame(video[t], policy[min(t, len(policy) - 1)], title, info_fn(t), badge_fn(t))
        for _ in range(rep):
            writer.append_data(f)


def loss_plot(log_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps, losses = [], []
    if log_path and os.path.exists(log_path):
        for line in open(log_path, errors="ignore"):
            m = re.search(r"step:(\S+).*?loss:([\d.]+)", line)
            if m:
                s = m.group(1)
                mul = 1000 if s.endswith("K") else 1
                steps.append(float(s.rstrip("K")) * mul); losses.append(float(m.group(2)))
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor("#121826"); ax.set_facecolor("#121826")
    if steps:
        ax.plot(steps, losses, color="#ffc440", lw=2.5)
        ax.set_yscale("log")
    ax.set_xlabel("training step", color="w", fontsize=14); ax.set_ylabel("ACT loss (L1 + KL)", color="w", fontsize=14)
    ax.tick_params(colors="w"); [s.set_color("#666") for s in ax.spines.values()]
    ax.grid(alpha=0.2)
    ax.set_title("(2) LeRobot ACT training (Colab GPU)", color="w", fontsize=18)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return np.asarray(Image.open(buf).convert("RGB").resize((W, H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--train_log", default=None)
    ap.add_argument("--out", default="PAI_final.mp4")
    ap.add_argument("--n_demo", type=int, default=2)
    ap.add_argument("--n_eval", type=int, default=6)
    ap.add_argument("--only_success", action="store_true", help="show only successful trials")
    a = ap.parse_args()

    demos = sorted(glob.glob(os.path.join(a.raw, "demo_video_*.npz")))[: a.n_demo]
    n_raw = len(glob.glob(os.path.join(a.raw, "ep_*.npz")))
    evals = sorted(glob.glob(os.path.join(a.eval, "eval_*.npz")))
    succ_all = [bool(np.load(f)["success"]) for f in evals]
    # show successes first, but keep at least one failure if there is one (honest results)
    order = [f for f, s in zip(evals, succ_all) if s][: a.n_eval - 1] + [f for f, s in zip(evals, succ_all) if not s][:1]
    order = order[: a.n_eval] if order else evals[: a.n_eval]
    if a.only_success:
        order = [f for f, s in zip(evals, succ_all) if s][: a.n_eval]

    w = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                           pixelformat="yuv420p")
    hold(w, card([("Physical AI 応用1 最終課題", 44, ACCENT),
                  ("SO-101 ピック＆プレース（第2版）：2本指・摩擦把持", 40, (255, 255, 255))],
                 ["ROS 2  ×  Gazebo  ×  LeRobot (ACT)  ×  Google Colab", "剛体ブロックを、両側が開閉する2本指で挟み、摩擦力だけで運ぶ"]), 4)
    hold(w, card([("パイプライン", 40, ACCENT)],
                 ["(1) Gazebo上でIKエキスパートが自動デモ → ROS 2で画像・関節角を記録",
                  f"      → LeRobotデータセット化（{n_raw} エピソード）",
                  "(2) LeRobot ACT（Transformer行動チャンク）をColab GPUで学習",
                  "(3) 学習済みポリシーが カメラ画像＋関節角 だけで 自律実行",
                  "      （ROS 2ノード ⇔ TCP推論サーバ）"]), 7)
    for i, f in enumerate(demos):
        d = np.load(f)
        sim_clip(w, d["video"], d["image"], f"(1) 学習データ収集：スクリプトIKエキスパート  demo {i + 1}",
                 lambda t, n=len(d["video"]): ["行動 = IK軌道（関節角5＋指の開き幅）", f"step {t + 1}/{n}", "10 Hz で画像・状態・行動を記録"])
    if a.train_log:
        hold(w, loss_plot(a.train_log), 6)
    n_ok = 0
    for i, f in enumerate(order):
        d = np.load(f)
        vid = d["video"] if d["video"].ndim == 4 and len(d["video"]) else np.stack([np.asarray(Image.fromarray(x).resize((640, 480))) for x in d["image"]])
        s = bool(d["success"])
        n_ok += s
        done = len(vid) - 16 if s else None
        sim_clip(w, vid, d["image"], f"(3) 学習済みACTポリシーによる自律実行  trial {i + 1}",
                 lambda t, n=len(vid): ["入力: カメラ画像 + 関節角", "出力: 関節目標＋指の開き（チャンク）", f"step {t + 1}/{n}"],
                 lambda t, s=s, done=done: (("SUCCESS", (40, 160, 80)) if s and t >= done else
                                            (("FAILED", (190, 50, 50)) if (not s and t >= len(vid) - 10) else None)))
    rate = np.mean(succ_all) if succ_all else 0
    hold(w, card([("結果", 44, ACCENT), (f"成功率 {rate * 100:.0f}%  ({sum(succ_all)}/{len(succ_all)} 試行, ランダム配置)", 40, (255, 255, 255))],
                 [f"学習データ: エキスパートデモ {n_raw} エピソード", "物体位置はカメラ画像からポリシーが自ら推定", "把持は摩擦のみ（固定・吸着などの補助なし）"]), 5)
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
