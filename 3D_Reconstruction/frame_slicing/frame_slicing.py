#!/usr/bin/env python3
"""
裁剪视频并按固定帧步长导出图像帧

示例：
python cut_and_extract.py input.mp4 --start 75.3 --out_dir frames --step 30
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path

import cv2
from moviepy.editor import VideoFileClip
from tqdm import tqdm


# ---------- 工具函数 ----------
def parse_time(ts: str) -> float:
    """支持 'SS(.ms)' 或 'HH:MM:SS(.ms)' 两种写法，返回秒数(float)。"""
    if ":" not in ts:
        return float(ts)
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def ffmpeg_fast_trim(src: Path, dst: Path, start_sec: float) -> bool:
    """若系统已装 ffmpeg，用 -c copy 无重编码裁剪，加速处理."""
    cmd = ["ffmpeg", "-y",
           "-ss", f"{start_sec:.3f}",
           "-i", str(src),
           "-c", "copy",
           str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False  # 回退 MoviePy


def extract_frames(clip: VideoFileClip, out_dir: Path, step: int):
    """
    按 step (e.g. 30) 保存帧。frame_000000.png 对应裁剪后第 0 帧，
    frame_000001.png 对应第 step 帧，以此类推。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total_frames = int(clip.fps * clip.duration)
    saved = 0

    for idx, frame in enumerate(tqdm(clip.iter_frames(),
                                     total=total_frames,
                                     desc=f"Saving every {step}th frame")):
        if idx % step:         # 不是目标帧，跳过
            continue

        cv2.imwrite(str(out_dir / f"frame_{saved:06d}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        saved += 1


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="Cut video and dump frames")
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("--start", required=True,
                        help="start time (seconds or HH:MM:SS[.ms])")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="directory to save frames")
    parser.add_argument("--save_trim", type=Path,
                        help="optional path to save trimmed video")
    parser.add_argument("--step", type=int, default=30,
                        help="frame interval to save (default: 30)")
    args = parser.parse_args()

    start_sec = parse_time(args.start)
    step = max(1, args.step)  # 防止传入 0 或负数

    # (可选)FFmpeg 极速裁剪
    if args.save_trim and ffmpeg_fast_trim(args.video, args.save_trim, start_sec):
        src_clip = VideoFileClip(str(args.save_trim))
    else:
        full_clip = VideoFileClip(str(args.video))
        src_clip = full_clip.subclip(start_sec)

        if args.save_trim:
            src_clip.write_videofile(str(args.save_trim),
                                     codec="libx264", audio_codec="aac")

    extract_frames(src_clip, args.out_dir, step)


if __name__ == "__main__":
    sys.exit(main())
