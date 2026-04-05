"""
HybridMatt Inference Script.

Usage examples:

  # Matte a video file:
  python inference.py --input video.mp4 --output-dir output/ --checkpoint checkpoints/stage1/best.pth

  # Matte a folder of images (processed as a sequence):
  python inference.py --input frames/ --output-dir output/ --checkpoint checkpoints/stage1/best.pth

  # Process at half resolution (faster, less VRAM):
  python inference.py --input video.mp4 --output-dir output/ --checkpoint best.pth --downsample-ratio 0.5

Outputs saved to output/:
  - alpha.mp4      — the alpha matte (greyscale, white = foreground)
  - composite.mp4  — the person composited over a green background
  - fgr.mp4        — the raw foreground colour output
"""

import argparse
import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from model import HybridMattingNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='Path to input video file or folder of images.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save output videos.')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained model checkpoint (.pth).')
    parser.add_argument('--downsample-ratio', type=float, default=1.0,
                        help='Process at this fraction of input resolution. E.g. 0.5 = half res.')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda or cpu. Auto-detected if not set.')
    parser.add_argument('--output-fps', type=int, default=30,
                        help='FPS for output videos.')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    model = HybridMattingNetwork(variant='mobilenetv3', pretrained_backbone=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    print(f'Loaded model from {checkpoint_path}')
    return model


def load_frames(input_path):
    """Returns a list of PIL Images."""
    if os.path.isdir(input_path):
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = sorted([f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in exts])
        frames = []
        for f in files:
            with Image.open(os.path.join(input_path, f)) as img:
                frames.append(img.convert('RGB').copy())
        return frames
    else:
        # Video file
        cap = cv2.VideoCapture(input_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames


def make_video_writer(path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


@torch.no_grad()
def run_inference(model, frames, device, downsample_ratio, output_dir, fps):
    os.makedirs(output_dir, exist_ok=True)

    H, W = frames[0].size[1], frames[0].size[0]
    writer_pha = make_video_writer(os.path.join(output_dir, 'alpha.mp4'), fps, W, H)
    writer_com = make_video_writer(os.path.join(output_dir, 'composite.mp4'), fps, W, H)
    writer_fgr = make_video_writer(os.path.join(output_dir, 'fgr.mp4'), fps, W, H)

    # Recurrent hidden states — start as None, updated each frame
    r1 = r2 = r3 = r4 = None

    for frame in tqdm(frames, desc='Processing frames', dynamic_ncols=True):
        src = to_tensor(frame).unsqueeze(0).to(device)   # [1, 3, H, W]

        fgr, pha, r1, r2, r3, r4 = model(src, r1, r2, r3, r4, downsample_ratio=downsample_ratio)

        # Convert to numpy uint8 for writing
        pha_np = (pha[0, 0].cpu().numpy() * 255).astype(np.uint8)
        fgr_np = (fgr[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Alpha: greyscale → BGR
        pha_bgr = cv2.cvtColor(pha_np, cv2.COLOR_GRAY2BGR)

        # Composite over green background
        green = np.zeros_like(fgr_np)
        green[:, :, 1] = 177   # green channel
        alpha_3ch = np.stack([pha_np, pha_np, pha_np], axis=2).astype(np.float32) / 255.0
        composite = (fgr_np.astype(np.float32) * alpha_3ch + green.astype(np.float32) * (1 - alpha_3ch)).astype(np.uint8)

        writer_pha.write(pha_bgr)
        writer_com.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        writer_fgr.write(cv2.cvtColor(fgr_np, cv2.COLOR_RGB2BGR))

    writer_pha.release()
    writer_com.release()
    writer_fgr.release()

    print(f'\nSaved outputs to {output_dir}:')
    print('  alpha.mp4     — alpha matte (white = foreground)')
    print('  composite.mp4 — composited on green background')
    print('  fgr.mp4       — foreground colour estimate')


if __name__ == '__main__':
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = load_model(args.checkpoint, device)
    print(f'Loading frames from {args.input}...')
    frames = load_frames(args.input)
    print(f'Loaded {len(frames)} frames ({frames[0].size[0]}x{frames[0].size[1]})')

    run_inference(model, frames, device, args.downsample_ratio, args.output_dir, args.output_fps)
