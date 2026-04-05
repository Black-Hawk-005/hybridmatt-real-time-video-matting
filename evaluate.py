"""
HybridMatt Evaluation Script.

Evaluates the trained model on the VideoMatte240K test set.

Metrics:
  - MAD  : Mean Absolute Difference (alpha matte)
  - MSE  : Mean Squared Error (alpha matte)
  - dtSSD: Temporal coherence — mean squared difference of consecutive alpha differences

Usage:
  python evaluate.py --checkpoint checkpoints/stage2/step-1000.pth

  # Save sample visual outputs (first N frames of first clip):
  python evaluate.py --checkpoint checkpoints/stage2/step-1000.pth --save-samples output/samples --num-samples 20
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm

from model import HybridMattingNetwork
from training.train_config import DATA_PATHS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained model checkpoint (.pth).')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--downsample-ratio', type=float, default=1.0,
                        help='Process at this fraction of input resolution.')
    parser.add_argument('--save-samples', type=str, default=None,
                        help='Directory to save sample visual outputs.')
    parser.add_argument('--num-samples', type=int, default=30,
                        help='Number of sample frames to save.')
    parser.add_argument('--max-clips', type=int, default=None,
                        help='Limit evaluation to first N clips (for speed).')
    parser.add_argument('--max-frames-per-clip', type=int, default=100,
                        help='Max frames per clip to evaluate (default 100).')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    model = HybridMattingNetwork(variant='mobilenetv3', pretrained_backbone=False)
    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing keys (randomly initialized): {len(missing)}')
    if unexpected:
        print(f'  Unexpected keys (ignored): {len(unexpected)}')
    model.eval()
    model.to(device)
    return model


def get_test_clips(test_dir):
    """Returns list of (fgr_dir, pha_dir) for each test clip."""
    fgr_root = os.path.join(test_dir, 'fgr')
    pha_root = os.path.join(test_dir, 'pha')
    clips = sorted(os.listdir(fgr_root))
    return [(os.path.join(fgr_root, c), os.path.join(pha_root, c)) for c in clips]


def load_frame(fgr_path, pha_path):
    """Load a single frame's foreground and alpha."""
    fgr = Image.open(fgr_path).convert('RGB')
    pha = Image.open(pha_path).convert('L')
    return fgr, pha


def get_background(bg_dir, size):
    """Load a random background image and resize to match frame size."""
    bgs = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    bg_path = os.path.join(bg_dir, bgs[0])  # deterministic: always first
    bg = Image.open(bg_path).convert('RGB').resize(size, Image.BILINEAR)
    return to_tensor(bg)


@torch.no_grad()
def evaluate(model, device, args):
    test_dir = DATA_PATHS['videomatte']['valid']
    bg_dir = DATA_PATHS['background_images']['valid']
    clips = get_test_clips(test_dir)

    if args.max_clips:
        clips = clips[:args.max_clips]

    all_mad, all_mse, all_dtssd = [], [], []
    sample_count = 0
    save_dir = args.save_samples

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f'Evaluating {len(clips)} clips from {test_dir}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Downsample ratio: {args.downsample_ratio}')
    print()

    for clip_idx, (fgr_dir, pha_dir) in enumerate(clips):
        clip_name = os.path.basename(fgr_dir)
        frames = sorted([f for f in os.listdir(fgr_dir) if f.endswith('.jpg')])
        frames = frames[:args.max_frames_per_clip]

        # Get frame size from first frame
        first_fgr = Image.open(os.path.join(fgr_dir, frames[0])).convert('RGB')
        W, H = first_fgr.size
        bg_tensor = get_background(bg_dir, (W, H)).to(device)

        # Reset recurrent states for each clip
        r1 = r2 = r3 = r4 = None
        prev_pred_pha = None
        prev_true_pha = None

        clip_mad, clip_mse, clip_dtssd = [], [], []

        for frame_name in tqdm(frames, desc=f'Clip {clip_idx+1}/{len(clips)} ({clip_name})',
                               dynamic_ncols=True, leave=False):
            fgr_path = os.path.join(fgr_dir, frame_name)
            pha_path = os.path.join(pha_dir, frame_name)

            fgr_pil, pha_pil = load_frame(fgr_path, pha_path)
            fgr_t = to_tensor(fgr_pil).to(device)          # [3, H, W]
            true_pha = to_tensor(pha_pil).to(device)        # [1, H, W]

            # Composite: src = fgr * alpha + bg * (1 - alpha)
            src = fgr_t * true_pha + bg_tensor * (1 - true_pha)
            src = src.unsqueeze(0)  # [1, 3, H, W]

            # Run model
            pred_fgr, pred_pha, r1, r2, r3, r4 = model(
                src, r1, r2, r3, r4, downsample_ratio=args.downsample_ratio)

            pred_pha_np = pred_pha[0, 0].cpu().numpy()   # [H, W]
            true_pha_np = true_pha[0].cpu().numpy()       # [H, W]

            # MAD
            mad = np.mean(np.abs(pred_pha_np - true_pha_np))
            clip_mad.append(mad)

            # MSE
            mse = np.mean((pred_pha_np - true_pha_np) ** 2)
            clip_mse.append(mse)

            # dtSSD (temporal stability)
            if prev_pred_pha is not None:
                pred_diff = pred_pha_np - prev_pred_pha
                true_diff = true_pha_np - prev_true_pha
                dtssd = np.mean((pred_diff - true_diff) ** 2)
                clip_dtssd.append(dtssd)

            prev_pred_pha = pred_pha_np
            prev_true_pha = true_pha_np

            # Save sample outputs
            if save_dir and sample_count < args.num_samples:
                # Save: input, predicted alpha, ground truth alpha, composite on green
                src_np = (src[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pred_a = (pred_pha_np * 255).astype(np.uint8)
                true_a = (true_pha_np * 255).astype(np.uint8)
                pred_fgr_np = (pred_fgr[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                # Green-screen composite
                green = np.zeros_like(pred_fgr_np)
                green[:, :, 1] = 177
                alpha_3ch = np.stack([pred_pha_np]*3, axis=2)
                composite = (pred_fgr_np.astype(np.float32) * alpha_3ch +
                             green.astype(np.float32) * (1 - alpha_3ch)).astype(np.uint8)

                prefix = f'{sample_count:04d}'
                Image.fromarray(src_np).save(os.path.join(save_dir, f'{prefix}_input.png'))
                Image.fromarray(pred_a).save(os.path.join(save_dir, f'{prefix}_pred_alpha.png'))
                Image.fromarray(true_a).save(os.path.join(save_dir, f'{prefix}_true_alpha.png'))
                Image.fromarray(composite).save(os.path.join(save_dir, f'{prefix}_composite.png'))
                sample_count += 1

        avg_mad = np.mean(clip_mad)
        avg_mse = np.mean(clip_mse)
        avg_dtssd = np.mean(clip_dtssd) if clip_dtssd else 0.0
        all_mad.append(avg_mad)
        all_mse.append(avg_mse)
        all_dtssd.append(avg_dtssd)

        print(f'  {clip_name}: MAD={avg_mad:.4f}, MSE={avg_mse:.6f}, dtSSD={avg_dtssd:.6f}')

    # Overall metrics
    overall_mad = np.mean(all_mad)
    overall_mse = np.mean(all_mse)
    overall_dtssd = np.mean(all_dtssd)

    print(f'\n{"="*60}')
    print(f'OVERALL RESULTS ({len(clips)} clips)')
    print(f'{"="*60}')
    print(f'  MAD   : {overall_mad:.4f}')
    print(f'  MSE   : {overall_mse:.6f}')
    print(f'  dtSSD : {overall_dtssd:.6f}')
    print(f'{"="*60}')

    return {'mad': overall_mad, 'mse': overall_mse, 'dtssd': overall_dtssd}


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}')

    model = load_model(args.checkpoint, device)
    results = evaluate(model, device, args)
