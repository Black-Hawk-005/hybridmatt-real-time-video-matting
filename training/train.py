"""
HybridMatt Training Script — Single GPU.

Two-stage training curriculum (adapted from RVM):

Stage 1 — short sequences, low resolution:
    python train.py --stage 1 --checkpoint-dir checkpoints/stage1

Stage 2 — longer sequences (better long-term memory):
    python train.py --stage 2 --checkpoint checkpoints/stage1/best.pth --checkpoint-dir checkpoints/stage2

To skip segmentation datasets (if you only have VideoMatte240K + backgrounds):
    python train.py --stage 1 --checkpoint-dir checkpoints/stage1 --no-seg
"""

import argparse
import os
import sys
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms.functional import center_crop
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path so we can import model/ and dataset/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset.videomatte import VideoMatteDataset, VideoMatteTrainAugmentation, VideoMatteValidAugmentation
from dataset.augmentation import TrainFrameSampler, ValidFrameSampler
from dataset.spd import SuperviselyPersonDataset
from dataset.coco import CocoPanopticDataset, CocoPanopticTrainAugmentation
from dataset.youtubevis import YouTubeVISDataset, YouTubeVISAugmentation
from model import HybridMattingNetwork
from training.train_config import DATA_PATHS
from training.train_loss import matting_loss, segmentation_loss


# ── Stage configs ─────────────────────────────────────────────────────────────

STAGES = {
    1: dict(resolution=256, seq_length=8, epochs=3,
            lr_backbone=1e-4, lr_aspp=2e-4, lr_decoder=2e-4),
    2: dict(resolution=256, seq_length=16, epochs=2,
            lr_backbone=5e-5, lr_aspp=1e-4, lr_decoder=1e-4),
}


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2],
                        help='Training stage (1 or 2). Start with 1.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a .pth checkpoint to resume from.')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory to save checkpoints.')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Tensorboard log directory. Defaults to checkpoint-dir/logs.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Dataloader workers. Reduce if you run out of RAM.')
    parser.add_argument('--no-seg', action='store_true',
                        help='Skip segmentation datasets (use if you only have VideoMatte240K).')
    parser.add_argument('--save-interval', type=int, default=500,
                        help='Save a checkpoint every N steps.')
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision (use if you get NaN losses).')
    parser.add_argument('--rvm-pretrained', type=str, default=None,
                        help='Path to RVM pretrained weights (.pth). Loads into HybridMatt with strict=False.')
    parser.add_argument('--freeze-rvm', action='store_true',
                        help='Freeze all layers except SpatialAttention (use with --rvm-pretrained).')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Stop training after this many steps (useful for fine-tuning).')
    return parser.parse_args()


# ── Main trainer ──────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, args):
        self.args = args
        self.cfg = STAGES[args.stage]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        self.log_dir = args.log_dir or os.path.join(args.checkpoint_dir, 'logs')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        self._init_datasets()
        self._init_model()

    def _init_datasets(self):
        size = (self.cfg['resolution'], self.cfg['resolution'])
        seq_len = self.cfg['seq_length']

        self.dataset_train = VideoMatteDataset(
            videomatte_dir=DATA_PATHS['videomatte']['train'],
            background_image_dir=DATA_PATHS['background_images']['train'],
            background_video_dir=DATA_PATHS['background_videos']['train'],
            size=self.cfg['resolution'],
            seq_length=seq_len,
            seq_sampler=TrainFrameSampler(),
            transform=VideoMatteTrainAugmentation(size))

        self.dataset_valid = VideoMatteDataset(
            videomatte_dir=DATA_PATHS['videomatte']['valid'],
            background_image_dir=DATA_PATHS['background_images']['valid'],
            background_video_dir=DATA_PATHS['background_videos']['valid'],
            size=self.cfg['resolution'],
            seq_length=seq_len,
            seq_sampler=ValidFrameSampler(),
            transform=VideoMatteValidAugmentation(size))

        self.dataloader_train = DataLoader(
            self.dataset_train, batch_size=self.args.batch_size,
            num_workers=self.args.num_workers, shuffle=True, pin_memory=True)

        self.dataloader_valid = DataLoader(
            self.dataset_valid, batch_size=self.args.batch_size,
            num_workers=self.args.num_workers, pin_memory=True)

        if not self.args.no_seg:
            seg_datasets = []
            try:
                seg_datasets.append(CocoPanopticDataset(
                    imgdir=DATA_PATHS['coco_panoptic']['imgdir'],
                    anndir=DATA_PATHS['coco_panoptic']['anndir'],
                    annfile=DATA_PATHS['coco_panoptic']['annfile'],
                    transform=CocoPanopticTrainAugmentation(size)))
                print('Loaded COCO Panoptic dataset.')
            except Exception as e:
                print(f'Skipping COCO (could not load): {e}')

            try:
                seg_datasets.append(SuperviselyPersonDataset(
                    imgdir=DATA_PATHS['spd']['imgdir'],
                    segdir=DATA_PATHS['spd']['segdir'],
                    transform=CocoPanopticTrainAugmentation(size)))
                print('Loaded Supervisely Person Dataset.')
            except Exception as e:
                print(f'Skipping SPD (could not load): {e}')

            if seg_datasets:
                self.dataset_seg_image = ConcatDataset(seg_datasets)
                self.dataloader_seg_image = DataLoader(
                    self.dataset_seg_image,
                    batch_size=self.args.batch_size * seq_len,
                    num_workers=self.args.num_workers, shuffle=True, pin_memory=True)
                self.iter_seg_image = iter(self.dataloader_seg_image)
            else:
                self.dataloader_seg_image = None

            try:
                self.dataset_seg_video = YouTubeVISDataset(
                    videodir=DATA_PATHS['youtubevis']['videodir'],
                    annfile=DATA_PATHS['youtubevis']['annfile'],
                    size=self.cfg['resolution'],
                    seq_length=seq_len,
                    seq_sampler=TrainFrameSampler(speed=[1]),
                    transform=YouTubeVISAugmentation(size))
                self.dataloader_seg_video = DataLoader(
                    self.dataset_seg_video, batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers, shuffle=True, pin_memory=True)
                self.iter_seg_video = iter(self.dataloader_seg_video)
                print('Loaded YouTubeVIS dataset.')
            except Exception as e:
                print(f'Skipping YouTubeVIS (could not load): {e}')
                self.dataloader_seg_video = None
        else:
            self.dataloader_seg_image = None
            self.dataloader_seg_video = None

    def _init_model(self):
        # Skip ImageNet backbone pretraining if we're loading full RVM weights
        pretrained_backbone = not self.args.rvm_pretrained
        self.model = HybridMattingNetwork(variant='mobilenetv3', pretrained_backbone=pretrained_backbone)

        if self.args.rvm_pretrained:
            print(f'Loading RVM pretrained weights: {self.args.rvm_pretrained}')
            rvm_state = torch.load(self.args.rvm_pretrained, map_location='cpu')
            missing, unexpected = self.model.load_state_dict(rvm_state, strict=False)
            print(f'  Loaded RVM weights — missing (new): {len(missing)}, unexpected (ignored): {len(unexpected)}')
            for k in missing:
                print(f'    [new] {k}')

        if self.args.checkpoint:
            print(f'Loading checkpoint: {self.args.checkpoint}')
            state = torch.load(self.args.checkpoint, map_location='cpu')
            self.model.load_state_dict(state, strict=False)

        self.model.to(self.device)

        if self.args.freeze_rvm:
            # Freeze everything, then unfreeze only SpatialAttention
            for param in self.model.parameters():
                param.requires_grad = False
            trainable_count = 0
            for name, param in self.model.named_parameters():
                if 'spatial_attn' in name:
                    param.requires_grad = True
                    trainable_count += param.numel()
            total_count = sum(p.numel() for p in self.model.parameters())
            print(f'Frozen mode: training {trainable_count:,} / {total_count:,} params '
                  f'({trainable_count/total_count*100:.1f}% trainable — SpatialAttention only)')

            # Only optimize SpatialAttention parameters
            spatial_attn_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = Adam(spatial_attn_params, lr=self.cfg['lr_decoder'])
        else:
            self.optimizer = Adam([
                {'params': self.model.backbone.parameters(), 'lr': self.cfg['lr_backbone']},
                {'params': self.model.aspp.parameters(),     'lr': self.cfg['lr_aspp']},
                {'params': self.model.decoder.parameters(),  'lr': self.cfg['lr_decoder']},
                {'params': self.model.project_mat.parameters(), 'lr': self.cfg['lr_decoder']},
                {'params': self.model.project_seg.parameters(), 'lr': self.cfg['lr_decoder']},
            ])
        self.scaler = GradScaler(enabled=not self.args.no_amp)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        self.step = 0
        done = False
        for epoch in range(self.cfg['epochs']):
            if done:
                break
            print(f'\n=== Epoch {epoch + 1} / {self.cfg["epochs"]} ===')
            self.validate(epoch)
            self.model.train()

            for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_train, dynamic_ncols=True):
                self._train_mat(true_fgr, true_pha, true_bgr)

                if not self.args.no_seg:
                    if self.step % 2 == 0 and self.dataloader_seg_video is not None:
                        imgs, segs = self._next_seg_video()
                        self._train_seg(imgs, segs, label='seg_video')
                    elif self.dataloader_seg_image is not None:
                        imgs, segs = self._next_seg_image()
                        self._train_seg(imgs.unsqueeze(1), segs.unsqueeze(1), label='seg_image')

                if self.step % self.args.save_interval == 0:
                    self._save(epoch)

                self.step += 1

                if self.args.max_steps and self.step >= self.args.max_steps:
                    print(f'\nReached --max-steps={self.args.max_steps}. Stopping.')
                    done = True
                    break

        self._save(epoch, final=True)
        print('Training complete.')

    def _train_mat(self, true_fgr, true_pha, true_bgr):
        true_fgr = true_fgr.to(self.device, non_blocking=True)
        true_pha = true_pha.to(self.device, non_blocking=True)
        true_bgr = true_bgr.to(self.device, non_blocking=True)
        true_fgr, true_pha, true_bgr = self._random_crop(true_fgr, true_pha, true_bgr)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)

        with autocast(enabled=not self.args.no_amp):
            pred_fgr, pred_pha = self.model(true_src)[:2]
            loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)

        self.scaler.scale(loss['total']).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.step % self.args.log_interval == 0:
            for k, v in loss.items():
                self.writer.add_scalar(f'train/{k}', v.item(), self.step)

        if self.step % (self.args.log_interval * 25) == 0:
            self.writer.add_image('train/pred_pha', make_grid(pred_pha.flatten(0, 1), nrow=pred_pha.size(1)), self.step)
            self.writer.add_image('train/pred_fgr', make_grid(pred_fgr.flatten(0, 1), nrow=pred_fgr.size(1)), self.step)
            self.writer.add_image('train/true_src', make_grid(true_src.flatten(0, 1), nrow=true_src.size(1)), self.step)

    def _train_seg(self, true_img, true_seg, label):
        true_img = true_img.to(self.device, non_blocking=True)
        true_seg = true_seg.to(self.device, non_blocking=True)
        true_img, true_seg = self._random_crop(true_img, true_seg)

        with autocast(enabled=not self.args.no_amp):
            pred_seg = self.model(true_img, segmentation_pass=True)[0]
            loss = segmentation_loss(pred_seg, true_seg)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.step % self.args.log_interval == 0:
            self.writer.add_scalar(f'train/{label}_loss', loss.item(), self.step)

    def validate(self, epoch):
        print('Validating...')
        self.model.eval()
        total_loss, count = 0, 0
        with torch.no_grad():
            with autocast(enabled=not self.args.no_amp):
                for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_valid, dynamic_ncols=True):
                    true_fgr = true_fgr.to(self.device)
                    true_pha = true_pha.to(self.device)
                    true_bgr = true_bgr.to(self.device)
                    true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                    pred_fgr, pred_pha = self.model(true_src)[:2]
                    total_loss += matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)['total'].item() * true_src.size(0)
                    count += true_src.size(0)
        avg = total_loss / count
        print(f'Validation loss: {avg:.4f}')
        self.writer.add_scalar('valid/loss', avg, self.step)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _random_crop(self, *imgs):
        h, w = imgs[0].shape[-2:]
        w = random.choice(range(w // 2, w))
        h = random.choice(range(h // 2, h))
        results = []
        for img in imgs:
            B, T = img.shape[:2]
            img = img.flatten(0, 1)
            img = F.interpolate(img, (max(h, w), max(h, w)), mode='bilinear', align_corners=False)
            img = center_crop(img, (h, w))
            img = img.reshape(B, T, *img.shape[1:])
            results.append(img)
        return results

    def _next_seg_video(self):
        try:
            return next(self.iter_seg_video)
        except StopIteration:
            self.iter_seg_video = iter(self.dataloader_seg_video)
            return next(self.iter_seg_video)

    def _next_seg_image(self):
        try:
            return next(self.iter_seg_image)
        except StopIteration:
            self.iter_seg_image = iter(self.dataloader_seg_image)
            return next(self.iter_seg_image)

    def _save(self, epoch, final=False):
        name = 'best.pth' if final else f'step-{self.step}.pth'
        path = os.path.join(self.args.checkpoint_dir, name)
        torch.save(self.model.state_dict(), path)
        print(f'Saved checkpoint: {path}')


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
