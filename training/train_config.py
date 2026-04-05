"""
DATA PATHS — edit these to match where you downloaded the datasets.

Expected folder structure for each dataset:

VideoMatte240K (train + valid):
    VideoMatte240K/
    ├── train/
    │   ├── fgr/
    │   │   ├── 0001/
    │   │   │   ├── 00000.jpg
    │   │   │   └── ...
    │   └── pha/
    │       ├── 0001/
    │       │   ├── 00000.jpg
    │       │   └── ...
    └── valid/
        ├── fgr/
        └── pha/

Background images (train + valid):
    Backgrounds/
    ├── train/
    │   ├── img001.jpg
    │   └── ...
    └── valid/
        └── ...

Background videos (train + valid):
    BackgroundVideos/
    ├── train/
    │   ├── 0001/
    │   │   ├── 0000.jpg
    │   │   └── ...
    └── valid/
        └── ...

Supervisely Person Dataset (segmentation — optional but recommended):
    SuperviselyPersonDataset/
    ├── img/
    │   ├── img001.jpg
    │   └── ...
    └── seg/
        ├── img001.png
        └── ...

COCO Panoptic (segmentation — optional, large ~25GB):
    coco/
    ├── train2017/
    ├── panoptic_train2017/
    └── annotations/
        └── panoptic_train2017.json

YouTubeVIS (segmentation — optional, large):
    YouTubeVIS/
    └── train/
        ├── JPEGImages/
        └── instances.json
"""

# ── EDIT THESE PATHS ──────────────────────────────────────────────────────────

DATA_PATHS = {

    # Main matting dataset (required)
    'videomatte': {
        'train': '/path/to/VideoMatte240K/train',
        'valid': '/path/to/VideoMatte240K/test',
    },

    # Background images (required) — using BG-20K dataset
    'background_images': {
        'train': '/path/to/BG-20k/train',
        'valid': '/path/to/BG-20k/testval',
    },

    # Background videos — can point to same folder as background_images if not available
    'background_videos': {
        'train': '/path/to/BG-20k/train',
        'valid': '/path/to/BG-20k/testval',
    },

    # Supervisely Person Dataset (segmentation — set use_seg=False in train.py to skip)
    'spd': {
        'imgdir': '/path/to/SuperviselyPersonDataset/img',
        'segdir': '/path/to/SuperviselyPersonDataset/seg',
    },

    # COCO Panoptic (segmentation — optional, large)
    'coco_panoptic': {
        'imgdir': '/path/to/coco/train2017/',
        'anndir': '/path/to/coco/panoptic_train2017/',
        'annfile': '/path/to/coco/annotations/panoptic_train2017.json',
    },

    # YouTubeVIS (segmentation — optional, large)
    'youtubevis': {
        'videodir': '/path/to/YouTubeVIS/train/JPEGImages',
        'annfile':  '/path/to/YouTubeVIS/train/instances.json',
    },
}
