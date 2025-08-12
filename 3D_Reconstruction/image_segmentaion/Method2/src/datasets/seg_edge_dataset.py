import cv2, glob, numpy as np, albumentations as A
from pathlib import Path
from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk
from packaging import version

_ALB_VER = version.parse(A.__version__)

def _rrc(img_size):
    if _ALB_VER >= version.parse("2.0.0"):
        return A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6,1.0), ratio=(0.9,1.1), p=1.0)
    else:
        return A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6,1.0), ratio=(0.9,1.1), p=1.0)

def _pad_if_needed(img_size):
    return A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0)

def build_aug(img_size=896, training=True):
    if training:
        return A.Compose([
            _rrc(img_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-30,30), shear=(-5,5), scale=(0.9,1.1), p=0.7),
            A.ColorJitter(0.4,0.4,0.2,0.2, p=0.8),
            A.RandomBrightnessContrast(0.2,0.2, p=0.8),
            A.GaussNoise(var_limit=(0,40), p=0.3),
            A.RandomFog(fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size, p=1.0),
            _pad_if_needed(img_size)
        ])

def find_pair(mask_dir: Path, stem: str):
    cands = sorted(glob.glob(str(mask_dir / f"{stem}.*")))
    return Path(cands[0]) if cands else None

def mask_to_edge(mask_bin: np.ndarray, edge_width=3, mode="outer"):
    bnd = find_boundaries(mask_bin, mode=mode).astype(np.uint8)
    if edge_width > 1:
        bnd = dilation(bnd, disk(max(1, edge_width//2))).astype(np.uint8)
    return bnd

class SegEdgeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=896, training=True, edge_width=3):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.training = training
        self.edge_width = edge_width
        self.aug = build_aug(img_size, training)
        exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
        self.imgs = [p for p in self.img_dir.iterdir() if p.suffix.lower() in exts]
        self.imgs.sort()

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        mp = find_pair(self.mask_dir, ip.stem)
        if mp is None:
            raise FileNotFoundError(f"Mask not found for {ip.name}")
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if img is None or msk is None:
            raise FileNotFoundError(ip, mp)

        if msk.ndim == 3:
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        msk = (msk > 127).astype(np.uint8)

        # augment
        aug = self.aug(image=img, mask=msk)
        img, msk = aug["image"], aug["mask"].astype(np.uint8)

        # edge label on-the-fly
        edge = mask_to_edge(msk, edge_width=self.edge_width, mode="outer").astype(np.float32)

        # to tensor-like
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        msk = msk.astype(np.float32)[None, ...]
        edge = edge.astype(np.float32)[None, ...]
        return img, msk, edge
