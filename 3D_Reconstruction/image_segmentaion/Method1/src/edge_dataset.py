# src/edge_dataset.py
import cv2, random, numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A

def build_aug(img_size=896):
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.6,1.0), ratio=(0.9,1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-30,30), shear=(-5,5), scale=(0.9,1.1), p=0.7),
        A.ColorJitter(0.4,0.4,0.2,0.2, p=0.8),
        A.RandomBrightnessContrast(0.2,0.2, p=0.8),
        A.GaussNoise(var_limit=(0, 40), p=0.3),
        A.RandomFog(fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
    ])

class EdgeDataset(Dataset):
    def __init__(self, img_dir, edge_dir, img_size=896, training=True):
        self.img_paths = sorted(list(Path(img_dir).glob('*')))
        self.edge_dir = Path(edge_dir)
        self.training = training
        self.img_size = img_size
        self.aug = build_aug(img_size) if training else A.Compose([
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        ip = self.img_paths[i]
        ep = self.edge_dir/ip.name
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        edge = cv2.imread(str(ep), cv2.IMREAD_GRAYSCALE)
        if img is None or edge is None:
            raise FileNotFoundError(ip, ep)
        # to 0/1
        edge = (edge>127).astype(np.uint8)
        auged = self.aug(image=img, mask=edge)
        img, edge = auged['image'], auged['mask']
        img = img[:, :, ::-1]  # BGR->RGB
        # to tensor-like CHW float32
        img = np.transpose(img.astype(np.float32)/255., (2,0,1))
        edge = edge.astype(np.float32)[None, ...]
        return img, edge
