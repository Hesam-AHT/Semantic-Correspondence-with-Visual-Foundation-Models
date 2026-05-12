import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset
from src.datasets.spair_dataset import read_img, Normalize

PF_CLASSES = [
    "car(G)", "car(M)", "car(S)", "duck(S)", "motorbike(G)", "motorbike(M)", "motorbike(S)", "winebottle(M)",
    "winebottle(wC)", "winebottle(woC)"
]

class PFWillowDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.df = pd.read_csv(os.path.join(root, f"{split}_pairs.csv"))
        self.normalize = Normalize(['src_img', 'trg_img'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract keypoints from separate columns
        # XA1, XA2, ..., XA10 and YA1, YA2, ..., YA10
        xa = np.array([row[f'XA{i+1}'] for i in range(10)])
        ya = np.array([row[f'YA{i+1}'] for i in range(10)])
        xb = np.array([row[f'XB{i+1}'] for i in range(10)])
        yb = np.array([row[f'YB{i+1}'] for i in range(10)])

        src_kps = torch.tensor(np.stack([xa, ya], axis=1)).float()
        trg_kps = torch.tensor(np.stack([xb, yb], axis=1)).float()

        src_path = os.path.join(self.root, row['imageA'])
        trg_path = os.path.join(self.root, row['imageB'])
        src_img = read_img(src_path)
        trg_img = read_img(trg_path)

        sample = {
            'pair_id': idx,
            'filename': os.path.basename(row['imageA']),
            'src_imname': os.path.basename(row['imageA']),
            'trg_imname': os.path.basename(row['imageB']),
            'src_imsize': src_img.size(),  # (C,H,W)
            'trg_imsize': trg_img.size(),
            'category': PF_CLASSES[0],  # PF-Willow doesn't have class info in CSV
            'src_pose': None, 'trg_pose': None,
            'src_img': src_img, 'trg_img': trg_img,
            'src_kps': src_kps, 'trg_kps': trg_kps,
            'kps_ids': list(range(len(src_kps))),
            'mirror': False, 'vp_var': None, 'sc_var': None,
            'truncn': None, 'occlsn': None,
        }
        return self.normalize(sample)
