import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset
from src.datasets.spair_dataset import read_img, Normalize

PF_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]

def _parse_coords(s):  # "x1;x2;..." -> np.ndarray
    return np.fromstring(s, sep=';')

class PFPascalDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.df = pd.read_csv(os.path.join(root, f"{split}_pairs.csv"))
        self.normalize = Normalize(['src_img', 'trg_img'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xa, ya = _parse_coords(row['XA']), _parse_coords(row['YA'])
        xb, yb = _parse_coords(row['XB']), _parse_coords(row['YB'])
        assert len(xa) == len(xb) == len(ya) == len(yb), "keypoint length mismatch"
        src_kps = torch.tensor(np.stack([xa, ya], axis=1)).float()
        trg_kps = torch.tensor(np.stack([xb, yb], axis=1)).float()

        src_path = os.path.join(self.root, row['source_image'])
        trg_path = os.path.join(self.root, row['target_image'])
        src_img = read_img(src_path)
        trg_img = read_img(trg_path)

        sample = {
            'pair_id': idx,
            'filename': os.path.basename(row['source_image']),
            'src_imname': os.path.basename(row['source_image']),
            'trg_imname': os.path.basename(row['target_image']),
            'src_imsize': src_img.size(),  # (C,H,W)
            'trg_imsize': trg_img.size(),
            'category': PF_CLASSES[int(row['class'])-1],  # class is 1-indexed
            'src_pose': None, 'trg_pose': None,
            'src_img': src_img, 'trg_img': trg_img,
            'src_kps': src_kps, 'trg_kps': trg_kps,
            'kps_ids': list(range(len(src_kps))),
            'mirror': False, 'vp_var': None, 'sc_var': None,
            'truncn': None, 'occlsn': None,
        }
        return self.normalize(sample)
