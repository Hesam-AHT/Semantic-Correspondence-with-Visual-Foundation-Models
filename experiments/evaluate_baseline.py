import torch
from src.datasets.pf_pascal_dataset import PFPascalDataset
from experiments.evaluate import evaluate, save_results, evaluate_no_spair71k
from src.models.dinov2.dinov2.models.vision_transformer import vit_base, vit_small, vit_large
import configs.paths as paths
import os
from datetime import datetime

#parameters
thresholds = [0.05, 0.1, 0.2]

base = paths.PF_PASCAL_ROOT
use_windowed_softargmax = False


#results folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'{paths.RESULTS_STEP1}/dinov2/zero-shot/dinov2_base_pfpascal'
results_dir+= '_wsoftargmax_' if use_windowed_softargmax else '_argmax_'
results_dir+=timestamp
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")


#patch size that matches the checkpoint (14 for vitb14)
model = vit_base(
    img_size=(518, 518),        # base / nominal size
    patch_size=14,             # patch size that matches the checkpoint
    num_register_tokens=0,     # <- no registers
    block_chunks=0,
    init_values=1.0,  # LayerScale initialization
)

device = "cuda" if torch.cuda.is_available() else "cpu" #use GPU if available
print("Using device:", device)
ckpt = torch.load(paths.DINOV2_WEIGHTS, map_location=device)
model.load_state_dict(ckpt, strict=True)
model.to(device)
model.eval()


test_dataset = PFPascalDataset(base, split='test')

per_image_metrics, all_keypoint_metrics, total_inference_time_sec = evaluate_no_spair71k(
    model,
    test_dataset,
    device,
    thresholds,
    use_windowed_softargmax,
    K=5,
    temperature=0.1
)

save_results(per_image_metrics, all_keypoint_metrics, results_dir, total_inference_time_sec, thresholds)
