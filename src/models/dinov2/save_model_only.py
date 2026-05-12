import torch
from models.dinov2.dinov2.models.vision_transformer import vit_base

device = "cuda" if torch.cuda.is_available() else "cpu"

#load full checkpoint
checkpoint = torch.load(
    "dinov2_training_checkpoint_epoch1_1temp.pth",
    map_location=device
)

# ======== RICOSTRUISCI MODELLO ========
model = vit_base(
    img_size=(518, 518),
    patch_size=14,
    num_register_tokens=0,
    block_chunks=0,
    init_values=1.0,
)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(device)
model.eval()

#save only the model weights
torch.save(
    model.state_dict(),
    "dinov2_vitb14_finetuned_only_model_1temp.pth"
)
