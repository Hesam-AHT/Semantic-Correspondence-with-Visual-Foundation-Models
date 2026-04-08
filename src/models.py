# DINO v2 
import torch
import torch.nn as nn
class DINOv2FeatureExtractore(nn.model):
    def __init__(self, model_size = 'vitb14'):
        # 'vits14' (small), 'vitb14' (base), 'vitl14' (large), or 'vitg14' (giant)
        super().__init__()

        #LOADIng the pre-trained MOdel
        print(f"load DINOv2 ({model_size})")
        self.model_name = f"dinov2_{model_size}"
        self.backbone =torch.hub.load("facebookresearch/dinov2", self.model_name)

        #Freezing the Layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        #evaluation
        self.backbone.ecal()

    def forward (self, img_tensor):
        with torch.no_grad():
            
            features = self.backbone.forward_features(img_tensor)

            patch_tokens = features.["x_norm_patchtokens"]
        
        return patch_tokens
    
# test for dinov2 implementation
if __name__ = "__init__":
    dummy_image = torch.randn(1, 3, 224, 224) 
    extractor = DINOv2FeatureExtractor(model_size='vits14')
    
    # Extract features
    features = extractor(dummy_img)

    print(f"Extracted features shape: {features.shape}")   

