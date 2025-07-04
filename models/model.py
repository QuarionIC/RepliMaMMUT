import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from PIL import Image as PIL_Image
from torchvision.models.vision_transformer import VisionTransformer

class MaMMUT(nn.Module):
    def __init__(self,
                 image_size: int = 272,
                 patch_size: int = 224,
                 vit_num_layers: int = 6,
                 vit_num_heads: int = 8,
                 vit_hidden_dim: int = 1024,
                 vit_mlp_dim: int = 512,
                 vit_dropout: float = 0.0,
                 vit_attention_dropout: float = 0.0,
                 contrastive_loss_weight: float = 1.0,
                 generative_loss_weight: float = 1.0):
        self.vit =VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            hidden_dim=vit_hidden_dim,
            mlp_dim=vit_mlp_dim,
            dropout=vit_dropout,
            attention_dropout=vit_attention_dropout,
            num_classes=1000
        )
        
        self.contrastive_loss_weight = contrastive_loss_weight
        self.generative_loss_weight = generative_loss_weight
    
    def forward(self, image, text):
        vision_features = self.get_vision_features(image)
        
        vision_features = self.img_feat_size_to_txt_feat_size(vision_features)
        
        constrastive_text_features = self.constrastive_text_features(text, bidrectional_mask)
        
        generative_text_features = self.generative_text_features(text, vision_features, causal_mask)
        
        contrastive_loss = self.contrastive_loss(vision_features, constrastive_text_features)
        
        generative_loss = self.generative_loss(generative_text_features, text)
        
        loss = self.contrastive_loss_weight * contrastive_loss + self.generative_loss_weight * generative_loss
        
        return loss
        
        
    
    def test_vit():

        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        preprocessing = ViT_B_16_Weights.DEFAULT.transforms()

        img = PIL_Image.open("example.png")
        img = preprocessing(img)

        # Add batch dimension
        img = img.unsqueeze(0)

        feats = vit._process_input(img)

        # Expand the CLS token to the full batch
        batch_class_token = vit.class_token.expand(img.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)

        feats = vit.encoder(feats)

        # We're only interested in the representation of the CLS token that we appended at position 0
        feats = feats[:, 0]

        print(feats.shape)