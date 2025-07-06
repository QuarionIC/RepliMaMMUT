import torch
import torch.nn as nn
from PIL import Image as PIL_Image
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import v2

class MaMMUT(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 14,
                 vit_num_layers: int = 32,
                 vit_num_heads: int = 16,
                 vit_hidden_dim: int = 1280,
                 vit_mlp_dim: int = 5120,
                 vit_dropout: float = 0.0, # Potential ablation / extension to add to the replication
                 vit_attention_dropout: float = 0.0, # Potential ablation / extension to add to the replication
                 contrastive_loss_weight: float = 1.0,
                 generative_loss_weight: float = 1.0,
                 text_decoder_depth: int = 6,
                 text_decoder_embed_dim: int = 512,
                 text_decoder_sub_layer_heads: int = 8):
        self.vit = VisionTransformer(
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
        
        self.text_decoder_embed_dim = text_decoder_embed_dim
        self.text_decoder_sub_layer_heads = text_decoder_sub_layer_heads
        
        self.contrastive_loss_weight = contrastive_loss_weight
        self.generative_loss_weight = generative_loss_weight
        
        self.ifs2tfs = nn.Linear(vit_hidden_dim, text_decoder_embed_dim)
        
        self.text_decoder_depth = text_decoder_depth
        self.text_decoder_layers = []

        self.pos_embedding = nn.Embedding(num_embeddings=(image_size // patch_size)**2, embedding_dim=vit_hidden_dim)
        
        for i in range(text_decoder_depth):
            # Layer norm before MHA from https://arxiv.org/abs/2002.04745
            
            if (i % 2 == 0): # Note that cross attention layer occurs every 2 layers to sastisfy M/N ~= 2 from paper
                self.text_decoder_layers.append([ # Figure out params
                    nn.LayerNorm(),
                    nn.MultiheadAttention(self.text_decoder_embed_dim, self.text_decoder_sub_layer_heads, batch_first = True),
                    nn.LayerNorm(),
                    nn.MultiheadAttention(self.text_decoder_embed_dim, self.text_decoder_sub_layer_heads, batch_first = True), # The cross attention layer that accepts image features
                    nn.LayerNorm(),
                    nn.Linear()
                ])
            else:
                self.text_decoder_layers.append([ # Figure out params
                    nn.LayerNorm(),
                    nn.MultiheadAttention(self.text_decoder_embed_dim, self.text_decoder_sub_layer_heads, batch_first = True),
                    nn.LayerNorm(),
                    nn.Linear(self.text_decoder_embed_dim, self.text_decoder_embed_dim, batch_first = True)
                ])
        
        self.decoder_output_features_to_text_tokens_layer = nn.Linear(self.text_decoder_embed_dim, self.token_size)
            
            
    def cropped_positional_encoding(self, feats):
        # feats shape: N x Hidden x H_p x W_p
        n, hidden, h, w = feats.shape
        # Reshape to N x Hidden x (H_p x W_p) 
        feats = feats.reshape(n, hidden, h * w)
        # Change shape to N x (H_p x W_p) x Hidden
        feats = feats.permute(0, -1, 1)

        # pos_embedding shape: N x (H_p x W_p) x Hidden
        pos_embeddings = self.pos_embedding(feats)
        # convert shape back to N x Hidden x H_p x W_p to upsample
        pos_embeddings = pos_embeddings.reshape(n, h, w, hidden).permute(0, -1, 1, 2)

        # Upsample using bilinear interpolation
        upsample_layer = nn.Upsample(mode='bilinear', scale=4, size=(pos_embedding.shape[2], pos_embedding.shape[3]))
        upsampled_pos_embeddings = upsample_layer(pos_embeddings)

        random_crop = v2.RandomCrop(pos_embedding.shape[2])
        cropped_pos_encoding = random_crop(upsampled_pos_embeddings)

        # cropped_pos_encoding shape: N x Hidden x H_p x W_p. Reshape to align with feats
        cropped_pos_encoding = cropped_pos_encoding.reshape(n, hidden, h*w)

        return feats + cropped_pos_encoding

        
    def get_vision_features(self, image: torch.tensor):
        # image has shape N x C x H x W where
        # N is the batch size
        # C is the channel size
        # H is the image height
        # W is the image width
        preprocessing = v2.Compose([
            v2.ToImage(),
            v2.Resize((272,272)),
            v2.RandomCrop(224)
        ])

        img = PIL_Image.open("example_2353642598754.jpeg")
        img = preprocessing(img)

        # Add batch dimension - for testing on one image, remove for training
        img = img.unsqueeze(0)
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w), converts into patches
        feats = self.vit._process_input(img)

        # Expand the CLS token to the full batch
        batch_class_token = self.vit.class_token.expand(img.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        
        feats = self.cropped_positional_encoding(feats)

        feats = self.vit.encoder(feats)

        # Fetch pre-prended CLS token at position 0 in dimension 1
        feats = feats[:, 0]
        
        print(feats.shape)
        
        return feats
    
    def img_feat_size_to_txt_feat_size(self, vision_features: torch.tensor):
        return self.ifs2tfs(vision_features)
    
    def contrastive_text_features(self, text: torch.Tensor):
        # text has shape N x S
        # Remember to pass bidirectional mask (as far as I understand, a mask that allows attention to all non-padded areas or maybe just all non-CLS areas and maybe stops cls from attending to padding TODO: Clarify)
        # Remember to perform residual additions
        pass
    
    def generative_text_features(self, text: torch.tensor, vision_features: torch.tensor):
        # Remember to toggle causal in forward pass
        # Remember to perform residual additions
        pass
    
    def contrastive_loss(self, vision_features: torch.tensor, constrastive_text_features: torch.tensor):
        pass
    
    def generative_loss(generative_text_features: torch.tensor, text: torch.tensor):
        pass
    
    def decoder_output_features_to_text_tokens(self, text_features: torch.tensor):
        return self.decoder_output_features_to_text_tokens_layer(text_features)
        
    
    def forward(self, image, text):
        # Pseudocode for now, need to fully implement and test
        # TODO: Implement average pooling over spatial dimension and sequence where appropriate
        # TODO: Add tokenizer & params
        vision_features = self.get_vision_features(image)
        
        vision_features = self.img_feat_size_to_txt_feat_size(vision_features) # projects image feature dim to text feature dim
        
        constrastive_text_features = self.constrastive_text_features(text)
        
        generative_text_features = self.generative_text_features(text, vision_features)
        
        contrastive_loss = self.contrastive_loss(vision_features, constrastive_text_features)
        
        text_logits = self.decoder_output_features_to_text_tokens(generative_text_features)
        
        generative_loss = self.generative_loss(text_logits, text)
        
        loss = self.contrastive_loss_weight * contrastive_loss + self.generative_loss_weight * generative_loss
        
        return loss