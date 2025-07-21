import torch
import torch.nn as nn
from PIL import Image as PIL_Image
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import v2
from text_decoder import TextDecoderLayer
import torch.nn.functional as F

class MaMMUT(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 vit_num_layers: int = 8,
                 vit_num_heads: int = 8,
                 vit_hidden_dim: int = 768,
                 vit_mlp_dim: int = 2048,
                 vit_dropout: float = 0.0, # Potential ablation / extension to add to the replication
                 vit_attention_dropout: float = 0.0, # Potential ablation / extension to add to the replication
                 contrastive_loss_weight: float = 1.0,
                 generative_loss_weight: float = 1.0,
                 text_decoder_depth: int = 4,
                 text_decoder_embed_dim: int = 256,
                 text_decoder_sub_layer_heads: int = 4,
                 text_decoder_feedforward_dim: int = 2048,
                 text_decoder_dk: int = 128,
                 vocab_size: int = 1000,
                 latent_dim: int = 512,
                 contrastive_loss_temp: float = 0.5,
                 contrastive_loss_gamma: float = 1.0):
                 
        super(MaMMUT, self).__init__()   
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

        self.token_size = vocab_size
        self.text_decoder_embed_dim = text_decoder_embed_dim
        self.text_decoder_sub_layer_heads = text_decoder_sub_layer_heads
        
        self.contrastive_loss_weight = contrastive_loss_weight
        self.generative_loss_weight = generative_loss_weight
        
        self.ifs2tfs = nn.Linear(vit_hidden_dim, text_decoder_embed_dim)
        
        self.text_decoder_depth = text_decoder_depth
        self.text_decoder_layers = []

        self.pos_embedding = nn.Parameter(torch.randn(1, (image_size // patch_size)**2 + 1, vit_hidden_dim))
        
        self.final_layernorm = nn.LayerNorm(self.token_size)

        self.latent_text_features = nn.Linear(text_decoder_embed_dim, text_decoder_embed_dim) # for contrastive loss
        self.pad_token_id = 0 # we can set this in the SentencePiece tokenizer

        self.text_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=text_decoder_embed_dim, padding_idx=self.pad_token_id)

        self.text_cls_token = nn.Parameter(torch.randn(text_decoder_embed_dim))
        self.contrastive_layernorm = nn.LayerNorm(text_decoder_embed_dim)

        self.loss_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id).to(self.device)
        self.contrastive_loss_temp = nn.Parameter(torch.Tensor([1.]).to(self.device))
        self.contrastive_loss_gamma = contrastive_loss_gamma
        self.image_size = image_size
        self.patch_size = patch_size
        
        
        # Changing logic for the decoder layer. This way we can disable cross-attention during the forward pass and keep everything else the same
        # for i in range(text_decoder_depth):
        #     self.text_decoder_layers.append(TextDecoderLayer(d_model=text_decoder_embed_dim, num_heads_mha=text_decoder_sub_layer_heads, \
        #                                                     num_heads_cross_attn=text_decoder_sub_layer_heads, d_feedforward=text_decoder_feedforward_dim, \
        #                                                      d_k=text_decoder_dk, d_v=(text_decoder_embed_dim // text_decoder_sub_layer_heads), vit_dim=vit_hidden_dim).to(self.device)
        #                                                      )
        self.text_decoder_layers = nn.ModuleList([TextDecoderLayer(d_model=text_decoder_embed_dim, num_heads_mha=text_decoder_sub_layer_heads, \
                                                            num_heads_cross_attn=text_decoder_sub_layer_heads, d_feedforward=text_decoder_feedforward_dim, \
                                                             d_k=text_decoder_dk, d_v=(text_decoder_embed_dim // text_decoder_sub_layer_heads), vit_dim=vit_hidden_dim).to(self.device)
                                                              for _ in range(text_decoder_depth)])
        self.decoder_output_features_to_text_tokens_layer = nn.Linear(self.text_decoder_embed_dim, self.token_size) # for captioning loss
            
        
    def cropped_positional_encoding(self, feats):
        # feats shape: N x (H_p x W_p) x Hidden
        n, h_w, hidden = feats.shape


        # take out cls token before upsampling
        cls_pos_embed = self.pos_embedding[:, 0, :]
        pos_embeddings = self.pos_embedding[:, 1: :]

        pos_embeddings = pos_embeddings.reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, hidden).permute(0, -1, 1, 2)

        # Upsample using bilinear interpolation
        upsample_layer = nn.Upsample(mode='bilinear', scale_factor=4)
        upsampled_pos_embeddings = upsample_layer(pos_embeddings)
        random_crop = v2.RandomCrop(pos_embeddings.shape[2])
        cropped_pos_encoding = random_crop(upsampled_pos_embeddings)

        # cropped_pos_encoding shape: N x (H_p x W_p) x Hidden. Reshape to align with feats
        cropped_pos_encoding = cropped_pos_encoding.reshape(1, h_w-1, hidden)
        
        cropped_pos_encoding = torch.cat([cropped_pos_encoding, cls_pos_embed.reshape(1, 1, hidden)], dim=1)


        return feats + cropped_pos_encoding

        
    def get_vision_features(self, img: torch.tensor):
        # image has shape N x C x H x W where
        # N is the batch size
        # C is the channel size
        # H is the image height
        # W is the image width
#         preprocessing = v2.Compose([
#             v2.ToImage(),
#             v2.Resize((272,272)),
#             v2.RandomCrop(224)
#         ])

#         img = PIL_Image.open("example_2353642598754.jpeg")
#         img = preprocessing(img)

        # Add batch dimension - for testing on one image, remove for training
#         img = img.unsqueeze(0)
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w), converts into patches
        feats = self.vit._process_input(img)
        # Expand the CLS token to the full batch
        batch_class_token = self.vit.class_token.expand(img.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        
        feats = self.cropped_positional_encoding(feats)

        feats = self.vit.encoder(feats)

        # Fetch pre-prended CLS token at position 0 in dimension 1
        feats = feats[:, 0].reshape(feats.shape[0], 1, feats.shape[-1])
                
        return feats
    
    def img_feat_size_to_txt_feat_size(self, vision_features: torch.tensor):
        return self.ifs2tfs(vision_features)
    
    def contrastive_text_features(self, text_embeds: torch.Tensor):
        # text has shape N x S
        # Remember to pass bidirectional mask (as far as I understand, a mask that allows attention to all non-padded areas or maybe just all non-CLS areas and maybe stops cls from attending to padding TODO: Clarify)
        # Remember to perform residual additions         
        # expand to match dimensions
        cls_tokens = self.text_cls_token.expand(text_embeds.shape[0], 1, self.text_decoder_embed_dim).to(self.device)
        # Add cls tokens to start of the sequences
        text_embeds = torch.cat([cls_tokens, text_embeds], dim=1)
        cls_padding_mask = (text_embeds == 0).all(dim=-1) # From nn.Embedding, padding tokens are embedded as vector of 0s. Result should be shape N x S.

        output = text_embeds.clone()
        for i, layer in enumerate(self.text_decoder_layers):
            # Disable cross-attention for contrastive features
            output = layer(output, enable_cross_attn=False, padding_mask=cls_padding_mask)

        output = output[:, 0]
        output = self.contrastive_layernorm(output)
        return output
    
    def generative_text_features(self, text_embeds: torch.tensor, vision_features: torch.tensor):
        # Remember to toggle causal in forward pass
        # Remember to perform residual additions

        attn_mask = torch.triu(torch.ones((text_embeds.shape[1], text_embeds.shape[1])), diagonal=1).bool().to(self.device) # Assuming shape[1] is the sequence dim
        output = text_embeds.clone()
        padding_mask = (text_embeds == 0).all(dim=-1)
        for i, layer in enumerate(self.text_decoder_layers):
            # Disable cross-attention for odd numbered layers
            if i % 2 != 0:
                output = layer(output, vision_features=None, enable_cross_attn=False, causal_mask=True, attn_mask=attn_mask, padding_mask=padding_mask)
            else:
                # enable cross-attention for even numbered layers
                output = layer(output, vision_features=vision_features, enable_cross_attn=True, causal_mask=True, attn_mask=attn_mask, padding_mask=padding_mask)
        return output

    
    def contrastive_loss(self, vision_features: torch.tensor, constrastive_text_features: torch.tensor):
        """Implement Focal-contrastive loss as in the paper"""
        similarity = (vision_features @ constrastive_text_features.T) / 0.07
        
        similarity = similarity.view(similarity.shape[0], -1)
        # print(similarity.shape)
        # similarity = similarity.squeeze(1)
        # In contrastive learning we aim to minimize loss for between the matching image and text pairs, and maximize loss 
        # for mismatching image text pairs.
        # after the matrix multipication, shape will be N x N
        # each row represents image i, and each column would represent each caption
        # Therefore, the matching pairs will be across the diagonal (0,0), (1, 1) ... and we can treat this as a classification task
        # where we compute the loss between the text_logits and its matching image and vice-versa for the image loss
    
        # We can construct the labels by just creating a diagonal matrix
        labels = torch.arange(similarity.shape[0]).to(self.device)
        # labels_one_hot = F.one_hot(labels, num_classes=similarity.shape[0]).to(self.device)
        # probs_imgs = F.softmax(similarity, dim=1) # using softmax instead of sigmoid
        # p_t_imgs = torch.sum(labels_one_hot * probs_imgs, dim=1)
        # p_t_imgs = torch.clamp(p_t_imgs, min=1e-5)
        # loss_i2t = -(((1 - p_t_imgs) ** self.contrastive_loss_gamma) * (torch.log(p_t_imgs))).mean()
        # probs_texts = F.softmax(similarity, dim=0)
        # p_t_texts = torch.sum(labels_one_hot * probs_texts, dim=1)
        # p_t_texts = torch.clamp(p_t_texts, min=1e-5)

        # loss_t2i = -(((1 - p_t_texts) ** self.contrastive_loss_gamma) * (torch.log(p_t_texts))).mean()
        loss_t2i = self.loss_criterion(similarity.T, labels)
        loss_i2t = self.loss_criterion(similarity, labels)
        total_contrastive_loss = (loss_i2t + loss_t2i) / 2

        return total_contrastive_loss.to(self.device)


    def generative_loss(self, generative_text_features: torch.tensor, text_labels: torch.tensor):
        generative_text_features = generative_text_features.permute(0, -1, 1) # cross-entropy expects N x C as first two dims
        loss = self.loss_criterion(generative_text_features, text_labels)
        return loss.to(self.device)

    def decoder_output_features_to_text_tokens(self, text_features: torch.tensor):
        return self.final_layernorm(self.decoder_output_features_to_text_tokens_layer(text_features))
        
    
    def forward(self, image, text, text_labels):
        # Pseudocode for now, need to fully implement and test
        # TODO: Implement average pooling over spatial dimension and sequence where appropriate
        # TODO: Add tokenizer & params ------- Tokenizer would be added in training pipeline
        text_embeds = self.text_embeddings(text)
#         print(text_embeds)
        vision_features = self.get_vision_features(image)
        vision_features_contrastive = self.img_feat_size_to_txt_feat_size(vision_features) # projects image feature dim to text feature dim
        
        constrastive_text_features = self.contrastive_text_features(text_embeds)
        constrastive_text_features = self.latent_text_features(constrastive_text_features)

        contrastive_loss = self.contrastive_loss(vision_features_contrastive, constrastive_text_features)
        
        # Since task will be to generate next token, we only go up until the second last token
        generative_text_features = self.generative_text_features(text_embeds[:, :-1], vision_features_contrastive)

        text_logits = self.decoder_output_features_to_text_tokens(generative_text_features)

        generative_loss = self.generative_loss(text_logits, text_labels)
        loss = self.contrastive_loss_weight * contrastive_loss + self.generative_loss_weight * generative_loss

        return loss, contrastive_loss, generative_loss