import torch
import torch.nn as nn

class TextDecoderLayer(nn.Module):
    def __init__(self, 
                d_model, 
                num_heads_mha, 
                num_heads_cross_attn, 
                d_feedforward, 
                d_k,
                d_v, 
                vit_dim):
        super(TextDecoderLayer, self).__init__()

        self.k = nn.Linear(d_model, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.MHA_1 = nn.MultiheadAttention(d_model, num_heads_mha, batch_first =True)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.k_cross_attn = nn.Linear(vit_dim, d_model)
        self.q_cross_attn = nn.Linear(d_model, d_model)
        self.v_cross_attn = nn.Linear(vit_dim, d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads_cross_attn, batch_first=True)
        self.layer_norm_cross_attn = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, d_feedforward)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_feedforward, d_model)
        self.layer_norm_ff = nn.LayerNorm(d_model)

    def forward(self, x, vision_features=None, enable_cross_attn=True, causal_mask=False, padding_mask=None, attn_mask=None):
        """Forward method for decoder layer with added option to disable cross attention and causal masking"""
        # k1 = self.k(x)
        # q1 = self.q(x)
        # v1 = self.v(x)

        out = self.MHA_1(query=x, key=x, value=x, is_causal=causal_mask, key_padding_mask=padding_mask, attn_mask=attn_mask)
        out = self.layer_norm1(out[0] + x)
        out_layer_norm1 = torch.clone(out)

        if enable_cross_attn:
            # k2 = self.k_cross_attn(vision_features)
            # q2 = self.q_cross_attn(x)
            # v2 = self.v_cross_attn(vision_features)
            out = self.cross_attn(query=out, key=vision_features, value=vision_features)
    
            out = self.layer_norm_cross_attn(out[0] + out_layer_norm1)
            out_layer_norm_cross_attn = torch.clone(out)
        
        out = self.fc2(self.relu(self.fc1(out)))
        if enable_cross_attn:
            out = self.layer_norm_ff(out + out_layer_norm_cross_attn)
        else:
            out = self.layer_norm_ff(out + out_layer_norm1)

        return out








