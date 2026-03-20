
# in the PiRO literature, it was mentioned of using 
# two single head self-attention layer
# two single head self-attention is multiheadattention

# standard architecture of multiheadattention
import numpy as np
from einops import rearrange # einops = einstein inspired notations
# used for simplifying tensor manipulation
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        # n_head = no of head in multiheadattention
        # d_model = dimension of the model
        # d_k = dimension of key and query vector
        # d_v = dimension of value vector
        self.n_head = n_head

        # weighted matrix of query, key, value
        self.w_qs = nn.Linear(d_model, n_head*d_k)
        self.w_ks = nn.Linear(d_model, n_head*d_k)
        self.w_vs = nn.Linear(d_model, n_head*d_v)

        # normalizing weighted matrix
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0/(d_model+d_v)))

        # quey, key, d_k compute attention (attention = query vector * key vector transpose / square root of d_k)
        # d_v carry information

        # final linear layer = combines all heads and project back to original dimension
        self.fc = nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight) # initializing weights of linear layer using xavier (glorot) normal initialization
        self.dropout = nn.Dropout(p = dropout)
        self.layer_norm = nn.LayerNorm(d_model) # layer normalization

    
    def forward(self,q,k,v,mask = None):
        # inputs are query, key, value vectors
        residual = q # for residual connection (skipping)
        
        # splitting query, key, value into multiple heads
        q = rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head = self.n_head)
        v = rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        # b = batch size, head = no of attention head, l = sequence length (no of tokens/views)
        # k = dimension of key/query vector, v = dimension of value vector
        # t = number of views

        # calculating attention
        attn = torch.einsum('hblk,hbtk->hblt', [q,k])/np.sqrt(q.shape[-1])
        # so [q,k] is calculating Q.K^T 
        # np.sqrt(q.shape[-1]) = np.sqrt(d_k)

        if mask is not None:
            attn = attn.masked_fill(mask[None],-np.inf) # applying masking
        attn = torch.softmax(attn, dim=3) # atten weights
        output=torch.einsum('hblt,hbtv->hblv',[attn,v]) # applying attn weights to v
        output = rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


"""
Use only when extracting features 
"""

def set_feature_extraction_mode(model, feature_extracting):
    if feature_extracting:
        print("VGGnet in feature extracting mode")
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Training VGG-16 backbone end-to-end")


"""
bbone = models.resnet50(pretrained=True)
    set_parameter_requires_grad(bbone, False)
    num_ftrs = bbone.fc.in_features
    bbone.fc = nn.Sequential(
                nn.Linear(num_ftrs, embDim)) 
"""



