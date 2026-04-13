
"""
Implementation of Dual Embedding model 
Code adopted from PiRO (CVPR 2024)
"""

from models.architecture_utils import MultiHeadAttention
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models


class DualModel(nn.Module):
    def __init__(self, inChannel, embDim, nHeads, nLayers, dropout, nCls):
        super(DualModel, self).__init__()

        print("Using Pose-invariant Attention Network with VGG16 backbone")
        self.backbone = models.vgg16(pretrained=True)
        print(self.backbone)

        num_ftrs = 25088  # output dimensionality of VGG 16
        self.nCls = nCls
        self.backbone.classifier = nn.Identity()  # removing the classification head
        # it returns the embeddings of input image from VGG 16 architecture

        self.obj_embedder = nn.Sequential(
            nn.Linear(num_ftrs, embDim))  # object embedding
        self.cls_embedder = nn.Sequential(
            nn.Linear(num_ftrs, embDim))  # category embedding
        self.embDim = embDim

        print("Using self-attention heads = ", nHeads)
        self.obj_mha = MultiHeadAttention(
            n_head=nHeads, d_model=embDim, d_k=embDim, d_v=embDim, dropout=dropout)
        self.cls_mha = MultiHeadAttention(
            n_head=nHeads, d_model=embDim, d_k=embDim, d_v=embDim, dropout=dropout)

    def forward(self, imgBatch):
        dim = imgBatch.shape
        b, v, c, h, w = dim
        # shape of imgBatch has
        # batch (b), view (v), channel (c), height (h), width (w)
        # rearranged for VGG
        imgBatch = rearrange(imgBatch, 'b v c h w -> (b v) c h w')
        # extracting visual features from image
        # output [b*v vgg_out_dim], vgg_out_dim = 25088
        imgFtrs = self.backbone(imgBatch)

        # getting compact object, category embedding
        objEmbs = self.obj_embedder(imgFtrs)  # [b*v, emb_dim]
        clsEmbs = self.cls_embedder(imgFtrs)  # [b*v, emb_dim]

        # restoring multi-view structure for both object embedding & category embedding
        # b=b : restoring along the batches
        objEmbs = rearrange(objEmbs, '(b v) e -> b v e',
                            b=b)  # [b, v, emb_dim]
        clsEmbs = rearrange(clsEmbs, '(b v) e -> b v e', b=b)

        # context, attn for object embedding
        objContext, objAttn = self.obj_mha(objEmbs, objEmbs, objEmbs)
        # dim = 2 for each views dimension
        SVOBJEmbs = F.normalize(objEmbs, dim=2, p=2)
        # p = 2 : L2 norm (euclidean distance)

        # multi-view object embedding takes objContext vector as input case
        # initially the views will be like v1, v2, v3
        # after passing through mutli-head attention
        # it will be like
        # v1 (updated using v2, v3)
        # v2 (updated using v1, v3)
        # v3 (updated using v1, v2)
        # dim = 1 signifies for over all batch multi-view object embedding for an obj
        MVOBJEmbs = F.normalize(torch.mean(objContext, 1), dim=1, p=2)

        # context, attn for category embedding
        clsContext, clsAttn = self.cls_mha(clsEmbs, clsEmbs, clsEmbs)
        # single-view category embedding
        SVCLSEmbs = F.normalize(clsEmbs, p=2, dim=2)
        # multi-view category embedding
        MVCLSEmbs = F.normalize(torch.mean(clsContext, 1), p=2, dim=1)

        return SVOBJEmbs, SVCLSEmbs, MVOBJEmbs, MVCLSEmbs, objAttn, clsAttn
