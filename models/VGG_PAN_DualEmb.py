
# Pose-invariant attention network for dual category and object embedding
# with this model
# model learns category-specific, object-specific discriminative features with two different embedding space (objEmb, clsEmb)


from models.architecture_utils import MultiHeadAttention
import torch.nn as nn
from torchvision import models
from einops import rearrange
import torch.nn.functional as F
import torch


class DualModel(nn.Module):
    def __init__(self, inChannel, embDim, nHeads, nLayers, dropout, nCls):
        super(DualModel, self).__init__()

        print("PAN for dual object, category embedding with VGG backbone")
        self.backbone = models.vgg16(pretrained=True)
        # output dim of VGG16 is 25088
        num_ftrs = 25088
        self.nCls = nCls
        self.backbone.classifier = nn.Identity()  # removing the classification head
        # it returns the embeddings from the VGG16 architecture
        # from this embeddings we'll create object embedding and category embedding of that particular obj
        self.obj_embedder = nn.Sequential(
            nn.Linear(num_ftrs, embDim))  # compact object embedding
        self.cls_embedder = nn.Sequential(
            nn.Linear(num_ftrs, embDim))  # compact category embedding

        self.embDim = embDim
        print("Using Self-Attention heads: nHeads =", nHeads)

        # applying self-attention across views for obejct embedding
        self.obj_mha = MultiHeadAttention(
            nHeads, embDim, embDim, embDim, dropout)  # object embedding
        self.cls_mha = MultiHeadAttention(
            nHeads, embDim, embDim, embDim, dropout)  # category embedding

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
