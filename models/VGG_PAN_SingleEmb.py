
# Pose-invariant attention network architecture for single object embedding
# with this model
# model learns category-specific and object-specific discriminative features within the same embedding space


from models.architecture_utils import MultiHeadAttention
import torch.nn as nn
from torchvision import models
from einops import rearrange  # einstein inspired notations
import torch.nn.functional as F
import torch


class SingleModel(nn.Module):
    def __init__(self, inChannel, embDim, nHeads, nLayers, dropout, nCls):
        super(SingleModel, self).__init__()

        print("PAN for single object embedding with VGG16 backbone")
        self.backbone = models.vgg16(
            pretrained=True)  # taking pretrained model
        # output dimension of vgg16 is 25088
        num_ftrs = 25088  # output feature vector size
        self.nCls = nCls  # total number of classes in dataset
        self.backbone.classifier = nn.Identity()  # removing the classifier layer

        # object embedder converts output dimension of model into object embeddings
        # this helps to create a compact representation
        # from 25088 -> specified object embedding
        self.obj_embedder = nn.Sequential(nn.Linear(num_ftrs, embDim))
        self.embDim = embDim

        # no of heads we'll be using in mutli-head attention
        print("Using Self-Attention: nHeads =", nHeads)
        # applying self-attention across views
        self.obj_mha = MultiHeadAttention(
            nHeads, embDim, embDim, embDim, dropout)

    def forward(self, imgBatch):  # imgBatch is shape of the input multi-view image
        dim = imgBatch.shape
        b, v, c, h, w = dim
        # batch (b), no of views per object (v), channel (c), height (h), width (w)
        # e.g. [8,5,3,224,224] = 8 object each with 5 views
        # VGG16 architecture expects
        # [batch, channels, height, width] so
        # [b,v,c,h,w] -> [(b,v), c, h, w]
        imgBatch = rearrange(imgBatch, 'b v c h w -> (b v) c h w').cuda()

        # feature extraction from the backbone of VGG16
        # embedding output [b*v, 25088], 25088 is the output feature vector dim by VGG16
        imgFtrs = self.backbone(imgBatch)
        # from the embedding of imgBatch
        # generating single-view object embedding
        objEmbs = self.obj_embedder(imgFtrs)  # [b*v, emb_dim]
        # rearraning dimensions [b,v, emb_dim]
        objEmbs = rearrange(objEmbs, '(b v) e -> b v e', b=b)

        # taking context and attn for object embedding
        # inputting key, query, value vector
        objContext, objAttn = self.obj_mha(objEmbs, objEmbs, objEmbs)
        # shape of objContext = [b,v,emb_dim]
        # initially all views are independent
        # now after putting through multi-head attention
        # each views are context aware w.r.t. other views
        # this helps to formulate multi-view object embeddings

        # normalized single-view object embeddings
        # each view has it's own normalized embeddings
        # objEmbs = [b,v,emb_dim]
        # p = 2 menns L2 norm (euclidean distance)
        # dim = 2 for each views dimension
        SVOBJEmbs = F.normalize(objEmbs, p=2, dim=2)

        # normalized multi-view object embedding
        # multi-view object embeddings across all views
        MVOBJEmbs = F.normalize(torch.mean(objContext, 1), dim=1, p=2)

        return SVOBJEmbs, MVOBJEmbs, objAttn
