
# total loss includes both pose-invariant losses + category loss of A and N
# Category loss using large-margin softmax loss 



import torch.nn as nn
from pytorch_metric_learning import losses, miners


class LossCAT(nn.Module):
    # category losses for both object A,N

    def __init__(self, Config, gamma=4.0):
        super(LossCAT, self).__init__()
        # margin for softmax loss (controls the seperation length)
        self.gamma = gamma
        self.embDim = Config.embedDim  # embedding dimension (128, 256 etc)

        # also Config.Ncls = total number of classes (categories) in dataset
        self.coarse_cls_criterion = losses.LargeMarginSoftmaxLoss(
            Config.Ncls, self.embDim, margin=self.gamma)  # large margin softmax loss
        self.mine_criterion = miners.BatchEasyHardMiner(
            pos_strategy='hard', neg_strategy='hard')  # hard mining
        print("Category loss of A,N")
        print("Gamma", self.gamma)

    def forward(self, catembA, catembN, catlabel):
        # inputs are
        # catembA = category embedding of object A
        # catembN = category embedding of object N
        # catlabel = category labels, consisting of all the labels that are used in dataset
        # e.g.
        # batch = 4, views = 3
        # catlabel = [batch x views] = [12] # so each views become training sample
        dim = catlabel.shape  # dimension of catlabel
        catlabel = catlabel.reshape(dim[0]*dim[1]).cuda()
        # if catlabel.shape = (batch, views)
        # then flatten it to treat each view as seperate sample
        # (4,3) -> (12,)

        # shape as [batch x views, d]
        catembA = catembA.reshape(dim[0]*dim[1], self.embDim)
        # e.g.
        # let say shape of catembA = (4,3,d), so it has 3 views and each views has different embeddings
        # why many views ? -> cause we're dealing with multi-views of any object
        # loss function dosen't understand views
        # it understand (samples, labels)
        # so we convert it into [batch x views, d]

        # similarly
        catembN = catembN.reshape(dim[0]*dim[1], self.embDim)

        # hard sampling for A, N
        # it'll find hard positive & hard negative pairs
        hard_sampA = self.mine_criterion(catembA, catlabel)
        hard_sampN = self.mine_criterion(catembN, catlabel)

        # large margin softmax loss
        # so for each embedding
        # it classify into correct category with margin constraint
        L_CAT_A = self.coarse_cls_criterion(catembA, catlabel, hard_sampA)
        L_CAT_N = self.coarse_cls_criterion(catembN, catlabel, hard_sampN)
        L_CAT = L_CAT_A+L_CAT_N
        return L_CAT  # total category loss for A,N
