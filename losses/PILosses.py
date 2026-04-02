
# in PiRO: Pose-invariant Representation of Object
# there are two losses that are mentioed
# pose invariant object loss (piobj) and pose-invariant category loss (picat)
# also in total loss there is also only category loss is present


# pose-invariant object loss and category loss in dual embedding space
# PiRO introduces dual embedding space for better representation, classification & retrieval of objects and categories

"""
Pose-invariant object loss in dual embedding space (piobj)
"""

import torch
# this is a collection of state-less functions that used to do stuffs with neural network
import torch.nn.functional as F
# torch.nn.functional = raw operations (functions)
# e.g.
# layer based: nn.ReLU()
# functional: F.relu(x)

import torch.nn as nn  # for neural networ layers
from utils.helperFunctions import expand_pairwise_distances  # eluclidean distances


class PILossOBJ(nn.Module):  # pose-invariant object loss for object embedding space
    def __init__(self, alpha=0.4, beta=2.0, lamda=2.0):
        super(PILossOBJ, self).__init__()
        self.alpha = alpha  # alpha is intra-class margin
        self.beta = beta  # beta is inter-class margin
        self.lamda = lamda  # weight balancing factor between clustring loss and seperation loss
        print("PI loss for object embedding space")
        print("alpha ", alpha, " beta ", beta, "lamda ", lamda)

    # for calculating loss all we need in input is the embeddings
    # embeddings of single-view, multi-view for both objects
    # as mentioned in PiRO paper for explain purpose we will take object as A,B

    def forward(self, SV_A, SV_N, MV_A, MV_N, size_average=True):
        # size_average controls how the final loss is aggregated across batches
        # if size_average = True: it takes average loss per sample
        # if size_average = False: it takes total loss over batch

        # euclidean distance between
        # single view object embedding of A and N
        # shape of SV_A : (nA,nB,d)
        # shape of SV_N: (nN,nB,d)
        dSV_A_SV_N = expand_pairwise_distances(
            SV_A, SV_N)  # output shape [nA, nN, nB]
        nA, nN, nB = dSV_A_SV_N.shape  # taking each values

        # calculating confusers of both object A,N
        # confusers: model confuses to differentiate between object A,N
        # confusers of both object A,N is the argmin of euclidean distance
        # 0 signifies for each cols (batch sample)
        confuser_A_N = torch.argmin(dSV_A_SV_N.reshape(nA*nN, nB), 0)
        # confusers are the most confused embeddings (hard negatives)
        confuser_A = confuser_A_N/nN  # confuser A (confusing view of A)
        confuser_N = confuser_A_N % nN  # confuser B (confusing view of N)

        # so
        # initially we got single-view embedding of A and N (SV_A, SV_N)
        # also we got confuser A, confuser B
        # now what if we represent confused embeddings (of A,N) in actual embedding vectors of both object A,N (SV_A, SV_N)
        # so
        # confusing embeddings of A, N
        f_confuser_A = torch.stack(
            [SV_A[int(confuser_A[i]), i, :] for i in range(nB)]
        )
        # originally the shape of SV_A is [nA,nB,d]
        # SV_A[int(confuser_A[i]),i,:] = for each batch i, it picks the most confusing view of A for that batch
        # similarly
        f_confuser_N = torch.stack(
            [SV_N[int(confuser_N[i]), i, :] for i in range(nB)]
        )

        # shape of f_confuser_A is (nB,d)
        # shape of f_confuser_N is (nB,d)

        # now calculating clustring loss
        # in clustring loss for both object A,N
        # mutli-view embeddings and confusing embeddings of both object
        """ Finding intra-class distance of both object A,N """
        A_cluster = torch.cdist(MV_A.unsqueeze(
            1), f_confuser_A.unsqueeze(1)).squeeze()
        # multi-view object embeddings for A,N is done by aggregating the single-view embeddings
        # shape of single-view embeddings of A is [nA,nB,d], for N is [nN,nB,d]
        # when we aggregate MV_A shape will become [nB,d] # one combined embedding per object

        # similary for N
        N_cluster = torch.cdist(MV_N.unsqueeze(
            1), f_confuser_N.unsqueeze(1)).squeeze()
        # shape of both clusters [nB,]
        # it will be 1D tensor containing one distance per sample

        # cluster loss of A,N
        # intra clustering loss for both A,N
        loss_A_cluster = F.relu(A_cluster - self.alpha)
        loss_N_cluster = F.relu(N_cluster - self.alpha)

        # for calculating seperation loss
        # for seperation loss we need
        # distances between confusers of A,N and distances between multi-view object embeddings of A,N
        # distance between the confusers of A,N comes from single-view object embeddings of A,N
        # we got the diff. single-view object embeddings of A,N : dSV_A_SV_N
        # for seperation loss, inter-class dist between the confusers of A,N
        AN_inter = torch.min(torch.min(dSV_A_SV_N, 0).values, 0).values

        # inter-class distance between the multi-view object embeddings of A,N
        MV_AMV_N_inter = torch.cdist(
            MV_A.unsqueeze(1), MV_N.unsqueeze(1)).squeeze()

        # seperation loss among confusers and multi-views
        loss_AN_inter = F.relu(self.beta - AN_inter)
        loss_MV_AMV_N_inter = F.relu(self.beta - MV_AMV_N_inter)

        # overall pose-invariant object loss is
        losses = self.lamda(loss_AN_inter+loss_MV_AMV_N_inter) + \
            loss_A_cluster+loss_N_cluster

        # info_quads contains
        # number of active losses: torch.numel(torch.nonzero(losses))
        # mean inter object distance
        # mean intra object distance
        # minimum inter object distance
        info_quads = [torch.numel(torch.nonzero(losses)), AN_inter.pow(0.5).mean(
        ), 0.5*(A_cluster.pow(0.5).mean()+N_cluster.pow(0.5).mean()), torch.min(AN_inter.pow(0.5))]
        avg_loss = losses.mean() if size_average else losses.sum()

        # returning pose-invariant average loss and info-quads
        return avg_loss, info_quads


"""  Pose-invariant category loss in dual-embedding space"""


class PILossCAT(nn.Module):
    def __init__(self, theta=0.4, lamda=2.0):
        super(PILossCAT, self).__init__()
        self.theta = theta  # theta is category margin
        self.lamda = lamda
        print("PI loss for category embedding space")
        print("theta", theta, "lamda ", lamda)

    def forward(self, SV_A, SV_N, MV_A, MV_N, size_avergae=True):
        # in the pose-invariant category loss
        # few things needs to calculate
        # that is mean distance between multi-view & single-view embeddings for an object x
        # shape of SV_A is [nA,nB,d]
        # shape of MV_A is [nB,d]
        # dim = 1 means row wise
        d_SV_A_MV_A = torch.mean(torch.cdist(
            SV_A, MV_A.unsqueeze(1)).squeeze(), 1)
        d_SV_N_MV_N = torch.mean(torch.cdist(
            SV_N, MV_N.unsqueeze(1)).squeeze(), 1)

        # distance between multi-view embeddings of both object A,N
        d_MV_A_MV_N = torch.cdist(
            MV_A.unsqueeze(1), MV_N.unsqueeze(1)).squeeze()

        # corresponding loss
        loss_SV_A_MV_A = F.relu(d_SV_A_MV_A - self.theta)
        loss_SV_N_MV_N = F.relu(d_SV_N_MV_N - self.theta)
        loss_MV_A_MV_N = F.relu(d_MV_A_MV_N - self.theta)

        # pose invariant category loss
        losses = loss_SV_A_MV_A + loss_SV_N_MV_N + loss_MV_A_MV_N
        avg_loss = losses.mean() if size_avergae else losses.sum()
        return avg_loss
