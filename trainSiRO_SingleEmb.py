
"""
===============================================================================
Code for State-invariant Object Representation (SIOR) using 
Pose-invariant Attention Network to learn category and object embeddings in the
same space by training jointly using L-Softmax and Pose-invariant losses and
curriculum learning for better results
===============================================================================
"""

"""
In PiRO code (https://github.com/sarkar-rohan/PiRO), 
author used pose invariant object loss and large margin softmax loss for 
single embedding model
"""


from torch import optim
from tqdm import tqdm
from ConfigLearn import ConfigOOWL, ConfigMNet40, ConfigFG3D, ConfigOWSC_SI, ConfigOWSC_GN, HyperParams
import torchvision.datasets as dset
import torch.multiprocessing
import sys
import torch
from losses.PILosses import PILossOBJ
from losses.CategoryLoss import LossCAT
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils.trainLogger_update import TrainingLogger
from models.VGG_PAN_SingleEmb import SingleModel
from utils.DataUtility import OOWLTrainDataset, MNet40TrainDataset, FG3DTrainDataset, OWSCTrainDataset, calculate_stats
from utils.InferenceUtility_SI import evaluate_SI_performance_single
# from models.VGG_PIE_SingleEmb import VGG_avg_picnn, VGG_avg_piproxy, VGG_avg_pitc
# from utils.InferenceUtility_PI import evaluate_performance_single
torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
VGG_PIE_SingleEmb: Pose Invariant Embedding for single embedding model
VGG_PAN_SingleEmb: Pose Invariant Attention Network for single embedding model
PILosses: Pose invariant losses (object loss & category loss)
CategoryLoss: Large margin softmax loss (category)
DataUtility: Custom DataUtilities (dataloader) for different datasets and other
             utility functions
InferenceUtility_PI: Functions for inference and computation of 
                     pose-invariant recognition accuracy and retrieval mAP
InferenceUtility_SI: For state invariant recognition accuracy and retrieval mAP
InferenceUtility_GN: For generalized recognition accuracy and retrieval mAP
ConfigLearn: Training and Testing Configurations for different datasets

"""

# Input information
dataset = sys.argv[1]  # OOWL, MNet40, FG3D, OWSC
expname = sys.argv[2]  # user-specified experiment name
seed = int(sys.argv[3])  # seed no
hp = HyperParams(dataset=dataset, expname=expname, seed_no=seed)

print("Large Margin Softmax Loss for classification: ",
      hp.gamma, "nHeads: ", hp.nHeads, "nLayers: ", hp.nLayers)

# Loading configuration for each dataset
if dataset == 'OOWL':
    Config = ConfigOOWL(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                        a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
elif dataset == 'MNet40':
    Config = ConfigMNet40(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                          a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
elif dataset == 'FG3D':
    Config = ConfigFG3D(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                        a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
elif dataset == 'OOWL':
    Config = ConfigOWSC_SI(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                           a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
else:
    print("Wrong Dataset")
# ConfigOWSC_GN is for generalization purpose, will be evaluated during testing

""" Training Logger For Model Training """
logger = TrainingLogger(Config=Config)

train_dataset = dset.ImageFolder(root=Config.gallery_dir)
test_dataset = dset.ImageFolder(root=Config.probe_dir)
trcv_model = SingleModel(inChannel=Config.inpChannel, embDim=Config.embedDim, nHeads=hp.nHeads,
                         # initializing single embedding model
                         nLayers=hp.nLayers, dropout=hp.dropout, nCls=Config.Ncls).to(device=device)
print(trcv_model)

"""
In PiRO code (https://github.com/sarkar-rohan/PiRO), 
author used pose invariant object loss and large margin softmax loss for 
single embedding model
"""
pi_obj_criterion = PILossOBJ(alpha=hp.alpha, beta=hp.beta, lamda=hp.lamda)
cat_criterion = LossCAT(Config=Config, gamma=hp.gamma)


# Initializing Optimizer & Scheduler
optimizer = optim.Adam(trcv_model.parameters(), lr=Config.LR)
scheduler = StepLR(optimizer=optimizer, step_size=Config.Nepochs/5, gamma=0.5)


# Single Embedding Model Training Loop
def train(epoch):
    sum_loss = 0.0
    avg_loss = 0.0
    infoQuads = 0

    """
    calculate_stats() function take input model, config, dataset and embedding space
    it returns
    1. maxdisintra = maximum distance within object
    2. disintra = average distance within object
    3. interCN_catg = average distance between obj i and other obj of the same class
    4. interCN_min = minimum dist between obj i and other obj of the same class
    """
    info = calculate_stats(trcv_model, Config, dataset, 'single')

    # choosing training dataset according to user input
    if dataset == 'OOWL':
        trainData = OOWLTrainDataset(Config=Config, name=dataset)
    elif dataset == 'MNet40':
        trainData = MNet40TrainDataset(Config=Config, name=dataset)
    elif dataset == 'FG3D':
        trainData = FG3DTrainDataset(Config=Config, name=dataset)
    elif dataset == 'OWSC':
        trainData = OWSCTrainDataset(Config=Config, name=dataset)
    else:
        print("Wrong Dataset")

    # loading data into dataloader
    tdataloader = DataLoader(trainData, shuffle=True,
                             num_workers=16, batch_size=Config.BS)

    trainloop = tqdm(tdataloader, leave=False)
    trcv_model.train()

    for data in trainloop:
        # for each training dataset
        # it will return 3 things
        # ximage, pimage, label (embedding of x, embedding of p, label category)
        # so, for each data in training loop, we can break data into 3 outputs
        # embedding of A, N and label category
        I_A, I_N, label_category = data

        # loading embeddings and label category into device
        I_A = I_A.to(device)
        I_N = I_N.to(device)
        label_category = label_category.to(device)

        optimizer.zero_grad()

        # SingleModel() returns 3 things
        # SVOBJEmbs = single view obj embeddings
        # MVOBJEmbs = multi view obj embeddings
        # objAttn = object attention
        SV_A, MV_A, _ = trcv_model(I_A)
        SV_N, MV_N, _ = trcv_model(I_N)

        # pose-invariant object loss & large margin softmax loss
        L_PiOBJ, IQuads = pi_obj_criterion(
            SV_A.transpose(1, 0), SV_N.transpose(1, 0), MV_A, MV_N)
        L_CAT = cat_criterion(SV_A, SV_N, label_category)

        infoQuads += (IQuads[0]/hp.batchSize)

        # this is hyperparameter part for specifying out which particular loss
        # the training model use
        # if its 'CAT' then it will use only large margin softmax loss
        # if its 'OBJ' then it will use pose invariant obj loss
        # if its 'JNT' then both large margin and pose invariant loss
        if hp.task == "CAT":
            L = L_CAT
        elif hp.task == "OBJ":
            L = L_PiOBJ
        elif hp.task == "JNT":
            L = L_CAT + L_PiOBJ
        else:
            print("Wrong task")

        sum_loss += L.item()
        L.backward()
        optimizer.step()
        trainloop.set_postfix(L_cat=(L_CAT).item(), L_piobj=L_PiOBJ.item())

    avg_loss = sum_loss/len(tdataloader)
    Info = []  #
    avg_infoQuads = infoQuads/len(tdataloader)
    Info = [avg_infoQuads, info[0], info[1], info[2], info[3]]
    # info[0] = maxdisintra = maximum distance within object
    # info[1] = disintra = avergae distance within object
    # info[2] = interCN_catg = average distance with obj i and other obj of the same class
    # info[3] = interCN_min = minimum distance with obj i and other obj of the same class
    return avg_loss, Info  # returning avg loss and Info of model training


for epoch in range(Config.Nepochs):
    train_loss, Info = train(epoch=epoch)
    ratio = Info[4]/Info[1]  # ratio between interCN_min & maxdisintra

    # save model at each epoch
    torch.save(trcv_model.state_dict(),
               Config.save_model_path + f"_epoch{epoch}.pth")

    print("-------------------------------------------------------------------")
    print("Epoch {} | LR {} | Train loss {} |\n".format(
        epoch, scheduler.get_lr(), train_loss))
    print("Exemplar Information {} %|\n".format(Info))
    print("Ratio: ", Info[4]/Info[1])
    print("-------------------------------------------------------------------")

    # storing logger points for evaluation of model's performance
    trcv_model.eval()
    # orm, crm, clea, svoc = evaluate_performance_single(
    #     dataset, Config, trcv_model, 1)
    SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc = evaluate_SI_performance_single(
        dataset=dataset, Config=Config, trcv_model=trcv_model, nview=1)

    logger.log_epoch(epoch=epoch, train_loss=train_loss,
                     test_acc=SV_C_acc, Info=Info, ratio=ratio)
    trcv_model.train()
    scheduler.step()

    # If ratio > pre-defined best_ratio then save the best model
    if ratio > logger.best_ratio:
        logger.best_ratio = ratio
        torch.save(trcv_model.state_dict(),
                   Config.best_model_path + "_best.pth")

    # If ratio > pre-defined early convergence ratio  and epochs >= min_epochs then save the model
    min_epochs = 10
    if epoch >= min_epochs and ratio > hp.ecc_ratio:
        print("Early Convergence Criterion Satisfied")
        torch.save(trcv_model.state_dict(),
                   Config.best_model_path + "_ECC.pth")
        break


"""
SV_C_mAP : single view class retrieval accuracy
SV_C_acc : single view class recognition accuracy
SV_O_mAP : single view object retrieval accuracy
SV_O_acc : single view object recognition accuracy
MV_C_acc :
MV_C_mAP :
MV_O_acc :
MV_O_mAP :
"""

print("Evaluating Final Model")
trcv_model.eval()
SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc = evaluate_SI_performance_single(
    dataset=dataset, Config=Config, trcv_model=trcv_model, nview=1)
MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc = evaluate_SI_performance_single(
    dataset=dataset, Config=Config, trcv_model=trcv_model, nview=Config.N_G)

print("Loaded model weights")

print("-------------------------------------------------------------------")
print("Test Results: \n")
print("-------------------------- Single-View ----------------------------")
print("SV Classification Accuracy: Category {} % | Object {} %|\n".format(
    SV_C_acc, SV_O_acc))
print("SV  Retrieval mAP: Category {} %| Object {} %|\n".format(SV_C_mAP, SV_O_mAP))
print("-------------------------- Multi-View ----------------------------")
print("MV Classification Accuracy: Category {} %| Object {} %| ".format(
    MV_C_acc, MV_O_acc))
print("MV Retrieval mAP: Category {} %| Object {} %| ".format(MV_C_mAP, MV_O_mAP))
print("-------------------------------------------------------------------")
