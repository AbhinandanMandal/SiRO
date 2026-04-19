"""
===============================================================================
Code for Pose-invariant Classification and Retrieval (PICR) using 
Pose-invariant Attention Network to learn category and object embeddings in the
same space by training jointly using L-Softmax and Pose-invariant losses
===============================================================================
"""

"""
Load Libraries
"""
from losses.PILosses import PILossOBJ # for single-embeddings,
from utils.trainLogger_update import TrainingLogger
import sys
import torch.multiprocessing
import torchvision.datasets as dset
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from models.VGG_PAN_SingleEmb import SingleModel
from losses.CategoryLoss import LossCAT
from utils.DataUtility_PiRO import OOWLTrainDataset, MNet40TrainDataset, FG3DTrainDataset, calculate_stats
from utils.InferenceUtility_large import evaluate_performance_single
from ConfigLearn_PiRO_update import ConfigOOWL, ConfigMNet40, ConfigFG3D, HyperParams # connecting trainPiRO_Singel_update with ConfigLearn_update
# paper considered only pose-invariant object loss and large-margin softmax loss, not any pose-invariant category loss
torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.__version__)

# device for model training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Load utility functions 
-------------------------------------------------------------------------------
VGG_PAN_SingleEmb : Pose-invariant Attention Network Architecture with VGG 
                    Backbone for learning embeddings in a single embedding space 
PILosses: Pose-invariant Object and Pose-invariant Category Loss
CategoryLoss: Category Loss using Large Margin Softmax Loss
DataUtility_PiRO: Custom dataloader for different datasets, and other 
                  data-related utility functions
InferenceUtility_large: Functions for inference and computation of 
                        pose-invariant recognition accuracy and retrieval mAP
helperFunctions: Other utility functions
ConfigLearn: Training and Testing Configurations for different datasets
-------------------------------------------------------------------------------
"""


# Input information and hyper parameters
dataset = sys.argv[1]  # OOWL, MNet40, FG3D
expname = sys.argv[2]  # user-specified experiment name
seed = int(sys.argv[3])  # seed
hp = HyperParams(dataset, expname, seed)

print("Large Margin Softmax Loss for classification: ",
      hp.gamma, " nHeads: ", hp.nHeads, "nLayers: ", hp.nLayers)


# Loading configuration for each dataset
if dataset == 'OOWL':
    Config = ConfigOOWL(hp.case, hp.embDim, hp.batchSize,
                        hp.alpha, hp.n_randsamp_class, hp.seed_inp)
elif dataset == 'MNet40':
    Config = ConfigMNet40(hp.case, hp.embDim, hp.batchSize,
                          hp.alpha, hp.n_randsamp_class, hp.seed_inp)
elif dataset == 'FG3D':
    Config = ConfigFG3D(hp.case, hp.embDim, hp.batchSize,
                        hp.alpha, hp.n_randsamp_class, hp.seed_inp)
else:
    print("Wrong Dataset")


""" LOGGER FOR MODEL TRAINING """
logger = TrainingLogger(Config)  # Config will take all the associates with it


train_dataset = dset.ImageFolder(root=Config.gallery_dir)  # training dataset
test_dataset = dset.ImageFolder(root=Config.probe_dir)  # testing dataset
# instantiate single embedding model
trcv_model = SingleModel(Config.inpChannel, Config.embedDim,

                         hp.nHeads, hp.nLayers, hp.dropout, Config.Ncls).to(device)
print(trcv_model)


# pose-invariant object loss &
# large margin softmax loss &
pi_obj_criterion = PILossOBJ(alpha=hp.alpha, beta=hp.beta, lamda=hp.lamda)
cat_criterion = LossCAT(Config, gamma=hp.gamma)

# optimizer &
# schedular
optimizer = optim.Adam(trcv_model.parameters(), lr=Config.LR)
scheduler = StepLR(optimizer, step_size=Config.Nepochs/5, gamma=0.5)


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
        trainData = OOWLTrainDataset(Config, name=dataset)
    elif dataset == 'MNet40':
        trainData = MNet40TrainDataset(Config, name=dataset)
    elif dataset == 'FG3D':
        trainData = FG3DTrainDataset(Config, name=dataset)
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
    orm, crm, clea, svoc = evaluate_performance_single(
        dataset, Config, trcv_model, 1)
    logger.log_epoch(epoch=epoch, train_loss=train_loss,
                     test_acc=clea, Info=Info, ratio=ratio)
    trcv_model.train()
    scheduler.step()

    # If ratio > pre-defined best_ratio then save the best model
    if ratio > logger.best_ratio:
        logger.best_ratio = ratio
        torch.save(trcv_model.state_dict(),
                   Config.best_model_path + "_best.pth")

    # If ratio > pre-defined early convergence ratio then save the model
    min_epochs = 10
    if epoch >= min_epochs and ratio > hp.ecc_ratio:
        print("Early Convergence Criterion Satisfied")
        torch.save(trcv_model.state_dict(),
                   Config.best_model_path + "_ECC.pth")
        break


"""
orm: sv object retrieval map
crm: sv class retrieval map
clea: sv class recognition accuracy
svoc: sv object recognition accuracy
mvcrm: mv class retrieval map
mvclea: mv class recognition accuracy
mvoc: mv object recogniton accuracy
mvoret: mv object retrieval map
"""

print("Evaluating final model")
trcv_model.eval()
orm, crm, clea, svoc = evaluate_performance_single(
    dataset, Config, trcv_model, 1)
mvcrm, mvclea, mvoc, mvoret = evaluate_performance_single(
    dataset, Config, trcv_model, Config.N_G)

print("Loaded model weights")

print("-------------------------------------------------------------------")
print("Test Results: \n")
print("-------------------------- Single-View ----------------------------")
print("SV Classification Accuracy: Category {} % | Object {} %|\n".format(clea, svoc))
print("SV  Retrieval mAP: Category {} %| Object {} %|\n".format(crm, orm))
print("-------------------------- Multi-View ----------------------------")
print("MV Classification Accuracy: Category {} %| Object {} %| ".format(mvclea, mvoc))
print("MV Retrieval mAP: Category {} %| Object {} %| ".format(mvcrm, mvoret))
print("-------------------------------------------------------------------")
