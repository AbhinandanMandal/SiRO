

# updated code of dual embd space
"""
===============================================================================
Code for Pose-invariant Classification and Retrieval (PICR) using 
Pose-invariant Attention Network to learn dual category and object embeddings
simultaneously by training jointly using L-Softmax and Pose-invariant losses
===============================================================================
"""

"""
Loading utility functions
--------------------------------------------------------------------------------------------
VGG_PAN_DualEmb: Pose-invariant attention network architecture with VGG backbone for learning embedding
                in dual embedding space(object embedding & category embedding)

PILosses: Pose-invariant object and Pose-invariant category losses
CategoryLoss: Category loss using large-margin softmax loss
DataUtility_PiRO: Custom dataloader for different datasets and other data-related utility functions
InferenceUtility_large: Functions for inference and computation of
                        pose-invariant recognition accuracy and retrieval mAP
helperFunctions: Other utility functions
ConfigLearn: Training & Testing configurations for different datasets
"""


# loading libraries
from utils.InferenceUtility_large import evaluate_performance_dual
from utils.trainLogger_update import TrainingLogger
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from ConfigLearn_update import ConfigMNet40, ConfigOOWL, ConfigFG3D, HyperParams
from utils.DataUtility_PiRO import OOWLTrainDataset, MNet40TrainDataset, FG3DTrainDataset, calculate_stats
from torch.optim.lr_scheduler import StepLR
from torch import optim
from losses.CategoryLoss import LossCAT
from losses.PILosses import PILossCAT, PILossOBJ
import torchvision.datasets as dset
from models.VGG_PAN_DualEmb import DualModel
import torch.multiprocessing
import sys
torch.multiprocessing.set_sharing_strategy('file_system')

print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')  # setting up device


# input information & hyperparameters
dataset = sys.argv[1]  # OOWL, MNet40, FG3D
expname = sys.argv[2]
seed = int(sys.argv[3])
hp = HyperParams(dataset=dataset, expname=expname, seed_no=seed)
print("Large Margin Softmax Loss for classification: ",
      hp.gamma, "nHeads: ", hp.nHeads, "nLayers: ", hp.nLayers)


# loading configuration files for the datasets
if dataset == "OOWL":
    Config = ConfigOOWL(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                        a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
elif dataset == "MNet40":
    Config = ConfigMNet40(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                          a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
elif dataset == "FG3D":
    Config = ConfigMNet40(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                          a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
else:
    print("Wrong Dataset")


""" DUAL EMBD MODEL TRAINING LOGGER """
logger = TrainingLogger(Config=Config)

# loading training, testing dataset and initiate model
train_dataset = dset.ImageFolder(root=Config.gallery_dir)  # gallery
test_dataset = dset.ImageFolder(root=Config.probe_dir)  # probe
trcv_model = DualModel(inChannel=Config.inpChannel, embDim=Config.embedDim,
                       nHeads=hp.nHeads, nLayers=hp.nLayers, dropout=hp.dropout, nCls=Config.Ncls)
print(trcv_model)


# loading pose-invariant object, category & large margin softmax loss
pi_obj_criterion = PILossOBJ(alpha=hp.alpha, beta=hp.beta, lamda=hp.lamda)
pi_cat_criterion = PILossCAT(theta=hp.theta, lamda=hp.lamda)
cat_criterion = LossCAT(Config=Config, gamma=hp.gamma)

# loading optimizer & schedular
optimizer = optim.Adam(trcv_model.parameters(), lr=Config.LR)
scheduler = StepLR(optimizer=optimizer, step_size=Config.Nepochs/5, gamma=0.5)

# training loop


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
    info = calculate_stats(net=trcv_model, Config=Config,
                           dataset=dataset, emb_space='dual')

    # loading training dataset as per user input
    if dataset == "OOWL":
        trainData = OOWLTrainDataset(Config=Config, name=dataset)
    elif dataset == "MNet40":
        trainData = MNet40TrainDataset(Config=Config, name=dataset)
    elif dataset == "FG3D":
        trainData = FG3DTrainDataset(Config=Config, name=dataset)
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

        # DualModel() returns
        # single-view, multi-view object & category embeddings and object attention, class attention
        SV_OBJ_A, SV_CAT_A, MV_OBJ_A, MV_CAT_A, _, _ = trcv_model(I_A)
        SV_OBJ_N, SV_CAT_N, MV_OBJ_N, MV_CAT_N, _, _ = trcv_model(I_N)

        # pose-invariant object loss and info quads
        L_PiOBJ, IQuads = pi_obj_criterion(SV_OBJ_A.transpose(
            1, 0), SV_OBJ_N.transpose(1, 0), MV_OBJ_A, MV_OBJ_N)
        # pose-invariant category loss
        L_PiCAT = pi_cat_criterion(SV_CAT_A, SV_CAT_N, MV_CAT_A, MV_CAT_N)
        # large-margin softmax category loss
        L_CAT = cat_criterion(SV_CAT_A, SV_CAT_N, label_category)

        infoQuads += (IQuads[0]/hp.batchSize)

        # this is hyperparameter part for specifying out which particular loss
        # the training model use
        # if its 'CAT' then it will use only large margin softmax loss
        # if its 'OBJ' then it will use pose invariant obj loss
        # if its 'JNT' then both large margin and pose invariant losses

        if hp.task == "CAT":
            L = L_CAT
        elif hp.task == "OBJ":
            L = L_PiOBJ
        elif hp.task == "JNT":
            L = L_CAT+L_PiCAT+L_PiOBJ
        else:
            print("Wrong Task")

        sum_loss += L.item()
        L.backward()
        optimizer.step()
        trainloop.set_postfix(L_cat=(L_CAT).item(),
                              L_picat=L_PiCAT.item(), L_piobj=L_PiOBJ.item())

    avg_loss = sum_loss/len(tdataloader)
    Info = []
    avg_infoQuads = infoQuads/len(tdataloader)
    Info = [avg_infoQuads, info[0], info[1], info[2], info[3]]
    # info[0] = maxdisintra = maximum distance within object
    # info[1] = disintra = avergae distance within object
    # info[2] = interCN_catg = average distance with obj i and other obj of the same class
    # info[3] = interCN_min = minimum distance with obj i and other obj of the same class
    return avg_loss, Info


for epoch in range(Config.Nepochs):
    train_loss, Info = train(epoch=epoch)
    ratio = Info[4]/Info[1]  # ratio between interCN_min & maxdisintra

    # saving model at each epoch
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
    orm, crm, clea, svoc = evaluate_performance_dual(
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

    # # If ratio > pre-defined early convergence ratio then save the model
    # if ratio > hp.ecc_ratio:
    #     print("Early Convergence Criterion Satisfied")
    #     torch.save(trcv_model.state_dict(),
    #                Config.best_model_path + "_ECC.pth")
    #     break

    # considering minimum number of epochs to run
    min_epochs = 10
    if epoch >= min_epochs and ratio > hp.ecc_ratio:
        print("Early Convergence Criterion Satisfied")
        torch.save(trcv_model.state_dict(), Config.best_model_path+"_ECC.pth")
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

orm, crm, clea, svoc = evaluate_performance_dual(
    dataset=dataset, Config=Config, trcv_model=trcv_model, nview=1)
mvcrm, mvclea, mvoc, mvoret = evaluate_performance_dual(
    dataset=dataset, Config=Config, trcv_model=trcv_model, nview=Config.N_G)

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
