
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
                in dual embedding space (object embedding & category embedding)

PILosses: Pose-invariant object and Pose-invariant category losses
CategoryLoss: Category loss using large-margin softmax loss
DataUtility_PiRO: Custom dataloader for different datasets and other data-related utility functions
InferenceUtility_large: Functions for inference and computation of 
                        pose-invariant recognition accuracy and retrieval mAP
helperFunctions: Other utility functions
ConfigLearn: Training & Testing configurations for different datasets
"""


# Loading libraries
from utils.InferenceUtility_large import evaluate_performance_dual
from utils.helperFunctions import plot_distance, plot_infoex
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.DataUtility_PiRO import OOWLTrainDataset, MNet40TrainDataset, FG3DTrainDataset, calculate_stats
from torch.optim.lr_scheduler import StepLR
from torch import optim
from losses.CategoryLoss import LossCAT
from losses.PILosses import PILossCAT, PILossOBJ
import torchvision.datasets as dset
from models.VGG_PAN_DualEmb import DualModel
import sys
from ConfigLearn import ConfigOOWL, ConfigFG3D, ConfigMNet40
from ConfigLearn import HyperParams # for hyperparameters
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.__version__)

# setting up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = sys.argv[1]  # OOWL, MNet40, FG3D
expname = sys.argv[2]
seed = int(sys.argv[3])
hp = HyperParams(dataset=dataset, expname=expname, seed_no=seed)
print("Large Margin Softmax Loss for classification: ",
      hp.gamma, "nHeads: ", hp.nHeads, "nLayers: ", hp.nLayers)


"""
Loading configuration files for the dataset
"""
if dataset == "OOWL":
    Config = ConfigOOWL(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                        a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
elif dataset == "MNet40":
    Config = ConfigMNet40(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                          a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
elif dataset == "FG3D":
    Config = ConfigFG3D(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                        a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
else:
    print("Wrong Dataset")


"""
Loading gallery (training) & probe (testing) datasets and instantiate network
"""
train_dataset = dset.ImageFolder(root=Config.gallery_dir)
test_dataset = dset.ImageFolder(root=Config.probe_dir)
# Instantiate model
trcv_model = DualModel(Config.inpChannel, Config.embedDim,
                       hp.nHeads, hp.nLayers, hp.dropout, Config.Ncls)
print(trcv_model)


"""
===============================================================================
                             Training framework
===============================================================================
"""
"""
Setting up loss, optimizer, scheduler for training
"""

# containers for storing
train_loss_history = []
test_acc_history = []
useful_exemplar_history = []
inter_dist_history = []  # distance between the objects of the diff class
min_inter_dist_history = []  # min distance objects of the diff class
intra_dist_history = []  # dist between the object of the same class
max_intra_dist_history = []  # max dist between the objects of the same class
ratio_history = []

"""Defining losses"""
pi_obj_criterion = PILossOBJ(alpha=hp.alpha, beta=hp.beta,
                             lamda=hp.lamda)  # pose-invariant object loss of the same category
pi_cat_criterion = PILossCAT(theta=hp.theta, lamda=hp.lamda)
# large-margin softmax category loss
cat_criterion = LossCAT(Config=Config, gamma=hp.gamma)

"""Optimizer and Scheduler"""
optimizer = optim.Adam(trcv_model.parameters(), lr=Config.LR)
scheduler = StepLR(optimizer=optimizer, step_size=Config.Nepochs/5, gamma=0.5)


# Model Training Loop
def train(epoch):
    sum_loss = 0.0
    avg_loss = 0.0
    infoQuads = 0

    # Computing stats for early convergence
    info = calculate_stats(net=trcv_model, Config=Config,
                           dataset=dataset)  # for dual embedding space

    """
    In each epoch randomly choose classes from same category for comparison 
    and generate multi-view training batches 
    """
    if dataset == "OOWL":
        trainData = OOWLTrainDataset(Config=Config, name=dataset)
    elif dataset == "MNet40":
        trainData = MNet40TrainDataset(Config=Config, name=dataset)
    elif dataset == "FG3D":
        trainData = FG3DTrainDataset(Config=Config, name=dataset)
    else:
        print("Wrong Dataset")

    tdataloader = DataLoader(
        dataset=trainData, shuffle=True, num_workers=16, batch_size=Config.BS)

    """
    Training PAN (Pose-invariant Attention Network) - Dual Embeddings jointly using losses
    """
    trainloop = tqdm(tdataloader, leave=False)
    trcv_model.train()

    for data in trainloop:
        I_A, I_N, label_category = data

        # loading embeddings and label categories into device
        I_A = I_A.to(device)
        I_N = I_N.to(device)
        label_category = label_category.to(device)

        optimizer.zero_grad()

        # single-view and multi-view object and category embeddings
        SV_OBJ_A, SV_CAT_A, MV_OBJ_A, MV_CAT_A, _, _ = trcv_model(I_A)
        SV_OBJ_N, SV_CAT_N, MV_OBJ_N, MV_CAT_N, _, _ = trcv_model(I_N)

        # pose-invariant object loss and info quads
        L_PiOBJ, IQuads = pi_obj_criterion(SV_OBJ_A.transpose(
            1, 0), SV_OBJ_N.transpose(1, 0), MV_OBJ_A, MV_OBJ_N)
        # pose-invariant category loss
        L_PiCAT = pi_cat_criterion(SV_CAT_A, SV_CAT_N, MV_CAT_A, MV_CAT_N)
        # large-margin softmax category loss
        L_CAT = cat_criterion(SV_CAT_A, SV_CAT_N, label_category)

        infoQuads += (IQuads[0]/hp.batchSize)  # info quads
        if hp.task == "CAT":
            L = L_CAT
        elif hp.task == "OBJ":
            L = L_PiOBJ
        elif hp.task == "JNT":
            L = L_CAT+L_PiCAT+L_PiOBJ
        else:
            print("Wrong Task")
        sum_loss += L.item()

        L.backward()  # backpropagation
        optimizer.step()
        # display additional information at the end of a progress bar while a loop is running.
        trainloop.set_postfix(L_cat=(L_CAT).item(),
                              L_picat=L_PiCAT.item(), L_piobj=L_PiOBJ.item())

    # Compute average loss
    avg_loss = sum_loss/len(tdataloader)
    Info = []
    avg_infoQuads = infoQuads/len(tdataloader)
    Info = [avg_infoQuads, info[0], info[1], info[2], info[3]]
    return avg_loss, Info


"""
Main script for learning
"""

name = Config.save_model_path + '_' + hp.expname + hp.task+'_'+str(hp.alpha)+str(hp.beta)+str(
    hp.gamma) + '_' + str(hp.nHeads) + '_' + str(hp.nLayers) + '-' + str(hp.embDim)+'.pth'
best_name = Config.best_model_path + '_' + hp.expname + hp.task+'_'+str(hp.alpha)+str(
    hp.beta)+str(hp.gamma) + '_' + str(hp.nHeads) + '_' + str(hp.nLayers) + '-' + str(hp.embDim)
print(name)
print(best_name)

for epoch in range(0, Config.Nepochs):
    train_loss, Info = train(epoch=epoch)
    torch.save(trcv_model.state_dict(), name)

    print("-------------------------------------------------------------------")
    print("Epoch {} | LR {} | Train loss {} |\n".format(
        epoch, scheduler.get_lr(), train_loss))
    print("Exemplar Information {} %|\n".format(Info))
    print("Ratio: ", Info[4]/Info[1])
    print("-------------------------------------------------------------------")

    train_loss_history.append(train_loss)
    useful_exemplar_history.append(Info[0]*100)
    max_intra_dist_history.append(Info[1])
    intra_dist_history.append(Info[2])
    inter_dist_history.append(Info[3])
    min_inter_dist_history.append(Info[4])
    # Compute Ratio for Early Convergence
    ratio = Info[4]/Info[1]
    ratio_history.append(ratio)
    scheduler.step()

    if ratio > hp.ecc_ratio:
        print("Early Convergence Criterion Satisfied. Ending Training!")
        torch.save(trcv_model.state_dict(), best_name+'1.5EC.pth')
        break

    """ Plotting distance, convergence ratio, stats. """
    plot_distance(useful_exemplar_history, max_intra_dist_history,
                  min_inter_dist_history, ratio_history, Config.save_plot_dist_path)
    plot_infoex(useful_exemplar_history, Config.save_plot_learn_path)
    print("Saved Plots")
    print(Config.save_plot_dist_path)
    print(Config.save_plot_learn_path)


print("Evaluating final model")
trcv_model.eval()

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

