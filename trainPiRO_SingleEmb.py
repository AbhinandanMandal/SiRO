

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
from ConfigLearn_PiRO import HyperParams
from ConfigLearn_PiRO import ConfigOOWL, ConfigMNet40, ConfigFG3D
from utils.InferenceUtility_large import evaluate_performance_single
from utils.helperFunctions import plot_distance, plot_infoex
from utils.DataUtility_PiRO import OOWLTrainDataset, MNet40TrainDataset, FG3DTrainDataset, calculate_stats
from losses.CategoryLoss import LossCAT
from losses.PILosses import PILossOBJ, PILossCAT
from models.VGG_PAN_SingleEmb import SingleModel
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import optim
import torch
import torchvision.datasets as dset
import sys
import torch.multiprocessing
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

"""
Input information and hyper-parameters from user
"""
dataset = sys.argv[1]  # OOWL, MNet40, FG3D
expname = sys.argv[2]  # user-specified experiment name
seed = int(sys.argv[3])  # seed
hp = HyperParams(dataset, expname, seed)

print("Large Margin Softmax Loss for classification: ",
      hp.gamma, " nHeads: ", hp.nHeads, "nLayers: ", hp.nLayers)
"""
Load configuration files for the dataset
"""
# Datasets for testing PICR
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


"""
Load gallery and probe datasets and instantiate network
"""
train_dataset = dset.ImageFolder(root=Config.gallery_dir)
test_dataset = dset.ImageFolder(root=Config.probe_dir)
# instantiate model
trcv_model = SingleModel(Config.inpChannel, Config.embedDim,
                         hp.nHeads, hp.nLayers, hp.dropout, Config.Ncls).to(device)
print(trcv_model)


"""
===============================================================================
                             Training framework
===============================================================================
"""

"""
    Set up loss, optimizer and scheduler for training
"""
train_loss_history = []
test_acc_history = []
useful_exemplar_history = []
inter_dist_history = []
min_inter_dist_history = []
intra_dist_history = []
max_intra_dist_history = []
ratio_history = []
"""
Define losses for training encoder
"""
""" Use PI Loss for separating objects from same category in 
    the Object Embedding Space """
pi_obj_criterion = PILossOBJ(alpha=hp.alpha, beta=hp.beta, lamda=hp.lamda)
""" Use LM Softmax loss for separating embeddings of different categories in 
    the Category Embedding Space """
cat_criterion = LossCAT(Config, gamma=hp.gamma)
""" Use PI Loss for clustering objects from same category in 
    the Category Embedding Space """
pi_cat_criterion = PILossCAT(theta=hp.theta, lamda=hp.lamda)

"""
Define optimizer and scheduler
"""
optimizer = optim.Adam(trcv_model.parameters(), lr=Config.LR)
scheduler = StepLR(optimizer, step_size=Config.Nepochs/5, gamma=0.5)


def train(epoch):
    sum_loss = 0.0
    avg_loss = 0.0
    infoQuads = 0
    """
    Compute stats for early convergence
    """
    info = calculate_stats(trcv_model, Config, dataset, 'single')

    """
    In each epoch randomly choose classes from same category for comparison 
    and generate multi-view training batches 
    """
    if dataset == 'OOWL':
        trainData = OOWLTrainDataset(Config, name=dataset)
    elif dataset == 'MNet40':
        trainData = MNet40TrainDataset(Config, name=dataset)
    elif dataset == 'FG3D':
        trainData = FG3DTrainDataset(Config, name=dataset)
    else:
        print("Wrong Dataset")
    tdataloader = DataLoader(trainData, shuffle=True,
                             num_workers=16, batch_size=Config.BS)
    """
    Training PAN - Dual Embeddings jointly using losses
    """

    trainloop = tqdm(tdataloader, leave=False)
    trcv_model.train()

    for data in trainloop:
        I_A, I_N, label_category = data

        # loading embeddings & label category into device
        I_A = I_A.to(device)
        I_N = I_N.to(device)
        label_category = label_category.to(device)

        optimizer.zero_grad()

        SV_A, MV_A, _ = trcv_model(I_A)
        SV_N, MV_N, _ = trcv_model(I_N)

        L_PiOBJ, IQuads = pi_obj_criterion(
            SV_A.transpose(1, 0), SV_N.transpose(1, 0), MV_A, MV_N)

        L_CAT = cat_criterion(SV_A, SV_N, label_category)

        infoQuads += (IQuads[0]/hp.batchSize)
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

    """
    Computing average loss
    """
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
    train_loss, Info = train(epoch)
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
        print("Early Convergence Criterion Satisfied. Ending training!")
        torch.save(trcv_model.state_dict(), best_name+'1.5EC.pth')
        break

    """ Plot distances, convergence ratio, stats. """
    plot_distance(useful_exemplar_history, max_intra_dist_history,
                  min_inter_dist_history, ratio_history, Config.save_plot_dist_path)
    plot_infoex(useful_exemplar_history, Config.save_plot_learn_path)
    print("Saved plots")
    print(Config.save_plot_dist_path)
    print(Config.save_plot_learn_path)


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
