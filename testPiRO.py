

"""
===============================================================================
Code for evaluating trained models on single and multi-view 
pose-invariant classification and retrieval tasks  
===============================================================================
"""
"""
Load Libraries
"""

from utils.InferenceUtility_large import evaluate_performance_dual, evaluate_performance_single
from ConfigLearn import HyperParams, ConfigOOWL, ConfigMNet40, ConfigFG3D
import torch
import torchvision.datasets as dset
import sys
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')  # setting up device


"""
Load utility functions
--------------------------------------------------------------------------
SingleModel, DualModel: Pose-invariant attention network architecture with
                        VGG backbone in Single and Dual embedding space
InferenceUtility_large: Functions for inference and computation of Pose-invariant classification & retrieval (PICR) task using 
                        single & dual embedding
helperFunctions: other utility functions
ConfigLearn: Training & Testing configurations for different datasets and the
            hyper-parameters for the corresponding datasets
"""

"""
Input information and hyper-parametes from user
"""
dataset = sys.argv[1]  # taking datasets from the user input
emb_space = sys.argv[2]  # single/dual embedding space
model_path = sys.argv[3]  # model path for testing

hp = HyperParams(dataset=dataset)  # hyper-parameters
print("Large Margin Softmax Loss for Classification: ",
      hp.gamma, " nHeads: ", hp.nHeads, "nLaters: ", hp.nLayers)

"""
Loading configuration files for the dataset
"""
if dataset == "OOWL":
    Config = ConfigOOWL(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                        a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
# seed_inp = random seed uses to control the randomness of the experiment
# it makes training process reproduciable
elif dataset == "MNet40":
    Config = ConfigMNet40(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                          a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
elif dataset == "FG3D":
    Config = ConfigFG3D(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                        a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
else:
    print("Wrong Dataset")

print("Load Data")


"""
Main script for evaluation
"""

"""
Output Meaning
--------------------------------------------------
orm: sv object retrieval map
crm: sv class retrieval map
clea: sv class recognition accuracy
svoc: sv object recognition accuracy
mvcrm: mv class retrieval map
mvclea: mv class recognition accuracy
mvoc: mv object recogniton accuracy
mvoret: mv object retrieval map
"""


if emb_space == "single":
    from models.VGG_PAN_SingleEmb import SingleModel
    trained_model = SingleModel(inChannel=Config.inpChannel, embDim=Config.embedDim, nHeads=hp.nHeads,
                                nLayers=hp.nLayers, dropout=hp.dropout, nCls=Config.Ncls).to(device=device)
    print("Evaluating model trained in the single embedding space")
    trained_model.load_state_dict(torch.load(model_path)) # loading model
    trained_model.eval() # evaluating model

    # for single-view object, category recognition & retrieval
    orm, crm, clea, svoc = evaluate_performance_dual(dataset=dataset,Config=Config,trcv_model=trained_model,nview=1)
    # for multi-view object, category recognition & retrieval
    mvcrm, mvclea, mvoc, mvoret = evaluate_performance_dual(dataset=dataset,Config=Config,trcv_model=trained_model,nview=Config.N_G)

elif emb_space == "dual":
    from models.VGG_PAN_DualEmb import DualModel
    trained_model = DualModel(Config.inpChannel, Config.embedDim,
                              hp.nHeads, hp.nLayers, hp.dropout, Config.Ncls).to(device)
    print("Evaluating model trained in the dual embedding space")
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()
    orm, crm, clea, svoc = evaluate_performance_dual(
        dataset, Config, trained_model, 1)
    mvcrm, mvclea, mvoc, mvoret = evaluate_performance_dual(
        dataset, Config, trained_model, Config.N_G)
else:
    print("Wrong embedding space. enter single or dual")


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



