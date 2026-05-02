
"""
===============================================================================
Code for State-invariant Representation of Object (SiRO) using 
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


from utils.DataUtility import loadDataset, calculate_stats
from utils.InferenceUtility_SI import evaluate_SI_performance_single
from models.VGG_PAN_SingleEmb import SingleModel
from utils.trainLogger_update import TrainingLogger
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from losses.CategoryLoss import LossCAT
from losses.PILosses import PILossOBJ
import torch
import sys
import torch.multiprocessing
from ConfigLearn import ConfigOOWL, ConfigMNet40, ConfigFG3D, ConfigOWSC_SI, ConfigOWSC_GN, HyperParams
from torch import optim
from tqdm import tqdm
from PIL import Image
import PIL.ImageOps
import numpy as np
import random
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
elif dataset == 'OWSC':
    Config = ConfigOWSC_SI(case=hp.case, edim=hp.embDim, bs=hp.batchSize,
                           a=hp.alpha, n_s=hp.n_randsamp_class, seed_no=hp.seed_inp)
else:
    print("Wrong Dataset")
# ConfigOWSC_GN is for generalization purpose, will be evaluated during testing

""" Training Logger For Model Training """
logger = TrainingLogger(Config=Config)

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


class CurriculumPairDataset(Dataset):
    """
    Pair sampler for the three curriculum strategies described in the SiRO paper.

    S1: random object from the same category.
    S2: similar object from the same category, mined from current embeddings.
    S3: similar object from any category, mined from current embeddings.
    """

    def __init__(self, Config, dataset_name, strategy="S1", mined_pairs=None):
        self.Config = Config
        self.transform = Config.train_dataAug
        self.should_invert = False
        self.dpath = Config.gallery_dir
        self.N_class = Config.Ntrain
        self.N_comp = Config.Ncomp
        self.obj2cls = Config.o2ctrain
        self.cls2obj = Config.class_list
        self.dataset = dataset_name
        self.N_G = Config.N_G
        self.gal_vp = Config.gal_vp
        self.strategy = strategy
        self.mined_pairs = mined_pairs or {}
        print("Curriculum sampling strategy:", self.strategy)

    def __len__(self):
        return self.N_class*self.N_comp

    def applyTransform(self, img_path):
        img = Image.open(img_path)
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _sample_views(self, obj):
        if self.dataset == "OWSC":
            views = list(self.gal_vp[obj+1])
            if len(views) >= self.N_G:
                return random.sample(views, self.N_G)
            return random.choices(views, k=self.N_G)

        views = list(self.gal_vp)
        if len(views) >= self.N_G:
            return random.sample(views, self.N_G)
        return random.choices(views, k=self.N_G)

    def _fallback_any_object(self, objx):
        candidates = [obj for obj in range(self.N_class) if obj != objx]
        return random.choice(candidates)

    def _random_same_category(self, objx):
        cls = int(self.obj2cls[objx])
        candidates = self.cls2obj[cls].copy()
        if objx in candidates:
            candidates.remove(objx)
        if len(candidates) == 0:
            return self._fallback_any_object(objx)
        return random.choice(candidates)

    def _path_for(self, obj, view):
        if self.dataset == "OOWL":
            return self.dpath+str(obj+1)+"/"+str(view)+".jpg"
        if self.dataset == "MNet40":
            return self.dpath+str(obj+1)+"/"+str(view).zfill(3)+".jpg"
        if self.dataset == "FG3D":
            return self.dpath+str(obj+1)+"/"+str(view).zfill(3)+".png"
        if self.dataset == "OWSC":
            return view
        raise ValueError("Dataset not recognized.")

    def _mined_pair(self, objx):
        candidates = self.mined_pairs.get(objx, [])
        if len(candidates) == 0:
            return self._random_same_category(objx)
        return random.choice(candidates)

    def __getitem__(self, index):
        objx = int(index/self.N_comp)
        if self.strategy == "S1":
            objp = self._random_same_category(objx)
        else:
            objp = self._mined_pair(objx)

        cls_x = int(self.obj2cls[objx])
        cls_p = int(self.obj2cls[objp])
        ximage = list()
        pimage = list()
        label_x = list()
        label_p = list()

        sampled_views_x = self._sample_views(objx)
        sampled_views_p = self._sample_views(objp)
        n_views = min(len(sampled_views_x), len(sampled_views_p), self.N_G)

        for i in range(n_views):
            ximage.append(self.applyTransform(
                self._path_for(objx, sampled_views_x[i])))
            pimage.append(self.applyTransform(
                self._path_for(objp, sampled_views_p[i])))
            label_x.append(torch.from_numpy(np.array(cls_x)))
            label_p.append(torch.from_numpy(np.array(cls_p)))

        return (torch.stack(ximage), torch.stack(pimage),
                torch.stack(label_x), torch.stack(label_p))


def extract_multi_image_embeddings(net, Config, dataset_name):
    """Return one learned object embedding per training object."""
    trainData = loadDataset(Config, 'val', dataset_name)
    trainLoader = DataLoader(trainData, batch_size=1,
                             shuffle=False, num_workers=16)
    object_embeddings = []
    net.eval()

    with torch.no_grad():
        for ref_data, _, _ in tqdm(trainLoader, leave=False):
            ref_data = ref_data.to(device)
            _, mv_emb, _ = net(ref_data)
            mv_emb = mv_emb.squeeze()
            if mv_emb.ndim > 1:
                mv_emb = mv_emb.mean(dim=0)
            object_embeddings.append(mv_emb.detach().cpu())

    return torch.stack(object_embeddings, dim=0)


def curriculum_strategy(epoch):
    """
    Gradually move from easy random same-category pairs to mined confusing pairs.

    The first epoch follows the paper exactly with S1. Later epochs rotate through
    S1, S2 and S3, so training keeps seeing diverse positives while increasingly
    receiving hard same-category and cross-category object pairs.
    """
    if epoch == 0:
        return "S1"
    return ("S1", "S2", "S3")[(epoch-1) % 3]


def curriculum_partition_count(epoch, Config):
    """
    Progressive n_e schedule from the paper: start coarse and use finer
    partitions as training proceeds.
    """
    n_min = 2
    n_max = max(n_min, Config.Ntrain)
    part_const = getattr(Config, "part_const", 2)
    n_epoch = int(part_const*(epoch+1))
    return min(n_max, max(n_min, n_epoch))


def build_curriculum_pairs(embeddings, Config, strategy, epoch):
    """
    Mine neighbors in the currently learned embedding space.

    S2 restricts neighbors to the same category. S3 allows any category and uses
    progressive embedding partitions so later epochs mine closer, harder pairs.
    """
    if strategy == "S1":
        return {}

    labels = np.asarray(Config.o2ctrain)
    distances = torch.cdist(embeddings, embeddings, p=2).numpy()
    np.fill_diagonal(distances, np.inf)
    mined_pairs = {}

    if strategy == "S2":
        for obj in range(Config.Ntrain):
            cls = labels[obj]
            candidates = [x for x in Config.class_list[cls] if x != obj]
            candidates = sorted(candidates, key=lambda x: distances[obj, x])
            mined_pairs[obj] = candidates[:max(1, Config.Ncomp)]
        return mined_pairs

    if strategy == "S3":
        n_partitions = curriculum_partition_count(epoch, Config)
        partition_size = max(Config.Ncomp, int(
            np.ceil(Config.Ntrain/n_partitions)))
        for obj in range(Config.Ntrain):
            candidates = np.argsort(distances[obj])[:partition_size].tolist()
            mined_pairs[obj] = candidates[:max(1, Config.Ncomp)]
        return mined_pairs

    raise ValueError("Unknown curriculum strategy: {}".format(strategy))


def large_margin_category_loss(cat_criterion, emb_a, emb_n, label_a, label_n):
    """
    LossCAT assumes both objects in a pair share one category label. Curriculum S3
    can pair objects across categories, so this applies the same classifier to
    each side with the correct labels.
    """
    dim_a = label_a.shape
    dim_n = label_n.shape
    label_a = label_a.reshape(dim_a[0]*dim_a[1]).to(device)
    label_n = label_n.reshape(dim_n[0]*dim_n[1]).to(device)
    emb_a = emb_a.reshape(dim_a[0]*dim_a[1], cat_criterion.embDim)
    emb_n = emb_n.reshape(dim_n[0]*dim_n[1], cat_criterion.embDim)

    hard_samp_a = cat_criterion.mine_criterion(emb_a, label_a)
    hard_samp_n = cat_criterion.mine_criterion(emb_n, label_n)
    loss_a = cat_criterion.coarse_cls_criterion(emb_a, label_a, hard_samp_a)
    loss_n = cat_criterion.coarse_cls_criterion(emb_n, label_n, hard_samp_n)
    return loss_a+loss_n


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

    strategy = curriculum_strategy(epoch)
    object_embeddings = extract_multi_image_embeddings(
        trcv_model, Config, dataset)
    mined_pairs = build_curriculum_pairs(
        object_embeddings, Config, strategy, epoch)

    trainData = CurriculumPairDataset(
        Config=Config, dataset_name=dataset,
        strategy=strategy, mined_pairs=mined_pairs)

    # loading data into dataloader
    tdataloader = DataLoader(trainData, shuffle=True,
                             num_workers=16, batch_size=Config.BS)

    trainloop = tqdm(tdataloader, leave=False)
    trcv_model.train()

    for data in trainloop:
        # for each training dataset
        # it will return 3 things
        # ximage, pimage, label_A, label_N
        # so, for each data in training loop, we can break data into 3 outputs
        # embedding of A, N and label category
        I_A, I_N, label_A, label_N = data

        # loading embeddings and label category into device
        I_A = I_A.to(device)
        I_N = I_N.to(device)
        label_A = label_A.to(device)
        label_N = label_N.to(device)

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
        L_CAT = large_margin_category_loss(
            cat_criterion, SV_A, SV_N, label_A, label_N)

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
        trainloop.set_postfix(strategy=strategy, L_cat=(
            L_CAT).item(), L_piobj=L_PiOBJ.item())

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
