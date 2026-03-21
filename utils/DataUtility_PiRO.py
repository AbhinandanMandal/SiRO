
# custom dataloader for different datasets and other data related helper functions
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import PIL.ImageOps  # for operations on image
import random
import torch
import numpy as np
import os
from tqdm import tqdm


# CUSTOM TRAIN DATASET FOR MENTIONED 3 DATASETS

# custom dataset for ObjectPI (OOWL) dataset


class OOWLTrainDataset(Dataset):
    def __init__(self, Config, simClass_list, name=""):
        # Config we'll get from ConfigOOWL
        self.transform = Config.train_dataAug
        self.should_invert = False
        self.gal_vp = Config.gap_vp
        self.N_vp = len(Config.gal_vp)  # length of gallery view-points
        self.dpath = Config.gallery_dir  # datapath
        self.N_class = Config.Ntrain
        self.N_comp = Config.Ncomp  # comparision samples with a particular sample img
        self.obj2cls = Config.o2ctrain  # object to class mapping
        self.cls2obj = Config.clas_list  # class to object mapping
        self.dataset = name  # dataset name
        self.N_G = Config.N_G
        print("Sample from same category !!!")

    def applyTransform(self, img_path):  # applying img transformation
        img = Image.open(img_path)
        if self.should_invert:
            img = PIL.ImageOps.invert(img)  # operations on image
        if self.transform is not None:
            img = self.transform(img)
        return img  # transformed image

    def __getitem__(self, index):
        # this randomly samples one object (objx) and then randomly samples another object from the same category (objp)
        objx = int(index/(self.N_comp))  # getting an obj x
        cls = self.obj2cls[objx]  # obj x class
        # all object belong to that class of obj x class
        allobjcls = self.cls2obj[cls].copy()
        allobjcls.remove(objx)  # removing that objx from all obj class
        # taking a random obj (objp) from that class
        objp = random.choice(allobjcls)

        ximage = list()  # for obj x
        pimage = list()  # for obj p
        label = list()  # labels for objects
        # sampled view points take randomly gallery vp and N_G (no of gallery views)
        sampled_vp = random.sample(self.gal_vp, self.N_G)
        for i, vp in enumerate(sampled_vp):
            ximage.append(self.applyTransform(
                self.dpath+str(objx+1)+"/"+str(vp)+".jpg"))
            pimage.append(self.applyTransform(
                self.dpath+str(objp+1)+"/"+str(vp)+".jpg"))
            label.append(torch.from_numpy(np.array(cls)))  # for storing labels
        return torch.stack(ximage), torch.stack(pimage), torch.stack(label)

    def __len__(self):
        return self.N_class*self.N_comp  # total length of class


# similarly
# custom dataset for ModelNet40
class MNet40TrainDataset(Dataset):

    def __init__(self, Config, simClass_list=[], name=""):
        self.transform = Config.train_dataAug
        self.should_invert = False
        self.gal_vp = Config.gal_vp
        self.N_vp = len(Config.gal_vp)
        self.dpath = Config.gallery_dir
        self.N_class = Config.Ntrain
        self.N_comp = Config.Ncomp
        self.obj2cls = Config.o2ctrain  # object to class mapping
        self.cls2obj = Config.class_list  # class to object mapping
        self.dataset = name
        self.N_G = Config.N_G
        print("Sample from same category !!!")

    def applyTransform(self, img_path):
        img = Image.open(img_path)
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        objx = int(index/(self.N_comp))
        cls = self.obj2cls[objx]
        allobjcls = self.cls2obj[cls].copy()
        allobjcls.remove(objx)
        objp = random.choice(allobjcls)
        ximage = list()
        pimage = list()
        label = list()
        sampled_vp = random.sample(self.gal_vp, self.N_G)
        for i, vp in enumerate(sampled_vp):
            ximage.append(self.applyTransform(
                self.dpath+str(objx+1)+"/"+str(vp).zfill(3)+".jpg"))
            pimage.append(self.applyTransform(
                self.dpath+str(objp+1)+"/"+str(vp).zfill(3)+".jpg"))
            label.append(torch.from_numpy(np.array(cls)))
        return torch.stack(ximage), torch.stack(pimage), torch.stack(label)

    def __len__(self):
        return self.N_class*self.N_comp


# similarly for
# custom dataset for FG3D (fine-grained 3D) dataset
class FG3DTrainDataset(Dataset):

    def __init__(self, Config, simClass_list=[], name=""):
        self.transform = Config.train_dataAug
        self.should_invert = False
        self.gal_vp = Config.gal_vp
        self.N_vp = len(Config.gal_vp)
        self.dpath = Config.gallery_dir  # gallery for training, probe for testing
        self.N_class = Config.Ntrain
        self.N_comp = Config.Ncomp
        self.obj2cls = Config.o2ctrain
        self.cls2obj = Config.class_list
        self.dataset = name
        self.N_G = Config.N_G
        print("Sample from same category !!!")

    def applyTransform(self, img_path):
        img = Image.open(img_path)
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        objx = int(index/(self.N_comp))
        cls = self.obj2cls[objx]
        allobjcls = self.cls2obj[cls].copy()
        allobjcls.remove(objx)
        objp = random.choice(allobjcls)
        ximage = list()
        pimage = list()
        label = list()
        sampled_vp = random.sample(self.gal_vp, self.N_G)
        for i, vp in enumerate(sampled_vp):
            ximage.append(self.applyTransform(
                self.dpath+str(objx+1)+"/"+str(vp).zfill(3)+".png"))
            pimage.append(self.applyTransform(
                self.dpath+str(objp+1)+"/"+str(vp).zfill(3)+".png"))
            label.append(torch.from_numpy(np.array(cls)))
        return torch.stack(ximage), torch.stack(pimage), torch.stack(label)

    def __len__(self):
        return self.N_class*self.N_comp


# CUSTOM DATALOADER FOR TESTING, VALIDATION AND EVALUATION
# custom dataset for loading multi-view images of each object
# for evaluating model performance, also we'll use this load dataset class

# load dataset is always for multi-view images
class loadDataset(Dataset):
    def __init__(self, Config, split, dataset_name):
        self.transform = transforms.Compose([
            transforms.Resize((Config.imgDim, Config.imgDim)),
            transforms.ToTensor(),  # img tensor conversion
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.split = split  # splitting of dataset (for training, testing etc)
        if self.split == "train" or self.split == "val":
            self.N_class = Config.Ntrain
            self.datadir = Config.gallery_dir
            self.obj2cls = Config.o2ctrain
        elif self.split == "test":
            self.N_class = Config.Ntest
            self.datadir = Config.probe_dir
            self.obj2cls = Config.o2ctest
        self.dataset = dataset_name  # datset name
        self.N_G = Config.N_G
        self.Config = Config

    # applying transformation on dataset images
    def applyTransform(self, img_path):
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    # getting items
    def __getitem__(self, index):
        obj_ind = index  # object index
        cls_ind = self.obj2cls[obj_ind]  # respective class index
        images = list()  # list of images
        obj_labels = list()  # object labels
        cls_labels = list()  # class labels

        # view points for train, test, val
        if self.split == "train":
            vp = self.Config.gal_vp
        elif self.split == "val":
            vp = self.Config.gal_vp
        elif self.split == "test":
            vp = self.Config.probe_vp

        for j in vp:
            if self.dataset == "OOWL":
                data_path = self.datadir+str(index+1)+"/"+str(j)+".jpg"
            elif self.dataset == "MNet40":
                data_path = self.datadir + \
                    str(index+1)+"/"+str(j).zfill(3)+".jpg"
            elif self.dataset == "FG3D":
                data_path = self.datadir + \
                    str(index+1)+"/"+str(j).zfill(3)+".png"
            else:
                print("Dataset not recognized.")

            # applying image transformation on the entire images
            images.append(self.applyTransform(data_path))
            # object labels (tetrieving from precomputed mapping)
            obj_labels.append(torch.from_numpy(np.array(obj_ind)))
            cls_labels.append(torch.from_numpy(
                np.array(cls_ind)))  # class labels (same)

        return torch.stack(images), torch.stack(obj_labels), torch.stack(cls_labels)

    def __len__(self):
        return self.N_class


# for loading class data
# this is alternate and easier format of __getitem__
# this loads all the view-points (mutli-view) of object i
def load_class_data(i, dataset, datadir, flag, Config):
    temp = list()
    if flag == 0:
        vp = Config.gal_vp  # train view points
    elif flag == 1:
        vp = Config.probe_vp  # testing views
    elif flag == 2:
        vp = Config.gal_vp  # validation views
    else:
        print("Wrong Falg")

    for j in vp:
        if dataset == "OOWL":
            data_path = datadir+str(i+1)+"/"+str(j)+".jpg"
        elif dataset == "MNet40":
            data_path = datadir+str(i+1)+"/"+str(j).zfill(3)+".jpg"
        elif dataset == "FG3D":
            data_path = datadir+str(i+1)+"/"+str(j).zfill(3)+".png"
        else:
            print("Dataset not recognized.")
        if os.path.isfile(data_path):
            img = Image.open(data_path)
            timg = transforms.Compose([
                transforms.Resize(Config.imgDim),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])
            t = timg(img)
            t.resize_((1, Config.inpChannel, Config.imgDim, Config.imgDim))
            temp.append(t)
    return temp  # temporary image list


# calculating statistics from the current embedding soace such as
# intra-object and inter-object distances that is used for tracking performances
# of algorithm during training and to check early convergence

# this is only for object
# and it is using dual embedding space for the same
# it checks how close same object views and how far different object views
# imput is only for multi-view embedding

def calculate_stats(net, Config, dataset, emb_space='dual'):
    dismetric = []  # avergae distance between all the views of the same object
    interCN_catg = []  # average distance between the object i and other object of the same class
    interCN_min = []  # minimum distance between object i and other object of the same class
    maxdisintra = np.zeros(Config.Ntrain)  # maximum distance within object
    disintra = np.zeros(Config.Ntrain)  # avergae distance within object
    ref_emb = {}  # for embeddings of each object

    def distance_stats(i):  # computing stats for each object i
        # distance between i's embeddings (intra-distance)
        ed_i = torch.cdist(ref_emb[i], ref_emb[i])
        maxdisintra[i] = torch.max(ed_i).data.cpu().numpy()  # maximum distance
        disintra[i] = torch.mean(ed_i).data.cpu().numpy()

        # category of that particular object
        category = Config.o2ctrain[i]
        # objects of that category (all)
        objects = Config.class_list[category]

        dismetric.append([])
        interCN_catg.append([])
        interCN_min.append([])
        interobj_cat = []  # defining inter object category

        # to filling interCN_cat, interCN_min
        # we need to find out inter-class distances
        # take it like
        # distance between object i and all the object of the same class
        # inside class 'chair' there can be multiple types of chairs

        # we're working with embeddings so
        # cateogry class representations all are in embeddings
        for j in objects:
            if j == i:
                continue
            ed_ij = torch.dist(ref_emb[i], ref_emb[j])
            # sotring mean of the dist
            interobj_cat.append(torch.mean(ed_ij).data.cpu().numpy())

            # avg distance between all the views of the same object
            dismetric[i] = np.asarray(interobj_cat)
            interCN_catg[i] = np.mean(dismetric[i])
            interCN_min[i] = np.min(dismetric[i])

    # for validation purpose
    # this extracts embeddings for each object
    # then measure
    # how good those embeddings are using distance statistics

    # data for validation purpose
    trainData = loadDataset(Config, 'val', dataset)
    # data loader for working onto it
    trainLoader = DataLoader(trainData, batch_size=1,
                             shuffle=False, num_workers=16)
    net.eval()  # net is neural network model

    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)):
            # for the case of multi-view images
            # img sizes are [b, v, c h, w]
            # here we're working for embedding for each obj
            # so ref_data = [v,c,h,w]

            if emb_space == "dual":
                # rOE can be reference object embedding, rOE shape [v, emb_dim]
                rOE, _, _, _, _, _ = net(ref_data)
            elif emb_space == "single":
                rOE, _, _ = net(ref_data)
            rOE = rOE.squeeze()  # removing extra dimension
            if rOE.ndim == 1:
                # if only one view exists then [emb_dim] -> [1, emb_dim]
                ref_emb[i] = rOE.reshape(1, rOE.shape[0])

            else:
                ref_emb[i] = rOE  # normal case

    # now computing stats for each obj
    for i in tqdm(range(Config.Ntrain)):
        distance_stats(i)
    distInfo = [np.mean(maxdisintra), np.mean(disintra), np.mean(
        # information of all the distances
        interCN_catg), np.mean(interCN_min)]
    return distInfo

