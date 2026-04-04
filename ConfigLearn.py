
# HYPERPARAMETERS USED FOR TRAINING
# AUGMENTATION AND OTHER STUFFS ASSOCIATED WITH TRAINING

# PiRO used 3 datasets
# ObjectPI, ModelNet-40, FG3D
# objectPI also known as OOWL

import torchvision.transforms as transforms  # for image pre-processing
import random
import numpy as np


class HyperParams():
    def __init__(self, dataset, expname="default", seed_no=1):
        self.dataset = dataset  # ObjectPI (OOWL), ModelNet-40, FG3D
        self.case = 1
        # embedding dimension of objects, categories (for multi-view)
        self.embDim = 2048

        # Margins for object-embedding space
        self.alpha = 0.25  # intra-class margin
        self.beta = 1.0  # inter-class margin

        # margins for category embedding space
        self.theta = 0.25  # category margin
        # controlling seperation between categories (large-margin softmax loss)
        self.gamma = 4.0

        # for each particular datasets
        # choosing batch size and randomly sampled images
        if dataset == 'MNet40':
            self.batchSize = 3
            self.n_randsamp_class = 12
            self.lamda = 1.0  # weight balancing factor between clusting loss and seperation loss
        elif dataset == 'OOWL':
            self.batchSize = 4
            self.n_randsamp_class = 8
            self.lamda = 1.0
        elif dataset == 'FG3D':
            self.batchSize = 9
            self.n_randsamp_class = 12
            self.lamda = 2.0
        # clustring loss & seperation loss occurs in object embedding space
        # so we can take
        # self.lamda as "controls the weight for inter-class seperation"
        # different images can be of same object
        # so lamda is for inter-class seperation of controling weight

        else:
            print("Dataset not specified")

        self.seed_inp = seed_no  # seed input for an experiemnt or experiment number
        self.nHeads = 1  # number of heads for transformer encoder
        # in dual obj cat embedding 1+1 = 2 single-head self-attention layer
        self.nLayers = 1  # layers for transformer encoder
        self.dropout = 0.25  # dropout for self-attention layer
        self.expname = expname  # experiment name
        self.task = 'JNT'  # joint training # the model is trained jointly on multiple-objectives/embedding-spaces at the same time
        # also can configured for joint loss (category + object. more detail in training loop)
        self.ecc_ratio = 1.5  # early convergence ratio

# configurations = all the setting required for running the experiment
# for this it is like
# how to train, what data to use, where to store results etc


# configurations for ObjectPI (OOWL) dataset
class ConfigOOWL():
    def __init__(self, case, edim, bs, a, n_s, seed_no):
        super(ConfigOOWL, self).__init__()
        random.seed(seed_no)
        self.root_path = "data/ObjectPI/"
        self.save_path = "results/ObjectPI/"
        self.case = case
        self.data_dir = self.root_path
        self.gallery_dir = self.data_dir+"train/"
        self.probe_dir = self.data_dir+"test/"
        self.save_model_path = self.save_path+'models/CAT_' + \
            str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.best_model_path = self.save_path+'models/CAT_Best' + \
            str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.save_result_path = self.save_path+str(case)+'_OOWLbenchmark.csv'
        self.save_plot_dist_path = self.save_path+str(case)+'_OOWLdistance.png'
        self.save_plot_learn_path = self.save_path+str(case)+'_OOWLlearn.png'

        self.BS = bs
        self.Nepochs = 25  # number of epochs
        self.Ncls = 25  # no of class (category)
        self.Niter = 1  # iterations
        self.Ntrain = 382  # training samples
        self.Ntest = 98  # testing samples

        # this is a part of metric learning
        # object to class mapping for training and testing
        self.o2ctrain = np.load(self.gallery_dir+'train_o2c.npy').astype('int')
        self.o2ctest = np.load(self.probe_dir+'test_o2c.npy').astype('int')
        clist = []  # class to obj list

        for c in range(25):
            clist.append([])
        for i, x in enumerate(self.o2ctrain):
            temp = clist[x]
            temp.append(i)
            clist[x] = temp
        self.class_list = clist  # class to obj mapping
        print(clist)

        self.LR = 0.00001  # learning rate
        self.alpha = a  # alpha = intra class margin
        self.inpChannel = 3  # color input channel
        self.imgDim = 224  # img size 224 x 224
        self.embedDim = edim  # embedding dimension
        self.vData = False  # most likely visual data flag/ view-based data flag
        # when flag = False then
        # treat input as individual images or structured multi-view batches
        # when flag = True
        # maybe for visualization
        # when self.vData = False that applies
        # don't treat views seperately for visualization - just use them normally in training
        # practically vData = False -> training mode
        # vData = True -> visualization mode
        self.Ncomp = 10  # most likely number of comparision samples
        # maybe used in metric learning & retrieval evalution
        # for each query -> compare with 10 candidates

        # for storing view-points
        # view-points are the multiple views of single object
        self.gal_vp = []  # gallery viewpoints
        self.probe_vp = []  # probe viewpoints
        # for retrieval systems
        # gallery = reference database (train set)
        # probe = query samples (test set)

        # augmentation of images before training
        self.train_dataAug = transforms.Compose([
            transforms.Resize((self.imgDim, self.imgDim)),
            transforms.RandomHorizontalFlip(),

            # updated RandomAffine
            transforms.RandomAffine(degrees=5, translate=None, scale=(
                0.9, 1.1), shear=[-1, 1, -1, 1], interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # maximum allowed number of gallery views per object (batch limit)
        N_G_B = 8
        self.metaCategories = [range(x-15, x+1) for x in range(16, 383, 16)]
        # for each group self.metaCategories contains 16 consecutive indices
        # each group represents one object (with mutiple views)
        # metaCategories = a group of images belonging to the SAME object across multiple views

        # view-points
        # different view-points is for multi-view learning
        # 8 viewpoints per object
        # this is for retrieval and recognition
        self.gal_vp = [1, 2, 3, 4, 5, 6, 7, 8]
        self.probe_vp = [1, 2, 3, 4, 5, 6, 7, 8]
        self.N_G = min(len(self.gal_vp), N_G_B)  # no of gallery views
        print(self.N_G)

        # configuration supports
        # obj image (multi-view) -> grouped by class -> sampled for training -> model learns embeddings -> test: match probe -> gallery


# similarly
# for ModelNet-40 dataset
class ConfigMNet40():
    def __init__(self, case, edim, bs, a, n_s, seed_no):
        super(ConfigMNet40, self).__init__()
        random.seed(seed_no)
        self.root_path = "data/ModelNet40/"
        self.save_path = "results/ModelNet40/"
        self.case = case
        self.data_dir = self.root_path
        self.gallery_dir = self.data_dir+"train/"
        self.probe_dir = self.data_dir+"test/"
        self.save_model_path = self.save_path+'models/CAT_' + \
            str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.best_model_path = self.save_path+'models/CAT_Best' + \
            str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.save_result_path = self.save_path+str(case)+'_MNet40benchmark.csv'
        self.save_plot_dist_path = self.save_path + \
            str(case)+'_MNet40distance.png'
        self.save_plot_learn_path = self.save_path+str(case)+'_MNet40learn.png'

        self.BS = bs
        self.Nepochs = 25  # 50
        self.Ncls = 40
        self.Niter = 1
        self.Ntrain = 3183
        self.Ntest = 800
        self.o2ctrain = np.load(self.gallery_dir+'train_o2c.npy')

        self.o2ctest = np.load(self.probe_dir+'test_o2c.npy')
        print("O2CTrain", self.o2ctrain)
        print("O2CTest", self.o2ctest)
        clist = []
        for c in range(self.Ncls):
            clist.append([])
        for i, x in enumerate(self.o2ctrain):
            temp = clist[x]
            temp.append(i)
            clist[x] = temp
        self.class_list = clist  # object class list
        print(clist)
        self.LR = 0.00001
        self.alpha = a
        self.inpChannel = 3
        self.imgDim = 224
        self.embedDim = edim
        self.vData = False
        self.Ncomp = 10
        self.gal_vp = []
        self.probe_vp = []

        # augmentation on image
        self.train_dataAug = transforms.Compose([
            transforms.Resize((self.imgDim, self.imgDim)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=5, translate=None, scale=(
                0.9, 1.1), shear=[-1, 1, -1, 1], interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        N_G_B = 12
        self.gal_vp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.probe_vp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.N_G = min(len(self.gal_vp), N_G_B)
        print(self.N_G)


# for FG3D dataset
class ConfigFG3D():
    def __init__(self, case, edim, bs, a, n_s, seed_no):
        super(ConfigFG3D, self).__init__()
        random.seed(seed_no)
        self.root_path = "data/FG3D/"
        self.save_path = "results/FG3D/"
        self.case = case
        self.data_dir = self.root_path
        self.gallery_dir = self.data_dir+"train/"
        self.probe_dir = self.data_dir+"test/"
        self.save_model_path = self.save_path+'models/CAT_' + \
            str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.best_model_path = self.save_path+'models/CAT_Best' + \
            str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.save_result_path = self.save_path+str(case)+'_FG3Dbenchmark.csv'
        self.save_plot_dist_path = self.save_path+str(case)+'_FG3Ddistance.png'
        self.save_plot_learn_path = self.save_path+str(case)+'_FG3Dlearn.png'

        self.BS = bs
        self.Nepochs = 25
        self.Ncls = 66
        self.Niter = 1
        self.Ntrain = 21575
        self.Ntest = 3977
        self.o2ctrain = np.load(self.gallery_dir+'train_o2c.npy')

        self.o2ctest = np.load(self.probe_dir+'test_o2c.npy')
        print("O2CTrain", self.o2ctrain)
        print("O2CTest", self.o2ctest)
        clist = []
        for c in range(self.Ncls):
            clist.append([])
        for i, x in enumerate(self.o2ctrain):
            temp = clist[x]
            temp.append(i)
            clist[x] = temp
        self.class_list = clist
        print(clist)
        self.LR = 0.00005
        self.alpha = a
        self.inpChannel = 3
        self.imgDim = 224
        self.embedDim = edim
        self.vData = False
        self.Ncomp = 3
        self.gal_vp = []
        self.probe_vp = []
        self.train_dataAug = transforms.Compose([
            transforms.Resize((self.imgDim, self.imgDim)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=5, translate=None, scale=(
                0.9, 1.1), shear=[-1, 1, -1, 1], interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        N_G_B = 12
        self.gal_vp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.probe_vp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.N_G = min(len(self.gal_vp), N_G_B)
        print(self.N_G)
