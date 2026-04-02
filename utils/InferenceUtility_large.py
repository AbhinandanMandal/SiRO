
"""
Functions for Pose-invariant Classification and Retrieval (PiCR)
category and object level task

"""

from tqdm import tqdm
import time
import faiss
import numpy as np
from utils.rank_metrics import calculate_mAP_large
import torch
from functools import reduce
import operator
from torch.utils.data import DataLoader
from utils.DataUtility_PiRO import loadDataset


"""
===============================================================================
    Class for Fast Similarity Search for classification and retrieval 
    based on k-Nearest Neighbors. This is based on Faiss library 
    Reference: https://github.com/facebookresearch/faiss
===============================================================================
"""


"""
This is custom k-nearestneighbors classifier built using faiss 
for fast similarity search

Basic Idea:
This class -> stores dataset (X,y)
           -> use faiss to quickly find nearest neighbors
           -> predicts labels using majority voting

"""


class FaissKNeighbors:
    def __init__(self, k=5):
        # indexing structure (faiss structure to store vectors)
        self.index = None
        self.y = None  # labels of training data
        self.k = k  # k value for nearest neighbors

    """ Building Faiss index """
    # X,y is the data for training model

    def fit(self, X, y):
        d = X.shape[1]  # no of features per data points (dimension)
        # creating index using L2 (euclidean distance)
        self.index = faiss.IndexFlatL2(d)
        self.index.add(X.astype(np.float32))  # storing all training vectors
        self.y = y  # storing labels

    """
    Finding k-nearest neighbors

    for each input point -> it finds k closest point and return 
    the indices of the nearest neighbors

    for example for any given points -> output is [[2, 5, 1],[0, 3, 4]]
    """

    def neighbors(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        return indices  # indices of the nearest neighbors

    """ Predict label based on majority voting """

    def predict(self, X):
        # indices of the nearest neighbors
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices].astype(np.int32)  # voting
        predictions = np.array([np.argmax(np.bincount(x))
                               for x in votes])  # finding predictions
        return predictions  # returing predictions (array of predicted labels)

    """ Used for SV (single-view) object recognition """

    def predict_SV(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k+1)
        votes = self.y[indices].astype(np.int32)  # voting
        predictions = np.array([np.argmax(np.bincount(x))
                               for x in votes[:, 1:]])
        return predictions


"""
===============================================================================
                             Inference framework
===============================================================================
"""


# prediction of object labels using FaissKNeighbors class


def predict_object_label(Query, Database, labels):
    CLS_KNN = FaissKNeighbors(1)  # predict using 1-nearest neighbors (NN)
    CLS_KNN.fit(np.array(Database), np.array(labels))  # X,y set
    # prediction labels for given query
    obj_predict = CLS_KNN.predict(np.array(Query))
    return obj_predict

# Multi-view object recognition and retrieval accuracy


def predict_MVOR(refemb, testemb, testlabel, Config):
    """Computing multi-view object-level recognition accuracy"""
    acc = 0
    op = predict_object_label(testemb, refemb, testlabel)
    for x in tqdm(range(testlabel)):
        if op[x] == testlabel[x]:
            acc += 1  # incrementing accuracy if output and actual is same
    print("MV Object Recognition Accuracy: ", acc/len(testlabel)*100)

    """Compute mutli-view object retrieval mAP"""
    # mAP is mean average precision
    # it measures how good model in retrieving correct items
    # ranking all objects in test database for a multi-view query of a particular object
    # NOTE: reference views and test views for each object are disjoint
    OR_KNN = FaissKNeighbors(Config.Ntest)
    OR_KNN.fit(np.array(refemb), np.array(testlabel))  # fitting X,y
    objranks = OR_KNN.neighbors(np.array(testemb))  # ranking of objects
    obj_mAP = calculate_mAP_large(objranks, torch.tensor(testlabel))
    print("MV Object Retrieval mAP: ", obj_mAP*100)
    return acc/len(testlabel)*100, obj_mAP*100


"""
For recognition we measure accuracy
For retrieval we meaure mAP

object recognition (accuracy)
object retrieval (mAP)
category recognition (accuracy)
category retrieval (mAP)

This function evaluates learned embeddings usign FaissKNeighbors approach
it answers the following
Can i recognize the object?
Can i retrieve the same object?
Can i recognize the category?
Can i retrieve the same category?
"""
# Nearest Neighbor Classifier Pose Invariant Embedding for Single View

# For the case of single-view object recognition, retrieval and category recognition, retrieval


def NNC_PIE_CLS(refC_emb, testO_emb, testC_emb, Config):
    """
    refC_emb: category embedding (reference/train)
    testO_emb: object embedding (test)
    testC_emb: category embedding (test)
    Config: config (metadata)
    """
    cacc = 0  # class recognition accuracy
    oacc = 0  # object recognition accuracy
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest

    for i, x in enumerate(refC_emb):
        if x.ndim == 1:
            refC_emb[i] = x.reshape(1, x.shape[0])

    # prep of training data
    # combine all the reference embedding into one matrix
    XTrain = np.concatenate(refC_emb, axis=0)

    # object labels, each embedding gets object ID
    oTrain = [[i]*refC_emb[i].shape[0] for i in range(Config.Ntrain)]
    oTrain = torch.tensor(reduce(operator.concat, oTrain))

    # category labels
    cTrain = [[o2cTrain[i]]*refC_emb[i].shape[0] for i in range(Config.Ntrain)]
    cTrain = torch.tensor(reduce(operator.concat, cTrain))

    # prep of testing data
    XCTest = np.concatenate(testC_emb, axis=0)  # category specific
    XOTest = np.concatenate(testO_emb, axis=0)  # object specific
    oTest = [[i]*testO_emb[i].shape[0] for i in range(Config.Ntest)]
    oTest = torch.tensor(reduce(operator.concat, oTest))
    cTest = [[o2cTest[i]]*testC_emb[i].shape[0] for i in range(Config.Ntest)]
    cTest = torch.tensor(reduce(operator.concat, cTest))
    # reduce(operator.concat) = helps to flatten a list of list into single list

    """Computing single-view object-level retrieval mAP
    For a given single-view query of an object retrieve other views of the same object
    """
    OR_KNN = FaissKNeighbors(
        Config.Ntest*Config.N_G)  # single-view object-level faiss search
    # Config.Ntest = no of test samples
    # Config.N_G = no of gallery views (no of views per object)
    OR_KNN.fit(np.array(XOTest), np.array(oTest))
    # ranks of the retrieve objects
    objranks = OR_KNN.neighbors(np.array(XOTest))
    obj_mAP = calculate_mAP_large(objranks, oTest, "other_view")
    print("SV Object Retrieval mAP: ", obj_mAP*100)
    del OR_KNN
    del objranks

    # In retrievel FaissKNeighbors took (Config.Ntest * Config.N_G)
    # but in recogition took (1) cause,
    # retrievel needs many nearest neighbors to rank objects
    # recognition (classificiation) needs only single nearest neighbors to decide the label

    """Compute single-view object recognition accuracy"""
    OC_KNN = FaissKNeighbors(1)  # using 1 nearest neighbors (NN)
    OC_KNN.fit(np.array(XOTest), np.array(oTest))
    objpredict = OC_KNN.predict_SV(np.array(XOTest))
    for x, gt in enumerate(oTest):
        if objpredict[x] == gt:
            oacc += 1
    print("SV Object Recognition Accuracy: ", oacc/len(oTest)*100)
    del OC_KNN
    del objpredict

    """Compute single-view category recognition accuracy"""
    CLS_KNN = FaissKNeighbors(10)
    CLS_KNN.fit(np.array(XTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))
    for t in range(0, len(cTest)):
        if cls_predict[t] == cTest[t]:
            cacc += 1
    print("SV Category Recognition Accuracy: ", cacc/len(cTest)*100)
    del CLS_KNN
    del cls_predict
    del XTrain
    del cTrain
    del oTrain

    """Compute single-view category retrieval mAP"""
    CR_KNN = FaissKNeighbors(Config.Ntest*Config.N_G)
    CR_KNN.fit(np.array(XCTest), np.array(cTest))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP_large(crranks, cTest)
    print("SV Category Retrieval mAP: ", cr_mAP*100)
    del CR_KNN
    del crranks

    return obj_mAP*100, cr_mAP*100, cacc/len(cTest)*100, oacc/len(oTest)*100


# For the case of multi-view category recognition, retrieval
def NNC_PIE_CLS_MV(refC_emb, testC_emb, Config):
    cacc = 0  # category recognition accuracy
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest

    # Training data
    XTrain = np.concatenate(refC_emb, axis=0)
    # category labels of training data
    cTrain = torch.tensor([o2cTrain[i] for i in range(Config.Ntrain)])

    # Testing data
    XCTest = np.concatenate(testC_emb, axis=0)
    # category labels of testing data
    cTest = torch.tensor([o2cTest[i] for i in range(Config.Ntest)])

    """Computing multi-view category retrieval mAP"""
    CR_KNN = FaissKNeighbors(
        Config.Ntest)  # for retrieval we need large amount of data
    CR_KNN.fit(np.array(XCTest), np.array(cTest))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP_large(crranks, cTest)
    print("MV Category Retrieval mAP: ", cr_mAP*100)

    del CR_KNN
    del crranks

    """Computing multi-view category recognition accuracy"""
    CLS_KNN = FaissKNeighbors(10)
    CLS_KNN.fit(np.array(XTrain), np.array(cTrain))
    # we'll predict from the testing points
    cls_predict = CLS_KNN.predict(np.array(XCTest))

    for t in range(0, len(cTest)):
        if cls_predict[t] == cTest[t]:
            cacc += 1
    print("MV Category Recognition Accuracy: ", cacc/len(cTest)*100)
    del CLS_KNN
    del cls_predict

    return cr_mAP*100, cacc/len(cTest)*100


"""
===============================================================================
Evaluate PI (Pose-Invariant) category and object-level classification and retrieval performance 
                            for Dual embedding space
===============================================================================
"""


def evaluate_performance_dual(dataset, Config, trcv_model, nview):
    ref_OE = {}  # single-view reference (training) object embedding
    ref_CE = {}  # single-view class embedding
    ref_mv_CE = {}  # multi-view reference class embedding
    ref_mv_OE = {}  # multi-view object embedding
    test_OE = {}  # test object embedding
    test_CE = {}  # test class embedding
    label_obj = {}
    test_mv_OE = {}
    label_mv_OE = {}

    """
    Loading dual embedding space model and 
    extract the gallery (train) embeddings and probe (test) embeddings
    """
    trainData = loadDataset(Config, 'train', dataset)
    # can change according to GPU power
    trainLoader = DataLoader(trainData, shuffle=False,
                             num_workers=16, batch_size=1)

    trcv_model.eval()  # model testing mode
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)):
            # part of calculate_stats under loadDataset
            rOE, rCE, mvrOE, mvrCE, _, _ = trcv_model(ref_data)
            # reference obj. emb.
            ref_OE[i] = rOE.squeeze().detach().cpu().numpy()
            ref_CE[i] = rCE.squeeze().detach().cpu().numpy()  # ref. class emb.
            ref_mv_CE[i] = mvrCE.detach().cpu().numpy()  # ref. mv. class emb.

    del trainData
    del trainLoader

    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData, shuffle=False,
                            num_workers=16, batch_size=1)

    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)):
            if nview == 1:
                """If single-view query"""
                tOE, tCE, _, _, _, _ = trcv_model(test_data)
                test_OE[j] = tOE.squeeze().detach().cpu().numpy()
                test_CE[j] = tCE.squeeze().detach().cpu().numpy()
                label_obj[j] = obj_labels

            elif nview > 1:
                """ if multi-view query """
                tOE, tCE, mvtOE, mvtCE, _, _ = trcv_model(test_data)
                test_CE[j] = mvtCE.detach().cpu().numpy()
                """ 
                    evaluating if the model can extract multi-view embeddings where the 
                    reference images and the test query images of an unseen object are from disparate viewpoints                    
                    split all available views of test object into two disjoint sets: 
                    gallery: set comprising of reference views 
                    probe: set comprising of test views  
                """
                p = int(Config.N_G/2)
                _, _, refmvobj, _, _, _ = trcv_model(test_data[:, :p])
                _, _, testmvobj, _, _, _ = trcv_model(test_data[:, p:])
                ref_mv_OE[j] = refmvobj.squeeze().detach().cpu().numpy()
                test_mv_OE[j] = testmvobj.squeeze().detach().cpu().numpy()
                label_mv_OE[j] = j

    if nview == 1:
        """Computing performance from single-view query"""
        start2 = time.time()
        ormap, crmap, clsacc, svor = NNC_PIE_CLS(list(ref_CE.values()), list(
            test_OE.values()), list(test_CE.values()), Config)
        # ormap: object retrieval mAP
        # crmap: class retrieval mAP
        # clsacc: singel-view class recognition accuracy
        # svor: single-view object recognition accuracy

        end2 = time.time()
        print("Time (Inference): ", (end2 - start2))
        return ormap, crmap, clsacc, svor

    if nview > 1:
        """Computing Performance from multi-view query"""
        start2 = time.time()
        crmap, clsacc = NNC_PIE_CLS_MV(
            # multi-view category recognition & retrieval
            list(ref_mv_CE.values()), list(test_CE.values()), Config)
        mvor, mvoret = predict_MVOR(list(ref_mv_OE.values()), list(
            # multi-view object recognition and retrieval
            test_mv_OE.values()), list(label_mv_OE.values()), Config)
        end2 = time.time()
        print("Time (Inference): ", (end2 - start2))
        return crmap, clsacc, mvor, mvoret


"""
===============================================================================
Evaluate PI (Pose-Invariant) category and object-level classification and retrieval performance 
                        for Single embedding space
===============================================================================
"""


def evaluate_performance_single(dataset, Config, trcv_model, nview):
    ref_OE = {}
    ref_CE = {}
    ref_mv_CE = {}
    ref_mv_OE = {}
    test_OE = {}
    test_CE = {}
    label_obj = {}
    test_mv_OE = {}
    label_mv_OE = {}
    """
    Load the Single embedding space model and 
    extract the gallery (train) embeddings and probe (test) embeddings
    """
    trainData = loadDataset(Config, 'train', dataset)
    trainLoader = DataLoader(trainData, shuffle=False,
                             num_workers=16, batch_size=1)
    trcv_model.eval()
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)):
            rOE, mvrOE, _ = trcv_model(ref_data)
            ref_OE[i] = rOE.squeeze().detach().cpu().numpy()
            ref_CE[i] = rOE.squeeze().detach().cpu().numpy()
            ref_mv_CE[i] = mvrOE.detach().cpu().numpy()
    del trainData
    del trainLoader
    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData, shuffle=False,
                            num_workers=16, batch_size=1)
    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)):
            if nview == 1:
                """ if single-view query """
                tOE, _, _ = trcv_model(test_data)
                test_OE[j] = tOE.squeeze().detach().cpu().numpy()
                test_CE[j] = tOE.squeeze().detach().cpu().numpy()
                label_obj[j] = obj_labels

            elif nview > 1:
                """ if multi-view query """
                tOE, mvtOE, _ = trcv_model(test_data)
                test_CE[j] = mvtOE.detach().cpu().numpy()
                """ 
                    evaluating if the model can extract multi-view embeddings where the 
                    reference images and the test query images of an unseen object are from disparate viewpoints                    
                    split all available views of test object into two disjoint sets: 
                    gallery: set comprising of reference views 
                    probe: set comprising of test views 
                """
                p = int(Config.N_G/2)
                _, refmvobj, _ = trcv_model(test_data[:, :p])
                _, testmvobj, _ = trcv_model(test_data[:, p:])
                ref_mv_OE[j] = refmvobj.squeeze().detach().cpu().numpy()
                test_mv_OE[j] = testmvobj.squeeze().detach().cpu().numpy()
                label_mv_OE[j] = j
    if nview == 1:
        """ Compute performance from single-view query """
        start2 = time.time()
        ormap, crmap, clsacc, svor = NNC_PIE_CLS(list(ref_CE.values()), list(
            test_OE.values()), list(test_CE.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        return ormap, crmap, clsacc, svor
    if nview > 1:
        """ Compute performance from multi-view query """
        start2 = time.time()
        crmap, clsacc = NNC_PIE_CLS_MV(
            list(ref_mv_CE.values()), list(test_CE.values()), Config)
        mvor, mvoret = predict_MVOR(list(ref_mv_OE.values()), list(
            test_mv_OE.values()), list(label_mv_OE.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        return crmap, clsacc, mvor, mvoret

