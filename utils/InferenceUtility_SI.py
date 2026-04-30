
"""
Functions for inference on State-invariant Classification and Retrieval
category and object-level tasks for the ObjectsWithStateChange (OWSC) dataset
for State Invariance (SI)
"""

from tqdm import tqdm
import time
from utils.Faiss_KNN import FaissKNeighbors
from utils.DataUtility import loadDataset
from utils.rank_metrics import calculate_mAP
import operator
from functools import reduce
import faiss
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.__version__)


"""
===============================================================================
                             Inference framework
===============================================================================
"""

# Prediction of object label using faiss k nearest neighbors


def predict_object_label(Query, Database, labels):
    CLS_KNN = FaissKNeighbors(1)
    CLS_KNN.fit(np.array(Database), np.array(labels))
    obj_predict = CLS_KNN.predict(np.array(Query))
    return obj_predict

# Nearest Neighbor Classifier
# Single view object recognition & retrieval
# for the Object With State Change (OWSC) dataset


def NNC_OWSC_SV(refC_emb, refO_emb, testC_emb, testO_emb, Config):
    """
    refC_emb = category embedding (train)
    refO_emb = object embedding (train)
    testC_emb = category embedding (test)
    testO_emb = object embedding (test)
    Config = Metadata
    """
    cacc = 0  # class recognition accuracy
    oacc = 0  # object recognition accuracy
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest

    for i, x in enumerate(refC_emb):
        if x.ndim == 1:
            refC_emb[i] = x.reshape(1, x.shape[0])
    XCTrain = np.concatenate(refC_emb, axis=0)  # class training
    XOTrain = np.concatenate(refO_emb, axis=0)  # object training
    oTrain = [[i]*refO_emb[i].shape[0] for i in range(Config.Ntrain)]
    oTrain = torch.tensor(reduce(operator.concat, oTrain))
    cTrain = [[o2cTrain[i]]*refC_emb[i].shape[0] for i in range(Config.Ntrain)]
    cTrain = torch.tensor(reduce(operator.concat, cTrain))

    XCTest = np.concatenate(testC_emb, axis=0)
    XOTest = np.concatenate(testO_emb, axis=0)
    oTest = [[i]*testO_emb[i].shape[0] for i in range(Config.Ntest)]
    oTest = torch.tensor(reduce(operator.concat, oTest))
    cTest = [[o2cTest[i]]*testC_emb[i].shape[0] for i in range(Config.Ntest)]
    cTest = torch.tensor(reduce(operator.concat, cTest))

    """=====================================================================
                                SV Object Retrieval
       =====================================================================
    """
    OR_KNN = FaissKNeighbors(XOTrain.shape[0])
    OR_KNN.fit(np.array(XOTrain), np.array(oTrain))
    objranks = OR_KNN.neighbors(np.array(XOTest))
    obj_mAP = calculate_mAP(objranks, oTest)
    print("SV Object Retrieval mAP: ", obj_mAP)
    del OR_KNN
    del objranks
    """=====================================================================
                                SV Object Recognition
       =====================================================================
    """
    OC_KNN = FaissKNeighbors(1)
    OC_KNN.fit(np.array(XOTrain), np.array(oTrain))
    objpredict = OC_KNN.predict(np.array(XOTest))
    for x, gt in enumerate(oTest):
        if objpredict[x] == gt:
            oacc += 1
    print("SV Object Recognition: ", oacc/len(oTest))
    del OC_KNN
    del objpredict
    """=====================================================================
                                SV Category Recognition
       =====================================================================
    """
    CLS_KNN = FaissKNeighbors(10)
    CLS_KNN.fit(np.array(XCTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))

    for t in range(0, len(cTest)):
        if cls_predict[t] == cTest[t]:
            cacc += 1
    print("SV Category Recognition: ", cacc/len(cTest))

    del CLS_KNN
    del cls_predict
    """=====================================================================
                                SV Category Retrieval
       =====================================================================
    """
    CR_KNN = FaissKNeighbors(XCTrain.shape[0])
    CR_KNN.fit(np.array(XCTrain), np.array(cTrain))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP(crranks, cTest)
    print("SV Category Retrieval mAP: ", cr_mAP)

    del XCTrain
    del XOTrain
    del cTrain
    del oTrain
    del CR_KNN
    del crranks
    """
    cr_mAP = single view category retrieval map
    cacc = single view category recognition accuracy
    obj_mAP = single view object retrieval map
    oacc = single view object recognition accuracy
    """
    return cr_mAP*100, cacc/len(cTest)*100, obj_mAP*100, oacc/len(oTest)*100


# Multi view object recognition & retrieval
# for OWSC dataset
def NNC_OWSC_MV(refC_emb, refO_emb, testC_emb, testO_emb, Config):
    cacc = 0
    oacc = 0
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest
    XCTrain = np.concatenate(refC_emb, axis=0)
    cTrain = torch.tensor([o2cTrain[i] for i in range(Config.Ntrain)])
    XOTrain = np.concatenate(refO_emb, axis=0)
    oTrain = torch.tensor([i for i in range(Config.Ntrain)])

    XCTest = np.concatenate(testC_emb, axis=0)
    cTest = torch.tensor([o2cTest[i] for i in range(Config.Ntest)])
    XOTest = np.concatenate(testO_emb, axis=0)
    oTest = torch.tensor([i for i in range(Config.Ntest)])

    """=====================================================================
                                MV Category Retrieval
       =====================================================================
    """
    CR_KNN = FaissKNeighbors(Config.Ntest)
    CR_KNN.fit(np.array(XCTrain), np.array(cTrain))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP(crranks, cTest)

    print("MV Category Retrieval mAP: ", cr_mAP)

    del CR_KNN
    del crranks

    """=====================================================================
                                MV Category Recognition
       =====================================================================
    """
    CLS_KNN = FaissKNeighbors(5)
    CLS_KNN.fit(np.array(XCTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))

    for t in range(0, len(cTest)):
        if cls_predict[t] == cTest[t]:
            cacc += 1

    print("MV Category Recognition Accuracy: ", cacc/len(cTest))

    del CLS_KNN
    del cls_predict
    del XCTrain
    del XCTest
    del cTrain

    """=====================================================================
                                MV Object Retrieval
       =====================================================================
    """
    OR_KNN = FaissKNeighbors(Config.Ntest)
    OR_KNN.fit(np.array(XOTrain), np.array(oTrain))
    orranks = OR_KNN.neighbors(np.array(XOTest))
    or_mAP = calculate_mAP(orranks, oTest)

    print("MV Object Retrieval mAP: ", or_mAP)

    del OR_KNN
    del orranks

    """=====================================================================
                                MV Object Recognition
       =====================================================================
    """
    OBJ_KNN = FaissKNeighbors(1)
    OBJ_KNN.fit(np.array(XOTrain), np.array(oTrain))
    obj_predict = OBJ_KNN.predict(np.array(XOTest))

    for t in range(0, len(oTest)):
        if obj_predict[t] == oTest[t]:
            oacc += 1

    print("MV Object Recognition mAP: ", oacc/len(oTest))

    del OBJ_KNN
    del obj_predict

    return cr_mAP*100, cacc/len(cTest)*100, or_mAP*100, oacc/len(oTest)*100


"""
===============================================================================
Evaluate State-invariant category and object-level classification and retrieval
                performance for Single embedding space methods
===============================================================================
"""


"""
# Evaluation of state invariant performance for single embedding model
def evaluate_SI_performance_single(dataset, Config, trcv_model, nview):
    ref_OE = {}  # single view object embedding (test)
    ref_CE = {}  # single view category embedding (train)
    ref_mv_CE = {}  # multi view category embedding (train)
    test_mv_CE = {}  # multi view category embedding (test)
    test_OE = {}  # single view object embedding (test)
    test_CE = {}  # single view category embedding (test)
    label_cls = {}  # class labels
    label_obj = {}  # object labels
    ref_mv_OE = {}  # multi view object embedding (train)
    test_mv_OE = {}  # multi view object embedding (test)
    label_mv_OE = {}  # multi view object labels


    # Load the Single Pose-invariant embedding space model and
    # extract the gallery (train) embeddings and probe (test) embeddings

    trainData = loadDataset(Config, 'train', dataset)
    trainLoader = DataLoader(trainData, shuffle=False,
                             num_workers=8, batch_size=1)
    trcv_model.eval()
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)):
            rOE, mvrOE, _ = trcv_model(ref_data.cuda())
            ref_OE[i] = rOE.squeeze().detach().cpu().numpy()
            ref_CE[i] = rOE.squeeze().detach().cpu().numpy()
            ref_mv_CE[i] = mvrOE.detach().cpu().numpy()
            ref_mv_OE[i] = mvrOE.detach().cpu().numpy()
    del trainData
    del trainLoader
    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData, shuffle=False,
                            num_workers=8, batch_size=1)
    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)):
            tOE, mvtOE, _ = trcv_model(test_data.cuda())
            test_OE[j] = tOE.squeeze().detach().cpu().numpy()
            test_CE[j] = tOE.squeeze().detach().cpu().numpy()
            label_obj[j] = obj_labels
            test_mv_CE[j] = mvtOE.detach().cpu().numpy()
            test_mv_OE[j] = mvtOE.detach().cpu().numpy()
            label_mv_OE[j] = j

    # Compute performance from single-image query
    SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc = NNC_OWSC_SV(list(ref_CE.values()), list(
        ref_OE.values()), list(test_CE.values()), list(test_OE.values()), Config)

    # Compute performance from multi-image query
    MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc = NNC_OWSC_MV(list(ref_mv_CE.values()), list(
        ref_mv_OE.values()), list(test_mv_CE.values()), list(test_mv_OE.values()), Config)


    # SV_C_mAP : single view class retrieval accuracy
    # SV_C_acc : single view class recognition accuracy
    # SV_O_mAP : single view object retrieval accuracy
    # SV_O_acc : single view object recognition accuracy
    # # similarly for multi view

    return SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc, MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc

"""


# Updating 'evaluate_SI_performance_single for 'nview'
def evaluate_SI_performance_single(dataset, Config, trcv_model, nview):
    ref_OE = {}  # single view object embedding (test)
    ref_CE = {}  # single view category embedding (train)
    ref_mv_CE = {}  # multi view category embedding (train)
    test_mv_CE = {}  # multi view category embedding (test)
    test_OE = {}  # single view object embedding (test)
    test_CE = {}  # single view category embedding (test)
    label_cls = {}  # class labels
    label_obj = {}  # object labels
    ref_mv_OE = {}  # multi view object embedding (train)
    test_mv_OE = {}  # multi view object embedding (test)
    label_mv_OE = {}  # multi view object labels
    """
    Load the Single Pose-invariant embedding space model and
    extract the gallery (train) embeddings and probe (test) embeddings
    """
    trainData = loadDataset(Config, 'train', dataset)
    trainLoader = DataLoader(trainData, shuffle=False,
                             num_workers=8, batch_size=1)
    trcv_model.eval()
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)):
            rOE, mvrOE, _ = trcv_model(ref_data.cuda())
            ref_OE[i] = rOE.squeeze().detach().cpu().numpy()
            ref_CE[i] = rOE.squeeze().detach().cpu().numpy()
            ref_mv_CE[i] = mvrOE.detach().cpu().numpy()
            ref_mv_OE[i] = mvrOE.detach().cpu().numpy()
    del trainData
    del trainLoader
    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData, shuffle=False,
                            num_workers=8, batch_size=1)
    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)):
            """
             tOE, mvtOE, _ = trcv_model(test_data.cuda())
             test_OE[j] = tOE.squeeze().detach().cpu().numpy()
             test_CE[j] = tOE.squeeze().detach().cpu().numpy()
             label_obj[j] = obj_labels
             test_mv_CE[j] = mvtOE.detach().cpu().numpy()
             test_mv_OE[j] = mvtOE.detach().cpu().numpy()
             label_mv_OE[j] = j
            """
            if nview == 1:
                """ If single-view query """
                tOE, _, _ = trcv_model(test_data)
                test_OE[j] = tOE.squeeze().detach().cpu().numpy()
                test_CE[j] = tOE.squeeze().detach().cpu().numpy()
                label_obj[j] = obj_labels

            elif nview > 1:
                """ If multi-view query """
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
        """ Compute performance from single-image query """
        start2 = time.time()
        SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc = NNC_OWSC_SV(list(ref_CE.values()), list(
            ref_OE.values()), list(test_CE.values()), list(test_OE.values()), Config)
        end2 = time.time()
        print("Time (Inference): ", (end2 - start2))
        return SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc
    if nview > 1:

        """ Compute performance from multi-image query """
        start2 = time.time()
        MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc = NNC_OWSC_MV(list(ref_mv_CE.values()), list(
            ref_mv_OE.values()), list(test_mv_CE.values()), list(test_mv_OE.values()), Config)
        end2 = time.time()
        print("Time (Inference): ", (end2 - start2))
        return MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc
    """
    SV_C_mAP : single view class retrieval accuracy
    SV_C_acc : single view class recognition accuracy
    SV_O_mAP : single view object retrieval accuracy
    SV_O_acc : single view object recognition accuracy
    # similarly for multi view
    """
    # return SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc, MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc


"""
===============================================================================
Evaluate State-invariant category and object-level classification and retrieval
                performance for Dual embedding space methods
===============================================================================
"""

""" 
def evaluate_SI_performance_dual(dataset, Config, trcv_model, nview):
    ref_OE = {}
    ref_CE = {}
    ref_mv_CE = {}
    test_mv_CE = {}
    test_OE = {}
    test_CE = {}
    label_cls = {}
    label_obj = {}
    ref_mv_OE = {}
    test_mv_OE = {}
    label_mv_OE = {}
    
    # Load the Dual Pose-invariant embedding space model and 
    # extract the gallery (train) embeddings and probe (test) embeddings
   
    trainData = loadDataset(Config, 'train', dataset)
    trainLoader = DataLoader(trainData, shuffle=False,
                             num_workers=8, batch_size=1)
    trcv_model.eval()
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)):
            rOE, rCE, mvrOE, mvrCE, _, _ = trcv_model(ref_data)
            ref_OE[i] = rOE.squeeze().detach().cpu().numpy()
            ref_CE[i] = rCE.squeeze().detach().cpu().numpy()
            ref_mv_CE[i] = mvrCE.detach().cpu().numpy()
            ref_mv_OE[i] = mvrOE.detach().cpu().numpy()
    del trainData
    del trainLoader
    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData, shuffle=False,
                            num_workers=8, batch_size=1)
    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)):
            tOE, tCE, mvtOE, mvtCE, _, _ = trcv_model(test_data)
            test_OE[j] = tOE.squeeze().detach().cpu().numpy()
            test_CE[j] = tCE.squeeze().detach().cpu().numpy()
            label_obj[j] = obj_labels
            test_mv_CE[j] = mvtCE.detach().cpu().numpy()
            test_mv_OE[j] = mvtOE.detach().cpu().numpy()
            label_mv_OE[j] = j
    #  Compute performance from single-image query 
    SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc = NNC_OWSC_SV(list(ref_CE.values()), list(
        ref_OE.values()), list(test_CE.values()), list(test_OE.values()), Config)
    #  Compute performance from multi-image query 
    MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc = NNC_OWSC_MV(list(ref_mv_CE.values()), list(
        ref_mv_OE.values()), list(test_mv_CE.values()), list(test_mv_OE.values()), Config)

    # Similarly for dual encoder model
    return SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc, MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc

"""

# updating evaluate_SI_performance_dual for 'nview'


def evaluate_SI_performance_dual(dataset, Config, trcv_model, nview):
    ref_OE = {}
    ref_CE = {}
    ref_mv_CE = {}
    test_mv_CE = {}
    test_OE = {}
    test_CE = {}
    label_cls = {}
    label_obj = {}
    ref_mv_OE = {}
    test_mv_OE = {}
    label_mv_OE = {}

    """ 
    Load the Dual Pose-invariant embedding space model and
    extract the gallery (train) embeddings and probe (test) embeddings
    """

    trainData = loadDataset(Config, 'train', dataset)
    trainLoader = DataLoader(trainData, shuffle=False,
                             num_workers=8, batch_size=1)
    trcv_model.eval()
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)):
            rOE, rCE, mvrOE, mvrCE, _, _ = trcv_model(ref_data)
            ref_OE[i] = rOE.squeeze().detach().cpu().numpy()
            ref_CE[i] = rCE.squeeze().detach().cpu().numpy()
            ref_mv_CE[i] = mvrCE.detach().cpu().numpy()
            ref_mv_OE[i] = mvrOE.detach().cpu().numpy()
    del trainData
    del trainLoader
    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData, shuffle=False,
                            num_workers=8, batch_size=1)
    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)):
            # tOE, tCE, mvtOE, mvtCE, _, _ = trcv_model(test_data)
            # test_OE[j] = tOE.squeeze().detach().cpu().numpy()
            # test_CE[j] = tCE.squeeze().detach().cpu().numpy()
            # label_obj[j] = obj_labels
            # test_mv_CE[j] = mvtCE.detach().cpu().numpy()
            # test_mv_OE[j] = mvtOE.detach().cpu().numpy()
            # label_mv_OE[j] = j
            if nview == 1:
                """ If Single View Query """
                tOE, tCE, _, _, _, _ = trcv_model(test_data)
                test_OE[j] = tOE.squeeze().detach().cpu().numpy()
                test_CE[j] = tCE.squeeze().detach().cpu().numpy()
                label_obj[j] = obj_labels

            elif nview > 1:
                """ If Multi View Query """
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
                _, _, refmvobj, _, _, _ = trcv_model(
                    torch.unsqueeze(torch.squeeze(test_data)[:p], 0))
                _, _, testmvobj, _, _, _ = trcv_model(
                    torch.unsqueeze(torch.squeeze(test_data)[p:], 0))
                ref_mv_OE[j] = refmvobj.squeeze().detach().cpu().numpy()
                test_mv_OE[j] = testmvobj.squeeze().detach().cpu().numpy()
                label_mv_OE[j] = j

    if nview == 1:
        """ Compute performance from single-view query """
        start2 = time.time()
        SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc = NNC_OWSC_SV(
            list(ref_CE.values()), list(test_OE.values()), list(test_CE.values()), Config)
        end2 = time.time()
        print("Time (Inference): ", (end2 - start2))
        return SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc

    if nview > 1:
        """ Compute performance from multi-view query """
        start2 = time.time()
        MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc = NNC_OWSC_MV(list(ref_mv_CE.values()), list(
            ref_mv_OE.values()), list(test_mv_CE.values()), list(test_mv_OE.values()), Config)
        end2 = time.time()
        print("Time (Inference): ", (end2 - start2))
        return MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc

