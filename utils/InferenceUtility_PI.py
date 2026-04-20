
"""
Functions for Pose-invariant Classification and Retrieval (PiCR)
category and object level task
"""

import torch
from tqdm import tqdm
import time
from utils.DataUtility import loadDataset
from utils.Faiss_KNN import FaissKNeighbors
from utils.rank_metrics import calculate_mAP
import operator
from functools import reduce
import faiss
import numpy as np
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
print(torch.__version__)


"""
===============================================================================
                             Inference framework
===============================================================================
"""

# Prediction of object label using FaissKNeighbors


def predict_object_label(Query, Database, labels):
    CLS_KNN = FaissKNeighbors(1)  # K = 1
    CLS_KNN.fit(np.array(Database), np.array(labels))
    obj_predict = CLS_KNN.predict(np.array(Query))
    return obj_predict


# Retrieve of objects
def retrieve_object_otherwise(Query, Database, labels):
    RET_KNN = FaissKNeighbors(len(Database))  # K = whole database
    RET_KNN.fit(np.array(Database), np.array(labels))
    objranks = RET_KNN.neighbors(np.array(Query))
    return objranks


# Predict single view object recognition and retrieval accuracy
def predict_SVOR(testemb, testlabel):
    acc = 0
    ret_otherwise = []
    for k in tqdm(range(len(testemb))):
        database = testemb.copy()
        labels = testlabel.copy()
        probe = [testemb[k]]
        gt = testlabel[k]

        del database[k]
        del labels[k]
        print(labels)

        # in retrieval and recognition task
        # probe = query (from testemb)
        # in recognition task
        # gt = predition results
        op_cls = predict_object_label(probe, database, labels)
        if op_cls == gt:
            acc += 1
        op_ret = retrieve_object_otherwise(probe, database, labels)
        ret_otherwise.append(op_ret)

    obj_mAP = calculate_mAP(np.squeeze(
        np.asarray(ret_otherwise)), torch.tensor(testlabel))
    print("SV Obj Accuracy: ", acc/len(testlabel)*100)
    print("SV Obj Retrieval: ", obj_mAP*100)
    return acc/len(testlabel)*100


# Predict multi-view object recognition and retrieval accuracy
def predict_MVOR(refemb, testemb, testlabel, Config):
    """Compute multi-view object-level recognition accuracy"""
    acc = 0
    op = predict_object_label(testemb, refemb, testlabel)
    for x in tqdm(range(len(testlabel))):
        if op[x] == testlabel[x]:
            acc += 1
    print("MV Object Recognition Accuracy: ", acc/len(testlabel)*100)
    """Compute multi-view object-level retrieval mAP"""
    # Rank all objects in test database for a multi-view query of a particular object
    # Note: reference views and test views for each object are disjoint
    OR_KNN = FaissKNeighbors(Config.Ntest)
    OR_KNN.fit(np.array(refemb), np.array(testlabel))
    objranks = OR_KNN.neighbors(np.array(testemb))
    obj_mAP = calculate_mAP(objranks, torch.tensor(testlabel))
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


# Nearest Neighbors Classifers
# Single View object recognition & retrieval
# Single view category recognition & retrieval
def NNC_PIE_CLS(refC_emb, testO_emb, testC_emb, Config):
    """
    refC_emb: category embedding (train)
    testC_emb: category embedding (test)
    testO_emb: object embedding (test)
    Config: Dataset configurations (metadata)
    """

    cacc = 0  # class recognition accuracy
    oacc = 0  # object recognition accuracy
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest

    for i, x in enumerate(refC_emb):
        if x.ndim == 1:
            refC_emb[i] = x.reshape(1, x.shape[0])

    # Combining all the reference embeddings into 1 training data
    XTrain = np.concatenate(refC_emb, axis=0)
    oTrain = [[i]*refC_emb[i].shape[0]
              for i in range(Config.Ntrain)]  # Object label
    oTrain = torch.tensor(reduce(operator.concat, oTrain))

    cTrain = [[o2cTrain[i]]*refC_emb[i].shape[0]
              for i in range(Config.Ntrain)]  # Category label
    cTrain = torch.tensor(reduce(operator.concat, cTrain))

    XCTest = np.concatenate(testC_emb, axis=0)  # For category test
    XOTest = np.concatenate(testO_emb, axis=0)  # For object test
    oTest = [[i]*testO_emb[i].shape[0] for i in range(Config.Ntest)]
    oTest = torch.tensor(reduce(operator.concat, oTest))
    cTest = [[o2cTest[i]]*testC_emb[i].shape[0] for i in range(Config.Ntest)]
    cTest = torch.tensor(reduce(operator.concat, cTest))

    """Compute single-view object-level retrieval mAP
    For a given single-view query of an object retrieve other views of the same object"""
    OR_KNN = FaissKNeighbors(Config.Ntest*Config.N_G)
    OR_KNN.fit(np.array(XOTest), np.array(oTest))
    objranks = OR_KNN.neighbors(np.array(XOTest))
    obj_mAP = calculate_mAP(objranks, oTest, 'other_view')
    print("SV Object Retrieval mAP: ", obj_mAP)
    del OR_KNN
    del objranks

    """Compute single-view object recognition accuracy """
    OC_KNN = FaissKNeighbors(1)
    OC_KNN.fit(np.array(XOTest), np.array(oTest))
    objpredict = OC_KNN.predict_sv(np.array(XOTest))
    for x, gt in enumerate(oTest):
        if objpredict[x] == gt:
            oacc += 1
    print("SV Object Recognition Accuracy: ", oacc/len(oTest))
    del OC_KNN
    del objpredict

    """Compute single-view category recognition accuracy """
    CLS_KNN = FaissKNeighbors(10)
    CLS_KNN.fit(np.array(XTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))

    for t in range(0, len(cTest)):
        if cls_predict[t] == cTest[t]:
            cacc += 1
    print("SV Category Recognition: ", cacc/len(cTest))

    del CLS_KNN
    del cls_predict
    del XTrain
    del cTrain
    del oTrain

    """Compute single-view category retrieval mAP """
    CR_KNN = FaissKNeighbors(Config.Ntest*Config.N_G)
    CR_KNN.fit(np.array(XCTest), np.array(cTest))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP(crranks, cTest)
    print("SV Category Retrieval mAP: ", cr_mAP)

    del CR_KNN
    del crranks

    return obj_mAP*100, cr_mAP*100, cacc/len(cTest)*100, oacc/len(oTest)*100


"""
In recognition FaissKNeighbors took 1 sample but in retrieval took Config.Ntest*Config.N_G
recognition only needs 1 sample to recognize anything
but retrieval need many samples to retrieve similar items like given input samples
"""


# Multi view category recognition & retrieval
def NNC_PIE_CLS_MV(refC_emb, testC_emb, Config):
    cacc = 0  # Category recognition accuracy
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest
    XTrain = np.concatenate(refC_emb, axis=0)

    cTrain = torch.tensor([o2cTrain[i] for i in range(Config.Ntrain)])

    XCTest = np.concatenate(testC_emb, axis=0)
    cTest = torch.tensor([o2cTest[i] for i in range(Config.Ntest)])

    """Compute multi-view category retrieval mAP """
    CR_KNN = FaissKNeighbors(Config.Ntest)
    CR_KNN.fit(np.array(XCTest), np.array(cTest))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP(crranks, cTest)

    print("MV Category Retrieval mAP: ", cr_mAP)

    del CR_KNN
    del crranks

    """Compute multi-view category recognition accuracy """
    CLS_KNN = FaissKNeighbors(10)
    CLS_KNN.fit(np.array(XTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))

    for t in range(0, len(cTest)):
        if cls_predict[t] == cTest[t]:
            cacc += 1

    print("MV Category Recognition Accuracy: ", cacc/len(cTest))

    del CLS_KNN
    del cls_predict

    return cr_mAP*100, cacc/len(cTest)*100


"""
===============================================================================
Evaluate PI category and object-level classification and retrieval performance 
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
        # crmap: single view class retrieval map
        # ormap: single view object retrieval map
        # clsacc: single view class recognition accuracy
        # svor: single view object recognition accuracy
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
        # similarly for multi-view
        return crmap, clsacc, mvor, mvoret


"""
===============================================================================
Evaluate PI (Pose Invariant) category and object-level classification and retrieval performance 
                            for Dual embedding space
===============================================================================
"""


def evaluate_PI_performance_dual(dataset, Config, trcv_model, nview):
    ref_OE = {}  # Single view object embeddings (train)
    ref_CE = {}  # Single view category embedding (train)
    test_OE = {}  # Single view object embedding (test)
    test_CE = {}  # Single view category embedding (test)

    ref_mv_OE = {}  # Multi view ojbect embedding (train)
    test_mv_OE = {}  # Multi view object embedding (test)
    mv_ref_CE = {}  # Multi view category embedding (train)

    label_cls = {}  # Class/category labels
    label_obj = {}  # Object labels
    label_mv_OE = {}  # Multi view object embedding label

    """
    Load the Dual Pose-invariant embedding space model and 
    extract the gallery (train) embeddings and probe (test) embeddings
    """
    trainData = loadDataset(Config, 'train', dataset)
    trainLoader = DataLoader(trainData, shuffle=False,
                             num_workers=16, batch_size=1)
    trcv_model.eval()
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)):
            rOE, rCE, mvrOE, mvrCE, _, _ = trcv_model(ref_data)
            ref_OE[i] = rOE.squeeze().detach().cpu().numpy()
            ref_CE[i] = rCE.squeeze().detach().cpu().numpy()
            mv_ref_CE[i] = mvrCE.detach().cpu().numpy()
    del trainData
    del trainLoader
    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData, shuffle=False,
                            num_workers=16, batch_size=1)
    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)):
            if nview == 1:
                """ if single-view query """
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
        ormap, crmap, clsacc, svor = NNC_PIE_CLS(list(ref_CE.values()), list(
            test_OE.values()), list(test_CE.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        # crmap: single view class retrieval map
        # ormap: single view object retrieval map
        # clsacc: single view class recognition accuracy
        # svor: single view object recognition accuracy
        return crmap, clsacc, ormap, svor

    if nview > 1:
        """ Compute performance from multi-view query """
        start2 = time.time()
        crmap, clsacc = NNC_PIE_CLS_MV(
            list(mv_ref_CE.values()), list(test_CE.values()), Config)
        mvor, mvoret = predict_MVOR(list(ref_mv_OE.values()), list(
            test_mv_OE.values()), list(label_mv_OE.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        # similarly for multi-view also
        return crmap, clsacc, mvoret, mvor
