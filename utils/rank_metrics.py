
"""
Code adopted from https://gist.github.com/bwhite/3726239

"""
"""Information Retrieval metrics

Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
Learning to Rank for Information Retrieval (Tie-Yan Liu)
"""


"""Mean Reciprocal Rank"""
# it determines, how early does the first correct result appears
"""
For example,
rs = [
    [0, 0, 1],   # query 1
    [0, 1, 0],   # query 2
    [1, 0, 0]    # query 3
]
1 = relevant (correct)
0 = irrelevant (incorrect)

rs = (np.asarray(r).nonzero()[0] for r in rs)
This line converts list to array and gives the indices where value is not euqals 0
so rs = [[2],[1],[0]]
then it returns the mean
"""


import numpy as np
def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1./(r[0]+1) if r.size else 0. for r in rs])


"""
R-precision = precision at the point where all relevant items have been retrieved
Instead of asking 'how early is the first correct result? 
or
How good the ranking overall?

R-precision ask, "when i've seen all correct items, 
how clean were my results up to that point?"

Example,
r = [0,1,0], 1 = relevant, 0 = irrelevant
r = np.asarray(r)!=0 -> converts into boolean = [False, True, False]
z = r.nonzero()[0] -> get indices of relevant items = 1 (True is the relevant item)
r[:z[-1]+1] = z[-1] = 1, r[:2] = [0,1]
mean([0,1]) = 0.5

"""

def r_precision(r):
    r = np.asarray(r)!=0
    z = r.nonzero()[0]
    if not z.size:
        return 0
    return np.mean(r[:z[-1]+1])



"""
Precision at k = how many top k results are correct
"""
def precision_at_k(r,k):
    assert k>=1
    r = np.asarray(r)[:k]!=0
    if r.size!=k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r)!=0
    out = [precision_at_k(r,k+1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])



"""
DCG = Discounted Cumulative Gain
It measures "How good the results, giving more importance to 
higher ranks"

relevant results at the top -> more valuable
relevant results at the bottom -> less valuable
so we discount the lower ranks

"""
def dcg_at_k(r,k,method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0]+np.sum(r[1:]/np.log2(np.arange(2,r.size+1)))
        if method == 1:
            return np.sum(r/np.log2(np.arange(2,r.size+2)))
        else:
            raise ValueError('Method must be 0 or 1')
    return 0


# Normalized Discounted Cumulative Gain
def ndcg_at_k(r,k,method = 0):
    dcg_max = dcg_at_k(sorted(r,reverse=True),k,method)
    if not dcg_max:
        return 0
    return dcg_at_k(r,k,method)/dcg_max


# mAP = mean average precision
# pred = model prediction
# gt = ground truth
def calculate_mAP(pred, gt, mode=''):
    AP=[]
    for t in range(0,len(gt)):
        rank = pred[t,:] == gt[t].numpy()
        if mode == 'other_view':
            ap = average_precision(rank[1:])
        else:
            ap = average_precision(rank)
        AP.append(ap)
    mAP = np.mean(AP)
    return mAP


def calculate_mAP_large(pred_indices, labels, mode=''):
    AP=[]
    for t in range(0,len(labels)):
        rank = labels[pred_indices[t,:]] == labels[t]
        if mode == 'other_view':
            ap = average_precision(np.array(rank[1:]))
        else:
            ap  = average_precision(np.array(rank))
        AP.append(ap)
    mAP = np.mean(AP)
    return mAP

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    

