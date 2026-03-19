
# in PiRO: Pose-invariant Representation of Object
# there are two losses that are mentioed
# pose invariant object loss (piobj) and pose-invariant category loss (picat)
# also in total loss there is also only category loss is present


# pose-invariant object loss and category loss in dual embedding space
# PiRO introduces dual embedding space for better representation, classification & retrieval of objects and categories

"""
pose-invariant object loss in dual embedding space (piobj)
"""

import torch
import torch.nn.functional as F # this is a collection of state-less functions that used to do stuffs with neural network
# torch.nn.functional = raw operations (functions)
# e.g.
# layer based: nn.ReLU()
# functional: F.relu(x)

import torch.nn as nn # for neural networ layers

# the main idea of pose-invariant losses are dependent on euclidean distances





