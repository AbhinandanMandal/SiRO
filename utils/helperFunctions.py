
# main idea of pose-invariant losses are dependent on euclidean distances
# building euclidean distances for embeddings x and y

import torch
import matplotlib.pyplot as plt
import numpy as np
from csv import writer
import random


# euclidean distances between the embeddings x, y
def expand_pairwise_distances(x, y=None):
    # dist[i,j] = ||xi - yj||^2
    # x is Nxd matrix, y is Mxd matrix (optional)
    # dist shape NxM
    # explaination
    # x:(N,d), y:(M,d)
    # x.unsqueeze(1) = x(N,1,d)
    # y.unsqueeze(0) = y(1,M,d)
    # x[i] - y[j] = (N,M,d)
    # returns embeddings of shape (N,M)

    if y is not None:
        # when x,y both embeddings
        differences = x.unsqueeze(1) - y.unsqueeze(0)
    else:
        # only x embedding is present
        differences = x.unsqueeze(1) - x.unsqueeze(0)
    distances = torch.sum(differences*differences -
                          1)  # summing the differences
    return distances  # euclidean distance between x, y


def blocks(vs, n):  # this splits a list (vs) into n block
    vs_blocks = []
    length = int(np.ceil(len(vs)/n))  # length of each block
    for i in range(0, len(vs), length):
        vs_blocks.append(vs[i:i+length])
    return vs_blocks


def partition(list_in, n):  # splitting list input into validation list & test list
    random.shuffle(list_in)
    val_list = sorted(list_in[0:n])
    test_list = sorted(list_in[n:])
    return val_list, test_list


def imshow(img, text=None, should_save=False):  # for showing any image
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):  # showing plot
    plt.plot(iteration, loss)
    plt.show()


# plot convergence curve
def plot_ConvergenceCurve(train_loss_history, test_acc_history, path):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(train_loss_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Test Accuracy', color=color)
    ax2.plot(test_acc_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(path)


# writing & appending stuffs
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


# plotting informations
def plot_infoex(infoEx, path):
    fig, ax1 = plt.subplots()
    if len(infoEx) > 1: 
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Informative Example Percentage', color=color)
        ax1.plot(infoEx[1:], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        fig.savefig(path)


# plotting distance
def plot_distance(infoEx, max_intra, min_inter, ratio, path):
    fig, ax1 = plt.subplots()

    if len(infoEx) > 1:
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Ratio', color=color)
        ax1.plot(ratio[1:], color=color, label='Ratio')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        # we already handled the x-label with ax1
        ax2.set_ylabel('Average Distance', color=color)

        # ax2.plot(inter[1:], 'b:', label='Inter-class (10 NN)')
        ax2.plot(max_intra[1:], 'g', label='Max Intra-class')
        # ax2.plot(intra[1:], 'g:', label='Avg Intra-class')
        ax2.plot(min_inter[1:], 'b', label='Min Inter-class')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.show()

        legend1 = ax1.legend(loc=0)

        legend2 = ax2.legend()
        # Put a nicer background color on the legend.
        legend2.get_frame().set_facecolor('C0')

        plt.savefig(path)

