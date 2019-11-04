import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from tqdm import tqdm
from functools import reduce

from selection_model_analysis import get_entropy_profile_per_sequence, get_joint_entropy_profile_per_sequence, deltaG_profile_per_sequence



def generate_train_matrix(seq, alias, k=5, w=200):
    """
    generate the input matrix to train in the GMM model. feature generation + normalization
    :param seq: a sequence of a genome
    :param alias: sequence identifier
    :param k: the k-mer size for entropy calculations
    :param w: widow size for entropy profile
    :return: data frame with all features
    """

    ks = [1, 2, 3, 4, 5]
    dfs = []
    for k in tqdm(ks):
        data_type = 'Shannon'
        alias = data_type + '_k' + str(k)
        profile = get_entropy_profile_per_sequence(seq=seq, w=200, alias=alias, k=k)
        profile = profile / profile.max()
        profile['position'] = profile.index + 1
        dfs.append(profile)

    for k in tqdm(ks):
        data_type = 'Joint'
        alias = data_type + '_k' + str(k)
        profile = get_joint_entropy_profile_per_sequence(seq=seq, w=200, alias=alias, k=k)
        profile = profile / profile.max()
        profile['position'] = profile.index + 1
        dfs.append(profile)

    # delta G profile is not dependent on k
    data_type = 'DeltaG'
    alias = data_type
    profile = deltaG_profile_per_sequence(seq=seq, w=200, alias=alias)
    profile = profile / profile.min()
    profile['position'] = profile.index + 1
    dfs.append(profile)

    mat = reduce(lambda left,right: pd.merge(left,right, on=['position']), dfs)
    return mat

def fit_GMM(train, k, dim_reduction=False):
    """
    fits GMM model to a given matrix
    :param train:
    :param dim_reduction: indicator whether or not to remove the dimension
    :param k: number of clusters
    :return: a data frame containing cluster assignments
    """




