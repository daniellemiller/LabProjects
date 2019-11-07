import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from tqdm import tqdm
from functools import reduce
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pickle
import keras
import pandas_profiling
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input
from Bio import SeqIO
import glob

from selection_model_analysis import get_entropy_profile_per_sequence, get_joint_entropy_profile_per_sequence, deltaG_profile_per_sequence



def generate_train_matrix(seq, seq_id, w=200):
    """
    generate the input matrix to train in the GMM model. feature generation + normalization
    :param seq: a sequence of a genome
    :param seq_id: sequence identifier
    :param w: widow size for entropy profile
    :return: data frame with all features
    """

    ks = [1, 2, 3, 4, 5]
    dfs = []
    for k in tqdm(ks):
        data_type = 'Shannon'
        alias = data_type + '_k' + str(k)
        profile = get_entropy_profile_per_sequence(seq=seq, w=w, alias=alias, k=k)
        profile = profile / profile.max()
        profile['position'] = profile.index + 1
        dfs.append(profile)

    for k in tqdm(ks):
        data_type = 'Joint'
        alias = data_type + '_k' + str(k)
        profile = get_joint_entropy_profile_per_sequence(seq=seq, w=w, alias=alias, k=k)
        profile = profile / profile.max()
        profile['position'] = profile.index + 1
        dfs.append(profile)

    # delta G profile is not dependent on k
    data_type = 'DeltaG'
    alias = data_type
    profile = deltaG_profile_per_sequence(seq=seq, w=w, alias=alias)
    profile = profile / profile.min()
    profile['position'] = profile.index + 1
    dfs.append(profile)

    mat = reduce(lambda left,right: pd.merge(left,right, on=['position']), dfs)
    return mat

def fit_GMM(train, k=4, dim_reduction=False):
    """
    fits GMM model to a given matrix
    :param train:
    :param dim_reduction: indicator whether or not to remove the dimension
    :param k: number of clusters. default 4
    :return: a data frame containing cluster assignments
    """

    if dim_reduction:
        encoding_dim = 2
        input_data = Input(shape=(train.shape[1],))

        # Define encoding layer
        encoded = Dense(encoding_dim, activation='elu')(input_data)

        # Define decoding layer
        decoded = Dense(train.shape[1], activation='sigmoid')(encoded)

        encoder = Model(inputs=input_data, outputs=encoded)
        encoded_train = pd.DataFrame(encoder.predict(train))
        encoded_train.rename(columns={0: 'principal component 1', 1: 'principal component 2'}, inplace=True)
        gmm = GaussianMixture(n_components=k)
        gmm.fit(encoded_train)
        clusters_gmm = gmm.predict(train)
        proba = gmm.predict_proba(train)
        train['GMM_clusters'] = clusters_gmm

    else:
        gmm = GaussianMixture(n_components=k)
        gmm.fit(train)
        clusters_gmm = gmm.predict(train)
        proba = gmm.predict_proba(train)
        train['GMM_clusters'] = clusters_gmm


    # add the label as a string
    train['GMM_clusters'] = train['GMM_clusters'].apply(lambda x: str(int(x)))

    # add probabilities of each point to each cluster
    for i in range(k):
        train['prob_cluster_{}'.format(i)] = proba[:,i]

    return train


def cluster_score(cluster):
    """
    calculate the score of a cluster by feature properties
    :param cluster: data frame containing the cluster matrix
    :return: a numeric value of the result
    """

    cols_2_consider = ['Shannon_k{}'.format(k) for k in [1, 2, 3, 4, 5]] + ['Joint_k{}'.format(k) for k in
                                                                            [1, 2, 3, 4, 5]] + ['DeltaG']
    total_score = 0
    for c in cols_2_consider:
        med = cluster[c].median()
        std = cluster[c].std()

        if c == 'DeltaG':
            total_score += (1 / (med*(1-med))) * std
        else:
            total_score += (1 / med ) * std
    return total_score

def score_to_rank(scores_mapping):
    """
    translate the scores into rankings
    :param scores_mapping: a list of tuples containing the score for each cluster
    :return: a mapping of cluster to rank
    """
    # sort the list by value in tup[1]
    sorted_scores = sorted(scores_mapping, key = lambda tup:tup[1], reverse=True)
    cluster_2_rank = {tup[0]:sorted_scores.index(tup)+1 for tup in sorted_scores}
    return cluster_2_rank

def run_pipeline(fasta_file, out):
    """
    run conservation piepline
    :param fasta_file: a fasta file containing sequences
    :param out: output folder to save the results to
    :return:
    """
    prev = ''
    i = 0
    for rec in tqdm(SeqIO.parse(fasta_file, 'fasta')):
        if prev != '' and prev in rec.description:
            continue
        seq = str(rec.seq)
        alias = rec.description
        mat = generate_train_matrix(seq, alias)
        gmm_clusterd = fit_GMM(mat)
        gmm_clusterd['seq_id'] = alias
        gmm_clusterd.to_csv(os.path.join(out, 'input_mat_{}.csv'.format(alias)))
        i += 1
        prev = alias

    print("Done!, saved {} file to {}".format(i, out))


####### run analysis #########
fasta = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/data/Phylogeny/family/Togaviridae/Togaviridae.fasta'
out = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/training'

##### remove this commet to run the pipeline on all dataset
#run_pipeline(fasta, out)



def seq_2_id_mapping(fasta):
    """
    create a mapping between the refseq id and the sequence numeric value in the iteration
    :param fasta: a fasta file
    :return:  a dictionary with seq_num : seq_id
    """
    mapping = {}
    i = 0
    for rec in tqdm(SeqIO.parse(fasta, 'fasta')):
        mapping['seq_{}'.format(i)] = rec.description
    return mapping

def get_median_conservation_score(start, end, seq_id, alias):
    """
    return the median conservation score of a drop, by its start and end
    :param start: start position of a drop
    :param end: end position of a drop
    :param seq_id: the refseq id as decribed in the fasta file of a given sequence
    :param alias: the family alias
    :return: a float with the median conservation score
    """

    input_file = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/{}/{}_out_parsed.csv'.format(alias, seq_id)
    conv = pd.read_csv(input_file)

    median_conv = conv[conv['position'].isin(list(range(start, end+1)))]['score_bin'].median()
    return median_conv


def merge_drops_n_conservation(alias, drop_type):
    """
    merge drops data with conservation data
    :param alias: the family alias
    :param drop_type: shannon or joint
    :return:
    """

    fasta = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/data/Phylogeny/family/{}/{}.fasta'.format(alias, alias)
    drops = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/WebDB/combined/{}_stats.csv'.format(alias)
    train = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/training/{}'.format(alias)
    consv = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/{}'.format(alias)

    # create the mapping from seq ti id:
    mapping = seq_2_id_mapping(fasta)

    # define the
    cols_2_consider = ['Shannon_k{}'.format(k) for k in [1, 2, 3, 4, 5]] + ['Joint_k{}'.format(k) for k in
                                                                            [1, 2, 3, 4, 5]] + ['DeltaG', 'position']

    # get median conservation score for each drop
    drops = pd.read_csv(drops)
    drops = drops[drops['type'] == drop_type]
    drops['seq_id'] = drops['seq'].apply(lambda x: mapping[x])
    drops['median_conservation'] = drops.apply(lambda row: get_median_conservation_score(row['start'], row['end'],
                                                                                         row['seq_id'], alias), axis = 1)



    files = glob.glob(os.path.join(train, 'input_*.csv'))
    for f in files:
        df = pd.read_csv(f)
        df = df[cols_2_consider]     # entropies
        id_2_seq = [k for k, v in mapping if k[v] == alias][0] # choose the first seq arbitrarily



