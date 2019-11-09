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

def generate_list_of_sequetial_positions(drops):
    """
    create a list of positions [start_i, end_i for i in drops]
    :param drops: a data frame of a single sequence drops information
    :return: a list of all positions
    """
    l_start = drops['start'].values
    l_end = drops['end'].values

    l_combined = []
    for i in range(len(l_start)):
        l_combined.append(l_start[i])
        l_combined.append(l_end[i])
    return np.array(l_combined)

def get_drop_id_by_pos(pos, drops):
    """

    :param pos:
    :param drops:
    :return:
    """

    all_drop_positions = generate_list_of_sequetial_positions(drops)
    pos_idx = all_drop_positions.searchsorted(pos)

    # check if its between drops or within drop
    set_start = -1
    set_end = -1

    if pos_idx % 2 != 0: # within drop
        set_start = all_drop_positions[pos_idx - 1]
        set_end = all_drop_positions[pos_idx + 1]
    else:
        if pos in all_drop_positions:
            set_start = pos
            set_end = all_drop_positions[pos_idx + 1]

    if set_start == -1:
        return "no drop"
    else:
        return drops[(drops['start'] == set_start) & (drops['end'] == set_end)]['drop_id'].values[0]




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

    # remove duplicated seq_i rows
    ids_2_consider = []
    seq_id_used = []
    for key in mapping.keys():
        v = mapping[key]
        if v in seq_id_used:
            continue
        else:
            seq_id_used.append(v)
            ids_2_consider.append(key)
    drops = drops[drops['seq'].isin(ids_2_consider)]

    drops['median_conservation'] = drops.apply(lambda row: get_median_conservation_score(row['start'], row['end'],
                                                                                         row['seq_id'], alias), axis = 1)

    # add the drop identifier
    indices = ["Drop {}".format(i) for i in drops.index.values]
    drops['drop_id'] = indices

    # merge data with train matrix for each id separately
    for seq_id in drops['seq_id'].unique():
        train_data = pd.read_csv(os.path.join(train, "input_mat_{}.csv".format(seq_id)))
        cur_drop = drops[drops['seq_id'] == seq_id]

        train_data['drop_id'] = train_data['pos'].apply(lambda x: get_drop_id_by_pos(x, cur_drop))

        train_drops = train_data[train_data['drop_id'] != 'no drop']

        # now we have a matrix containing only drops for each sequence, with their identifier.
        # fit a GMM model to all positions remained, including probabilities and calculate the rank of each drop
        gmm_clusterd = fit_GMM(train_drops[cols_2_consider])
        merged = pd.merge(gmm_clusterd, train_drops[['position', 'seq_id', 'drop_id']])

        # add ranks to each drop.
        score_2_cluster = []
        for cluster in ["0", "1", "2", "3"]:
            score_2_cluster.append(tuple(cluster, cluster_score(merged[merged['GMM_clusters'] == cluster])))

        ranking = score_to_rank(score_2_cluster)
        merged['rank'] = merged['GMM_clusters'].apply(lambda x: ranking[x])
        merged.to_csv(os.path.join(train, '{}_clustered_and_ranked.csv'.format(alias)))












