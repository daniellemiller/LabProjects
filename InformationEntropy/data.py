import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.cluster import KMeans

import glob
import sys, os
from tqdm import tqdm

sys.path.append("/Users/daniellemiller/Documents/SternLab/")
from phylogenetic_utilities import total_branch_length
from train_rank import factorize_all_object_features


def load_data(fname):
    return pd.read_csv(fname)


def family_2_TBL():
    """
    map a viral family to its sum of BL (TBL -= total branch length)
    :return: dictionary with family name as key and TBL as value
    """
    trees = glob.glob(r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/recombination/'
                      r'recombination_free_fasta/*.aln.best.phy_phyml_tree.txt')
    mapping = {}
    for tree in tqdm(trees):
        family = os.path.basename(tree).split('.')[0]
        mapping[family] = total_branch_length(tree)
    return mapping

def SI_2_SL():
    """
    create a mapping of sequence id (SI) to sequence length (SL)
    :return: dictionary with SI as key and SL as value
    """
    mapping = {}
    fasta = glob.glob(r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/recombination/'
                 r'recombination_free_fasta/*.fasta')
    for f in fasta:
        for rec in SeqIO.parse(f, 'fasta'):
            mapping[rec.name] = len(rec.seq)
    return mapping

def mapper(data, map_by='TBL'):
    """
    map data by a defined mapper
    :param data: input dataframe
    :param map_by: TBL or SL
    :return: data with a mapped column
    """

    if map_by == 'TBL':
        mapping = family_2_TBL()
        data['TBL'] = data['family'].apply(lambda x: mapping[x])

    else:
        mapping = SI_2_SL()
        data['SL'] = data['seq_id'].apply(lambda x: mapping[x])
    return data

def bin_continous(vec, n_bins=3):
    """
    bin a continues vector
    """
    scores = np.sort(vec)
    bins = np.array_split(scores, n_bins)   # change to fit un equal partition
    bins = [bins[0][0]]+ [elem[-1] for elem in bins]
    return bins


if __name__ == "__main__":

    data = load_data(r'data/basic_marix.csv')
    data = mapper(data, 'TBL')
    data = mapper(data, 'SL')

    # divide into 3 sub groups - low- med- high
    data['TBL_class'] = np.digitize(data['TBL'], bin_continous(data['TBL'].values))
    data['SL_class'] = np.digitize(data['SL'], bin_continous(data['SL'].values))
    data['DS_class'] = np.digitize(data['drop_size'], bin_continous(data['drop_size'].values))

    classes = [1,2,3]
    metrics = ['TBL_class', 'SL_class', 'DS_class']
    mapping = {'1': 'LOW', '2':'MED', '3':'HIGH'}
    #split by metric
    for cls in classes:
        for met in metrics:
            data[data[met] == cls].to_csv(f"data/train_{met}_{mapping[str(cls)]}.csv", index=False)

    #split by clustering
    ks = np.arange(2,11)
    data = factorize_all_object_features(data).drop(columns=['TBL_class', 'TBL'])
    for k in ks:
        k_data = data.copy()
        k_data['Kmeans_cluster'] = KMeans(n_clusters=k).fit(data).labels_

        k_data.to_csv(f"data/train_Kmeans_K{k}.csv", index=False)