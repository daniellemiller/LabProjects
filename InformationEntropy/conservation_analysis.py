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
import pickle

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
        i += 1
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

    input_file = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/recombination/' \
                 r'recombination_free_fasta/r4s_out_aln_{}_{}_parsed.csv'.format(alias, seq_id)
    conv = pd.read_csv(input_file)

    median_conv = conv[conv['pos'].isin(list(range(start, end+1)))]['score_bin'].median()
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
        set_end = all_drop_positions[min(pos_idx, all_drop_positions.shape[0] - 1)]
    else:
        if pos in all_drop_positions:
            set_start = pos
            set_end = all_drop_positions[min(pos_idx + 1, all_drop_positions.shape[0] - 1)]

    if set_start == -1 or set_end == -1:
        return "no drop"
    else:
        val = drops[(drops['start'] == set_start) & (drops['end'] == set_end)]['drop_id'].values[0]
        return val

def get_drop_conservation_by_pos(pos, drops):
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
        set_end = all_drop_positions[min(pos_idx, all_drop_positions.shape[0] - 1)]
    else:
        if pos in all_drop_positions:
            set_start = pos
            set_end = all_drop_positions[min(pos_idx + 1, all_drop_positions.shape[0] - 1)]

    if set_start == -1 or set_end == -1:
        return "Error"
    else:
        val = drops[(drops['start'] == set_start) & (drops['end'] == set_end)]['median_conservation'].values[0]
        return val



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

    recomb_free_fasta = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/recombination/' \
                 r'recombination_free_fasta/{}.fasta'.format(alias, alias)

    # get the non recombinant ids to consider at the end
    non_recombinant_ids = []
    for rec in SeqIO.parse(recomb_free_fasta, 'fasta'):
        non_recombinant_ids.append(rec.description)

    # create the mapping from seq ti id:
    mapping = seq_2_id_mapping(fasta)

    # define the
    cols_2_consider = ['Shannon_k{}'.format(k) for k in [1, 2, 3, 4, 5]] + ['Joint_k{}'.format(k) for k in
                                                                            [1, 2, 3, 4, 5]] + ['DeltaG', 'position']

    # get median conservation score for each drop
    drops = pd.read_csv(drops)
    drops = drops[drops['type'] == drop_type]
    drops['seq_id'] = drops['seq'].apply(lambda x: mapping[x])

    if os.path.exists(r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/{}_mapped_drops.pickle'.format(alias)):
        drops = pd.read_pickle(r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/{}_mapped_drops.pickle'.format(alias))
    else:
        # remove duplicated seq_i rows
        ids_2_consider = []
        seq_id_used = []
        for key in mapping.keys():
            v = mapping[key]
            if v in seq_id_used or v not in non_recombinant_ids:
                continue
            else:
                seq_id_used.append(v)
                ids_2_consider.append(key)
        print(ids_2_consider)
        drops = drops[drops['seq'].isin(ids_2_consider)]

        drops['median_conservation'] = drops.apply(lambda row: get_median_conservation_score(row['start'], row['end'],
                                                                                             row['seq_id'], alias), axis = 1)

        with open(r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/{}_mapped_drops.pickle'.format(alias), 'wb') as fp:
            pickle.dump(drops, fp)


    # merge data with train matrix for each id separately
    for seq_id in tqdm(drops['seq_id'].unique()):
        train_data = pd.read_csv(os.path.join(train, "input_mat_{}.csv".format(seq_id)))
        cur_drop = drops[drops['seq_id'] == seq_id]
        cur_drop = cur_drop.reset_index(drop=True)
        cur_drop['drop_id'] = cur_drop.apply(lambda row: "Drop {}".format(row.name), axis=1)

        train_data['drop_id'] = train_data['position'].apply(lambda x: get_drop_id_by_pos(x, cur_drop))

        train_drops = train_data[train_data['drop_id'] != 'no drop']
        train_drops['median_conservation'] = train_drops['position'].apply(lambda x: get_drop_conservation_by_pos(x, cur_drop))

        # now we have a matrix containing only drops for each sequence, with their identifier.
        # fit a GMM model to all positions remained, including probabilities and calculate the rank of each drop
        gmm_clusterd = fit_GMM(train_drops[cols_2_consider])
        merged = pd.merge(gmm_clusterd, train_drops[['position', 'seq_id', 'drop_id', 'median_conservation']])

        # add ranks to each drop.
        score_2_cluster = []
        for cluster in ["0", "1", "2", "3"]:
            score_2_cluster.append(tuple((cluster, cluster_score(merged[merged['GMM_clusters'] == cluster]))))

        ranking = score_to_rank(score_2_cluster)
        merged['rank'] = merged['GMM_clusters'].apply(lambda x: ranking[x])
        merged.to_csv(os.path.join(train, '{}_clustered_and_ranked.csv'.format(seq_id)))


## here add an example for one family (will run on the cluster)
##

#merge_drops_n_conservation('Iflaviridae', 'shannon')

######## post run analysis

EPS = 0.000001
def enrichment(data, rank):
    """
    returns the conservation enrichment score of a given rank
    :param data: dataframe with drops ids, ranks and conservation score
    :param rank: int representing the rank class
    :return: float. the enrichment score for a given rank
    """

    data = data[data['rank'] == rank]
    passed = data[data['median_conservation'] <= 5]['drop_id'].unique().shape[0]
    not_passed = data['drop_id'].unique().shape[0] - passed
    enriched = np.log((passed + EPS) / (not_passed + EPS))

    return enriched



def run_enrichments():
    all_ranked_files = glob.glob(r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/training/*/*rank*.csv')
    ranks = [1,2,3,4]

    fams = np.unique(np.array([os.path.basename(os.path.dirname(f)) for f in all_ranked_files]))

    enrichments = []
    for f in tqdm(fams):
        drops = [pd.read_csv(r) for r in all_ranked_files if f in r]
        rank_enrichment = {str(k):[] for k in ranks}
        for drop_data in tqdm(drops):
            for rank in ranks:
                cur_df = pd.DataFrame({'family':f, 'rank':rank, 'enrichments':enrichment(drop_data, rank)}, index=[0])
                enrichments.append(cur_df)

    res = pd.concat(enrichments)
    return res


# aux function
def get_size(family, seq_id, drop_id, data_type='shannon'):
    """
    this function returns the relative nucleotide ratio comparing the the drop and to the entire genome
    :param family: family name
    :param seq_id: sequence identifier
    :param drop_id: drop identifier
    :param data_type: drop data type. default is shannon and not joint.
    :return: a list with tuples for each nucleotide in the following order: A,C,G,T. (nuc relative in drop, nuc relative to genome)
    """

    fasta_file = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/recombination/' \
                 r'recombination_free_fasta/{}.fasta'.format(family, family)

    seq = ''
    for rec in SeqIO.parse(fasta_file, 'fasta'):
        if rec.description == seq_id:
            seq = str(rec.seq).lower()

    if seq == '':  # this shpuld not happen, here just in case....
        print("here WTF")
        print(seq_id, family, drop_id)
        return 'Not valid! seq_id not in fasta file'

    # now we need to files start and end.
    ranked_file = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/training/{}/{}_clustered_and_ranked.csv'.format(
        family, seq_id)
    ranked = pd.read_csv(ranked_file)
    drop_positions = ranked[ranked['drop_id'] == drop_id]['position'].values

    start = int(drop_positions.min())
    end = int(drop_positions.max())

    return end - start + 1


def get_start_and_end(family, seq_id, drop_id, data_type='shannon'):
    """
    this function returns the relative nucleotide ratio comparing the the drop and to the entire genome
    :param family: family name
    :param seq_id: sequence identifier
    :param drop_id: drop identifier
    :param data_type: drop data type. default is shannon and not joint.
    :return: a list with tuples for each nucleotide in the following order: A,C,G,T. (nuc relative in drop, nuc relative to genome)
    """

    fasta_file = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/recombination/' \
                 r'recombination_free_fasta/{}.fasta'.format(family, family)

    seq = ''
    for rec in SeqIO.parse(fasta_file, 'fasta'):
        if rec.description == seq_id:
            seq = str(rec.seq).lower()

    if seq == '':  # this shpuld not happen, here just in case....
        print("here WTF")
        print(seq_id, family, drop_id)
        return 'Not valid! seq_id not in fasta file'

    # now we need to files start and end.
    ranked_file = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/training/{}/{}_clustered_and_ranked.csv'.format(
        family, seq_id)
    ranked = pd.read_csv(ranked_file)
    drop_positions = ranked[ranked['drop_id'] == drop_id]['position'].values

    start = int(drop_positions.min())
    end = int(drop_positions.max())

    return start, end


annotated = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/Entropy/conservation_analysis/sequences_annotated_all.gb'


def annotate_drop(family, seq_id, drop_id):
    # now we need to files start and end.
    ranked_file = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/training/{}/{}_clustered_and_ranked.csv'.format(
        family, seq_id)
    ranked = pd.read_csv(ranked_file)
    drop_positions = ranked[ranked['drop_id'] == drop_id]['position'].values

    start = int(drop_positions.min())
    end = int(drop_positions.max())

    if start == None or end == None:
        print("here")
        return "not valid none object!!"

    # proteins = []
    types = []
    products = []

    for rec in SeqIO.parse(annotated, 'gb'):
        if rec.name == seq_id:
            break
    for feature in rec.features:
        f_start = feature.location.start.real
        f_end = feature.location.end.real
        intersection = len(list(set(range(start, end + 1)) & set(range(f_start, f_end + 1))))

        if intersection > (end - start + 1) // 3:
            types.append(feature.type)
            if 'product' in feature.qualifiers:
                products.append(feature.qualifiers['product'])
            else:
                products.append(['no product data available'])
    return products, types


def extract_nucleotide_information(family, seq_id, drop_id, data_type='shannon'):
    """
    this function returns the relative nucleotide ratio comparing the the drop and to the entire genome
    :param family: family name
    :param seq_id: sequence identifier
    :param drop_id: drop identifier
    :param data_type: drop data type. default is shannon and not joint.
    :return: a list with tuples for each nucleotide in the following order: A,C,G,T. (nuc relative in drop, nuc relative to genome)
    """

    fasta_file = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/recombination/' \
                 r'recombination_free_fasta/{}.fasta'.format(family, family)

    seq = ''
    for rec in SeqIO.parse(fasta_file, 'fasta'):
        if rec.description == seq_id:
            seq = str(rec.seq).lower()

    if seq == '':  # this shpuld not happen, here just in case....
        print("here WTF")
        print(seq_id, family, drop_id)
        return 'Not valid! seq_id not in fasta file'

    # now we need to files start and end.
    ranked_file = r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/training/{}/{}_clustered_and_ranked.csv'.format(
        family, seq_id)
    ranked = pd.read_csv(ranked_file)
    drop_positions = ranked[ranked['drop_id'] == drop_id]['position'].values

    start = int(drop_positions.min())
    end = int(drop_positions.max())

    if start == None or end == None:
        print("here")
        return "not valid none object!!"

    drop_sequence = seq[start - 1: end + 1]

    a_total = seq.count('a')
    c_total = seq.count('c')
    g_total = seq.count('g')
    t_total = seq.count('t')

    a_in_drop = drop_sequence.count('a')
    c_in_drop = drop_sequence.count('c')
    g_in_drop = drop_sequence.count('g')
    t_in_drop = drop_sequence.count('t')

    len_drop = len(drop_sequence)

    a_rel_genome = a_in_drop / a_total
    c_rel_genome = c_in_drop / c_total
    g_rel_genome = g_in_drop / g_total
    t_rel_genome = t_in_drop / t_total

    a_rel_drop = a_in_drop / len_drop
    c_rel_drop = c_in_drop / len_drop
    g_rel_drop = g_in_drop / len_drop
    t_rel_drop = t_in_drop / len_drop

    return [(a_rel_genome, a_rel_drop), (c_rel_genome, c_rel_drop), (g_rel_genome, g_rel_drop),
            (t_rel_genome, t_rel_drop)]


def rescale(value, max_range=0.6, min_range=0.3):
    """
    rescale deltaG values by random distribution
    :param value: actual value
    :param max_range: max range
    :param min_range: min range
    :return: rescaled value
    """
    scaled = value

    if value > min_range:
        scaled = (value) * (max_range - min_range) + min_range

    return scaled

def calc_denum_by_seq(vec, thresh=0.3):
    """
    calculate denumerator by scaling formula
    :param vec: vector of numbers
    :param thresh: threshold for random
    :return: sum of all scaled value in vec
    """
    return sum([(x-thresh)**2 for x in vec])

def rescore_deltag(value, denum, thresh = 0.3):
    """
    score deltaG feature by weights
    :param value:
    :param denum:
    :param thresh:
    :return:
    """
    scaled = (value - thresh)**2 / denum
    return scaled

def generate_deltag_for_rank(data, max_range = 0.6, min_range=0.3):
    """
    generate deltag scores for the entire dataset
    :param data: all drops by seq id
    :param max_range:
    :param min_range:
    :return:
    """

    # first we need to rescale all delta g values by the same threshold
    data['scaled_DeltaG'] = data['DeltaG'].apply(lambda x: rescale(x, max_range=max_range, min_range=min_range))

    # now we need to have all denuminators by seq
    g = data.groupby(['family', 'seq_id'])['scaled_DeltaG'].agg(calc_denum_by_seq).reset_index(name='denum')
    merged = pd.merge(data, g, on=['family', 'seq_id'])

    merged['scored_DeltaG'] = merged.apply(lambda row: rescore_deltag(row['scaled_DeltaG'], row['denum'], thresh=min_range), axis=1)

    return merged


def dropplot(data, feature='median_conservation', genome_len=10 ** 4):
    mapping = {}
    vals = np.sort(data[feature].unique())
    for i, cons in enumerate(vals):
        mapping[str(cons)] = i

    n_colors = 2
    if vals.shape[0] > 2:
        n_colors = max(8, vals.shape[0])

    with sns.plotting_context(rc={"font.size": 14, "axes.titlesize": 18, "axes.labelsize": 18,
                                  "xtick.labelsize": 14, "ytick.labelsize": 14, 'y.labelsize': 16}):

        pal = sns.mpl_palette('seismic', n_colors)
        with sns.plotting_context(
                rc={"font.size": 12, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 12, 'aspect': 10}):
            f, ax = plt.subplots(figsize=(14, 4))
            for i, seq in enumerate(g['seq_id'].unique()):
                g_tag = data[data['seq_id'] == seq]
                ax.plot([1, genome_len], [i, i], color="black", alpha=0.7, linewidth=4)
                for row in g_tag.iterrows():
                    row = row[1]
                    ax.scatter([row['start'], row['end']], [i, i], marker='s', s=2 * row['drop_size'],
                               c=pal[mapping[str(row[feature])]], label="{} {}".format(row['product'], row['start']))

        plt.legend(bbox_to_anchor=[1.1, 1.1])
        sns.palplot(sns.mpl_palette('seismic', n_colors))
        plt.show()


