import pandas as pd
import numpy as np
import os
from utils import entropy_by_kmer
import re
from tqdm import tqdm
import math
from functools import reduce
import string
from Bio import SeqIO, Phylo
from itertools import combinations
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import heritability

def remove_punctuation(s):
    splitted = string.punctuation.split('_')
    marks = splitted[0]+splitted[1]

    regex = re.compile('[%s]' % re.escape(marks))
    return regex.sub('', s).strip()


def genome_2_entropy(fasta, k):
    """
    create a mapping of the complete genome to its entropy by kmer
    :param fasta: a file containing all fasta files
    :param k: the k-mer size
    :return: a data frame of ref-seq id and entropy measurement
    """

    k_entropy = []
    ref_seq_id = []

    sequences = re.split(">", open(fasta, "r").read().replace('\n', ''))[1:]
    for seq in tqdm(sequences):
        if '.' not in seq:
            print('no dot in sequence name\n')
            continue
        if 'complete genome' not in seq:
            print('not complete genome\n')
            continue

        # get identifier and genomic sequence
        splitted = seq.split('.')
        identifier = remove_punctuation(splitted[0].split()[0])
        #identifier = splitted[0].split()[0]
        genome = splitted[-1]

        # calculate entropy
        entropy = entropy_by_kmer(genome, k)


        k_entropy.append(entropy)
        ref_seq_id.append(identifier)

    df = pd.DataFrame({'refseq_id':ref_seq_id, 'entropy_{}'.format(k):k_entropy})
    return df

def string_by_codon_position(seq, position):
    """
    extract only the specified position of a codon triplet
    :param seq: a string
    :param position: the position in the codon can be 0,1,2
    :return: the sting containing only the codon at position in each triplet
    """

    triplets = [seq[i:i+3] for i in range(0, len(seq), 3)]
    filtered = ''
    filtered = filtered.join([tripl[1] for tripl in triplets if len(tripl) == 3])
    return filtered

def refseq_2_cds(cds):
    """
    creates a mapping of the cds to refseq
    :param cds: a fasta files
    :return:
    """


    # create a mapping between each id all the coding sequences it has.
    mapping = {}
    sequences = re.split(">", open(cds, "r").read().replace('\n', ''))[1:]
    for seq in sequences:
        splitted = seq.split('|')
        # if splitted[2] != '':
            # print(splitted[0], splitted[2], splitted[3])
        refseq_id = remove_punctuation(splitted[3])
        # refseq_id = splitted[3]
        if '}' in seq:
            coding = splitted[-1].split('}')[-1]
        else:
            coding = splitted[-1].split('}')[-1]

        # add the sequence to the dictionary
        if refseq_id in mapping:
            mapping[refseq_id].append(coding)
        else:
            mapping[refseq_id] = [coding]

    return mapping

    # now we have for each id the cds seqs.

def entropy_calculator_cds(key, values, jump=False, position=None):
    '''
    calculate the entropy according to coding regions only for a given set of . here we have two options - by reading frame or by
    some codon position
    :param key: refseq id of the sequences in values
    :param values: the cds sequences of that entry
    :param jump: reading frame indication
    :param position: the codon position
    :return: entropy
    '''

    # for each sequence get the entropy by reading frame, by k=5, and by k=1 according to each position.

    kmers = {}
    entropy = 0
    alias = ''

    for seq in values:
        if jump:
            alias = 'reading_frame'
            for i in range(0, len(seq) - 3, 3):
                kmer = seq[i:i + 3]
                if kmer in kmers:
                    kmers[kmer] += 1
                else:
                    kmers[kmer] = 1
        else:
            assert(position != None)
            alias = 'codon_{}'.format(position)
            # use k=1 in this case
            codon_trimmed = string_by_codon_position(seq, position)
            for i in range(len(codon_trimmed)):
                kmer = codon_trimmed[i]
            if kmer in kmers:
                kmers[kmer] += 1
            else:
                kmers[kmer] = 1

    # calculate entropy for all the cds of this refseq id
    total_kmers = sum(kmers.values())
    for kmer in kmers:
        p = kmers[kmer] / total_kmers
        entropy += -(p * math.log2(p))

    return entropy


def entropy_by_cds(mapping):
    """
    create a data frame with all entropy values for a given refseq id
    :param mapping: dictionary with refseq id and cds sequences
    :return: a data frame with all information.
    """
    result = []

    for key in mapping.keys():
        rf_entropy = entropy_calculator_cds(key, mapping[key], jump=True)
        codon1_entropy = entropy_calculator_cds(key, mapping[key], jump=False, position=0)
        codon2_entropy = entropy_calculator_cds(key, mapping[key], jump=False, position=1)
        codon3_entropy = entropy_calculator_cds(key, mapping[key], jump=False, position=2)
        df = pd.DataFrame({'refseq_id':key, 'rf_entropy':rf_entropy, 'codon1_entropy':codon1_entropy,
                           'codon2_entropy': codon2_entropy, 'codon3_entropy':codon3_entropy}, index=[0])
        result.append(df)

    all_values = pd.concat(result)
    return all_values



def tree_slope_calculator(tree, mapping, feature, alias):

    value_2_tree_distance = []

    term_names = [term.name.split('.')[0] for term in tree.get_terminals()]
    values = [(name, mapping[mapping['refseq_id'] == name][feature].values[0]) for name in term_names]

    # calculate all pairs
    pairs =  list(combinations(values, 2))
    for pair in pairs:
        if pair[0][0] == None or pair[0][1] == None:
            print("we have None for feature {}".format(feature))
            continue
        distance = (tree.distance(pair[0][0], pair[1][0]))
        diff = (pair[0][1] - pair[1][1])**2

        value_2_tree_distance.append((diff, distance))

    # linear regression for the coefficients
    x = [np.log(val[1]) for val in value_2_tree_distance]
    y = [val[0] for val in value_2_tree_distance]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    df = pd.DataFrame({'feature':feature, 'family':alias, 'slope':slope, 'intercept':intercept,
                       'r_value':r_value, 'p_value':p_value}, index=[0])
    return df

def get_family_slopes_and_graphs(tree, mapping, features, alias, out):

    all_df = []
    for feature in features:
        df = tree_slope_calculator(tree, mapping, feature, alias)
        all_df.append(df)

    features_df = pd.concat(all_df)

    # create the plot
    x = list(range(100))
    colors = ['#E9C025', '#63AB2B', '#2BA5AB', '#9C59DC', '#E9ADE6', '#961232']
    for i, feature in enumerate(features):
        feature_info = features_df[features_df['feature'] == feature]
        slope = feature_info['slope'][0]    # no need in values because its only one line
        intercept = feature_info['intercept'][0]
        y = [slope * i + intercept for i in x]
        plt.plot(x, y, '--', label=feature, color=colors[i])

    plt.xlabel('Evolutionary time', fontsize = 16)
    plt.ylabel('Entropy value difference', fontsize = 16)
    plt.title('Entropy difference as a factor of evolutionary time for {} family'.format(alias), fontsize=22)
    plt.legend()
    plt.savefig(os.path.join(out, 'entropy_diff.pdf'), format='pdf', dpi=400, bbox_inches='tight')

    return features_df

def run_entropy_selection_test(super_folder, mapping):

    all_trees = []
    results = []

    features = ['entropy_5', 'codon1_entropy', 'codon2_entropy', 'codon3_entropy', 'rf_entropy']

    for root, dirs, files in tqdm(os.walk(super_folder)):
        tree = [f for f in files if 'phyml_tree' in f]
        if tree != []:
            all_trees.append(os.path.join(root, tree[0]))

    for t in tqdm(all_trees):
        if heritability.tree_2_string(t) == '':
            continue
        alias = os.path.basename(t).split('.')[0].strip()
        print(alias)
        tree = Phylo.read(t, 'newick')
        df = get_family_slopes_and_graphs(tree, mapping, features, alias, os.path.dirname(t))
        results.append(df)

    final = pd.concat(results)
    final.to_csv(os.path.join(super_folder, 'entropy_selection_test.csv'), index=False)

def main():
    cds = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/entropy/virus_host/virushostdb.cds.fna'
    try_cds = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/entropy/virus_host_entropy/virushostdb.cds.try.fna'
    genomics = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/entropy/virus_host/virushostdb.genomic.fna'
    try_genomics = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/entropy/virus_host_entropy/virushostdb.genomics.try.fna'

    out = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/entropy/virus_host_entropy'

    virus_host_entropy = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/entropy/virus_host_entropy/virus_host_entropy.csv'

    vh = pd.read_csv(virus_host_entropy)
    vh['refseq_id'] = vh['refseq_id'].astype(str).apply(lambda x: remove_punctuation(x))
    vh = vh[['refseq_id', 'virus_name', 'family']]

    basic_genome_entropy = genome_2_entropy(genomics, 5)
    mapping = refseq_2_cds(cds)
    cds_mapping = entropy_by_cds(mapping)

    dfs = [vh, basic_genome_entropy, cds_mapping]

    merged = reduce(lambda left, right: pd.merge(left, right, on=['refseq_id'],
                                                 how='outer'), dfs)
    merged.to_csv(os.path.join(out, 'entropy_by_cds.csv'), index=False)


if __name__ == '__main__':
    main()






