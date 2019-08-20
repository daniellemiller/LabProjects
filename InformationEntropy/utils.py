import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
from Bio import SeqIO, Phylo
import math
from Bio.Seq import Seq
import random
import scipy.stats as stats
import statsmodels.stats.multitest as multi
import RNA
from random import sample
import glob
from functools import reduce

# k-mers distribution based entropy measures.
def entropy_by_kmer(seq, k):
    """
    calculate the entropy of a sequence according to its k-mers
    :param seq: a genome string
    :param k: size of the k-mer to use
    :return: entropy
    """

    # update kmers
    kmers = {}
    for i in range(len(seq) - k):
        kmer = seq[i:i+k]
        if kmer in kmers:
            kmers[kmer] += 1
        else:
            kmers[kmer] = 1

    # calculate entropy
    total_kmers = sum(kmers.values())
    entropy = 0
    for kmer in kmers:
        p = kmers[kmer] / total_kmers
        entropy += -(p * math.log2(p))

    return entropy

def joint_entropy (seq1, seq2, k):
    """
    calculates the joint entropy of two sequences.
    :param seq1: sequence #1
    :param seq2: sequence #2
    :param k: k-mer length
    :return: joint entropy value
    """

    kmers_1 = {}
    kmers_2 = {}

    # kmers in sequence #1
    for i in range(len(seq1) - k):
        kmer = seq1[i:i+k]
        if kmer in kmers_1:
            kmers_1[kmer] += 1
        else:
            kmers_1[kmer] = 1

    for i in range(len(seq2) - k):
        kmer = seq2[i:i+k]
        if kmer in kmers_2:
            kmers_2[kmer] += 1
        else:
            kmers_2[kmer] = 1

    # calculate joint entropy
    total_kmers_1 = sum(kmers_1.values())
    total_kmers_2 = sum(kmers_2.values())

    total = total_kmers_1 + total_kmers_2

    # compare the kmers space to be equal at both
    for kmer in kmers_1:
        if kmer not in kmers_2:
            kmers_2[kmer] = 0

    for kmer in kmers_2:
        if kmer not in kmers_1:
            kmers_2[kmer] = 0

    joint_entropy = 0
    for kmer1 in kmers_1:
        for kmer2 in kmers_2:
            p_xy = (kmers_1[kmer1] + kmers_2[kmer2]) / total

            joint_entropy += -(p_xy * math.log2(p_xy))

    return joint_entropy


def information_storage (seq1, seq2, k):
    """
    calculates the information storage of two sequences.
    :param seq1: sequence #1
    :param seq2: sequence #2
    :param k: k-mer length
    :return: information storage value
    """

    kmers_1 = {}
    kmers_2 = {}

    # kmers in sequence #1
    for i in range(len(seq1) - k):
        kmer = seq1[i:i+k]
        if kmer in kmers_1:
            kmers_1[kmer] += 1
        else:
            kmers_1[kmer] = 1

    for i in range(len(seq2) - k):
        kmer = seq2[i:i+k]
        if kmer in kmers_2:
            kmers_2[kmer] += 1
        else:
            kmers_2[kmer] = 1

    # calculate joint entropy
    total_kmers_1 = sum(kmers_1.values())
    total_kmers_2 = sum(kmers_2.values())

    total = total_kmers_1 + total_kmers_2

    # compare the kmers space to be equal at both
    for kmer in kmers_1:
        if kmer not in kmers_2:
            kmers_2[kmer] = 0

    for kmer in kmers_2:
        if kmer not in kmers_1:
            kmers_2[kmer] = 0

    inf_storage = 0
    for kmer1 in kmers_1:
        for kmer2 in kmers_2:

            p_xy = (kmers_1[kmer1] + kmers_2[kmer2]) / total
            p_x = kmers_1[kmer1] / total_kmers_1
            p_y = kmers_2[kmer2] / total_kmers_2

            if p_x == 0 or p_y == 0:
                continue
            inf_storage += p_xy * math.log2(p_xy/(p_x*p_y))

    return inf_storage



def get_reverse_complement(seq):
    """
    get reverse complement genome
    :param seq: a genome sequence
    :return: a string of reverse complement
    """
    seq = Seq(seq)
    reverse_complement = seq.reverse_complement()
    return reverse_complement



# shuffle scrambler for a sequence.
def _scrambler(word):
    word_to_scramble = list(word)
    random.shuffle(word_to_scramble)
    new_word = ''.join(word_to_scramble)
    return new_word

def scrambler(word):
   new_word = _scrambler(word)
   while new_word == word and len(word) > 1:
       new_word = _scrambler(word)
   return new_word




######## entropy profiles calculation ##################
def get_joint_entropy_profile(fasta, w, k=5, out=None, data_type='fasta'):
    """
    sliding window entropy profile of all sequences in a family
    :param fasta: a fasta file contatining viral sequences
    :param w: the window size
    :param out: optional. if != None a profile will be saved as a png
    :return: the vector of profile entropy
    """
    all_entropies = {}
    alias = os.path.basename(fasta).split('.')[0]

    i = 0
    for rec in tqdm(SeqIO.parse(fasta, data_type)):
        entropies = []
        # get identifier and genomic sequence

        genome = str(rec.seq)

        for j in range(len(genome) - w):
            sub_genome = genome[j:j+w]
            rc_sub_genome = str(get_reverse_complement(sub_genome))
            entropy = joint_entropy(sub_genome, rc_sub_genome, k)
            entropies.append(entropy)


        all_entropies['seq_{}'.format(i)] = entropies
        i += 1

    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in all_entropies.items()]))
    if out != None:
        df.to_csv(os.path.join(out, '{}_Joint_profile.csv'.format(alias)), index=False)

    return df



def get_entropy_profile(fasta, w, k=5, out=None, data_type='fasta'):
    """
    sliding window entropy profile of all sequences in a family
    :param fasta: a fasta file contatining viral sequences
    :param w: the window size
    :param out: optional. if != None a profile will be saved as a png
    :return: the vector of profile entropy
    """
    all_entropies = {}
    alias = os.path.basename(fasta).split('.')[0]

    i = 0
    for rec in tqdm(SeqIO.parse(fasta, data_type)):
        entropies = []
        # get identifier and genomic sequence

        genome = str(rec.seq)

        for j in range(len(genome) - w):
            sub_genome = genome[j:j+w]
            entropy = entropy_by_kmer(sub_genome, k)
            entropies.append(entropy)



        all_entropies['seq_{}'.format(i)] = entropies
        i += 1

    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in all_entropies.items()]))

    if out != None:
        df.to_csv(os.path.join(out, '{}_Shannon_profile.csv'.format(alias)), index=False)

    return df

def deltaG_calculator(seq):
    """
    calculate the minimum free energy (G) for a given sequence
    :param seq: an rna sequence
    :return: minimum free energy
    """
    ss, mfe = RNA.fold(seq)
    return mfe

def get_deltaG_profile(fasta, w, out=None, data_type='fasta'):

    """
    sliding window free energy profile of all sequences in a family
    :param fasta: a fasta file contatining viral sequences
    :param w: the window size
    :param out: output file
    :return: the vector of profile entropy
    """
    all_deltaG = {}
    alias = os.path.basename(fasta).split('.')[0]

    i = 0
    for rec in tqdm(SeqIO.parse(fasta, data_type)):
        values = []

        genome = str(rec.seq)

        for j in range(len(genome) - w):
            sub_genome = genome[j:j+w]
            mfe = deltaG_calculator(sub_genome)
            values.append(mfe)


        all_deltaG['seq_{}'.format(i)] = values
        i += 1

    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in all_deltaG.items()]))
    if out != None:
        df.to_csv(os.path.join(out, '{}_deltaG_profile.csv'.format(alias)), index=False)

    return df
