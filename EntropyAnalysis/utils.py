import pandas as pd
import numpy as np
import os
from collections import Counter
from tqdm import tqdm
import re
from Bio import SeqIO
import math
from Bio.Seq import Seq
import random


def refseq_2_seq(refseqs, fasta):
    """
    takes a list of refseq id's and returns the corresponding sequences
    :param refseqs: a list of refseq id's
    :param fasta: a path to a fasta file containing all genomes
    :return: a dictionary of refseq id as a key and its corresponding sequence
    """

    id_2_seq = {}

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
        identifier = splitted[0].split()[0]
        genome = splitted[-1]

        if identifier in refseqs:
            if identifier in id_2_seq:
                print('Refseq id is not unique!')

            id_2_seq[identifier] = genome

    return id_2_seq

def extract_fasta_seqs(in_fasta, out_fasta, ids):
    """
    parses a fasta file to contain only the sequences in ids
    :param in_fasta: input fasta file
    :param out_fasta: output fasta path
    :param ids: a list of identifications
    :return: saves the new_fasta file
    """

    fin = open(in_fasta, 'r')
    fout = open(out_fasta, 'w')

    for record in SeqIO.parse(fin, 'fasta'):
        for item in ids:
            if item.strip() == record.id
                fout.write(">" + record.id + "\n")
                fout.write(record.seq + "\n")

    fin.close()
    fout.close()


def entropy_by_kmer(seq, k):
    """
    calculate the entropy of a sequence according to its k-mers
    :param seq: a genome string
    :param k: tje size of the k-mer to use
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


def reading_frame_entropy(seq):
    """
    get entropy by reading frame
    :param seq: a genome sequence
    :param k: kmer length
    :return: entropy value by reading frame
    """

    kmers = {}
    for i in range(0,len(seq) - 3,3):
        kmer = seq[i:i+3]
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



def get_reverse_complement(seq):
    """
    get reverse complement genome
    :param seq: a genome sequence
    :return: a string of reverse complement
    """
    seq = Seq(seq)
    reverse_complement = seq.reverse_complement()
    return reverse_complement


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



def entropy_2_refseq(db_path, genomic, cds, out, k):
    """
    calculate for each complete genome its entropy, joint entropy with reverse complement and information storage
    :param db_path: a path to a db with viruses information
    :param genomic: a genomic fasta file
    :param cds: coding regions fasta file
    :param out: output path to save the files
    :param k: kmer length
    :return: a data frame with entropy and virus information
    """

    refseq_g = []
    entropy_g = []
    joint_g = []
    information_storage_g = []
    size_g = []
    scrambled_joint_g = []

    refseq_p = []
    frame_entropy_p = []
    entropy_p = []
    protein_p = []
    joint_p = []
    information_storage_p = []
    size_p = []
    scrambled_joint_p = []



    genomic_sequences = re.split(">", open(genomic, "r").read().replace('\n', ''))[1:]
    for seq in tqdm(genomic_sequences):
        if '.' not in seq:
            print('no dot in sequence name\n')
            continue
        if 'complete genome' not in seq:
            #print('not complete genome\n')
            continue

        # get identifier and genomic sequence
        splitted = seq.split('.')
        identifier = splitted[0].split()[0]
        genome = splitted[-1]

        rc_genome = get_reverse_complement(genome)
        rc_genome_scrambled = scrambler(rc_genome)

        joint = joint_entropy(genome, rc_genome, k)
        info_storage = information_storage(genome, rc_genome, k)
        entropy = entropy_by_kmer(genome, k)
        joint_scrambled = joint_entropy(genome, rc_genome_scrambled, k)

        refseq_g.append(identifier)
        entropy_g.append(entropy)
        joint_g.append(joint)
        information_storage_g.append(info_storage)
        size_g.append(len(genome))
        scrambled_joint_g.append(joint_scrambled)

    cds_sequences = re.split(">", open(cds, "r").read().replace('\n', ''))[1:]
    for seq in tqdm(cds_sequences):

        # get identifier and genomic sequence
        splitted = seq.split('|')
        identifier = splitted[0]
        assert(splitted[2] == '')
        protein = splitted[3]
        genome = splitted[-1]

        rc_genome = get_reverse_complement(genome)
        rc_genome_scrambled = scrambler(rc_genome)

        joint = joint_entropy(genome, rc_genome, k)
        info_storage = information_storage(genome, rc_genome, k)
        entropy = entropy_by_kmer(genome, k)
        rf_entropy = reading_frame_entropy(genome)
        joint_scrambled = joint_entropy(genome, rc_genome_scrambled, k)

        refseq_p.append(identifier)
        entropy_p.append(entropy)
        joint_p.append(joint)
        information_storage_p.append(info_storage)
        protein_p.append(protein)
        frame_entropy_p.append(rf_entropy)
        size_p.append(len(genome))
        scrambled_joint_p.append(joint_scrambled)



    # create a data frame for each cds\ genomic data and merge
    genomic_df = pd.DataFrame({'refseq_id': refseq_g, 'entropy_genome':entropy_g, 'genome_size':size_g,
                               'joint_entropy_genome':joint_g, 'information_storage_genome':information_storage_g,
                               'scrambled_joint_genpme': scrambled_joint_p})

    cds_df = pd.DataFrame({'refseq_id': refseq_p, 'protein_id':protein_p, 'entropy_cds': entropy_p, 'protein_size': size_p,
                               'joint_entropy_cds': joint_p, 'information_storage_cds': information_storage_p,
                           'rf_entropy':frame_entropy_p,'scrambled_joint_cds':scrambled_joint_p})

    merged = pd.merge(genomic_df, cds_df, on='refseq_id')

    merged.to_csv(os.path.join(out, 'entropy_measurements_{}.csv'.format(k)), index=False)

    return merged




