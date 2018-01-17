import pandas as pd
from scipy import stats
import os
import re
from tqdm import tqdm
import numpy as np
import fileinput
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random



''' Calculate Entropy for each sequence in each virus type, use the sequence as the distribution of the nucleotides'''


db_path = r'/Volumes/STERNADILABHOME$/volume1/shared/data/Viral-distributions/Data/Nucleotide'
#db_path = r'/Volumes/STERNADILABHOME$/volume1/shared/data/Viral-distributions/Data/cds'
out = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/entropy/plots'

def main():



    basic= False
    kmers = False
    sliding_window = True

    all_files_used = []
    if basic:

        df = fasta2entropy(db_path,all_files_used, complete_genome=True)
        df.to_csv(r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/entropy/complete_genome_kmers_scramble.csv',
                  index=False)

    if kmers:
        # kmers analysis
        df = kmers_analysis(db_path, all_files_used, complete_genome=True)
        df.to_csv(r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/entropy/complete_genome_kmers.csv',
                  index=False)

    if sliding_window:
        sliding_window_entropy(db_path, out)


def fasta2entropy(db, all_files_used, complete_genome=False):
    """
    update vectors to create a data frame of entropy calculation for each sequence in the db
    :param db: a path to a virus db
    :param complete_genome: a flag for taking complete genomes only. default False
    :return: update the vectors in place
    """

    fasta_id = []
    virus = []
    virus_id = []
    protein = []
    entropy = []
    len_seq = []


    for root, dirs, files in tqdm(os.walk(db)):
        if files == []: # continue if there are no files
            continue
        # we want to take only protein folders
        if (('selection' in root) or ('FD' in root) or ('.' not in root.split('/')[-1])) and (complete_genome == False):
            continue
        if (('selection' in root) or ('FD' in root)) and (complete_genome == True):
            continue
        # found in the nucleotide db, ignore this level.
        if ('.DS_Store' in files) or ('Thumbs.db' in files):
            continue

        # else - get the fasta file and sequence information
        fasta = [f for f in files if f.endswith('.fasta')]
        assert len(fasta) == 1
        sequences_path = os.path.join(root, fasta[0])
        sequences = re.split(">", open(sequences_path, "r").read().replace('\n', ''))[1:]
        cur_protein = root.split('/')[-1]
        for seq in tqdm(sequences):
            splitted = seq.split('.')
            identifier = get_full_fasta_id(splitted[:-1])
            data = splitted[-1].replace('-','') # remove gaps from sequence

            # scramble
            data = scrambler(data)

            # add randomization
            #data = random_genome(len(data))
            if complete_genome:
                if 'complete genome' in identifier:
                    virus_info = root.split('/')[-1].split('_')
                    virus_id.append(virus_info[0])
                    virus.append(virus_info[1])
                    fasta_id.append(identifier)
                    entropy.append(entropy_by_kmer(data, 6))
                    len_seq.append(len(data))


            else:
                virus_info = root.split('/')[-2].split('_')
                virus_id.append(virus_info[0])
                virus.append(virus_info[1])
                fasta_id.append(identifier)
                protein.append(cur_protein)
                entropy.append(ent(data))

    df = pd.DataFrame({'virus_name': virus, 'virus_index': virus_id, 'fasta': fasta_id,
                       'size': len_seq, 'entropy': entropy})

    return df




def get_full_fasta_id(parts):
    """
    get the fasta id
    :param parts: parts of the id after splitting by '.'
    :return: the full fasta id
    """
    res = parts[0]
    for part in parts[1:]:
        res = res + '.' + part
    return res



def ent(sequence):
    """
    calculate the shannon entropy for a given sequence
    :param sequence: a string
    :return: the entropy
    """
    data = pd.DataFrame({'seq':list(sequence)})
    p_data = data['seq'].value_counts()/data.shape[0] # calculates the probabilities
    entropy = stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy



def concat_txt_files(filenames, output):
    """
    concatenate all files to one
    :param filenames: a list of filenames
    :param output: output directory path
    :return:
    """
    print('merging {} files into one file\n'.format(len(filenames)))
    with open(output, 'w') as fout, fileinput.input(filenames) as fin:
        for line in fin:
            fout.write(line)

    print('Done mergeing files!')



def fasta2entropy_nucleotides(db, fasta_id, virus_id, virus, entropy):
    flag = False
    for root, dirs, files in os.walk(db):
        if not flag:
            flag = True
            continue
        print(files)
        fasta = [f for f in files if f.endswith('.fasta')]
        assert len(fasta) == 1
        sequences_path = os.path.join(root, fasta[0])

        sequences = re.split(">", open(sequences_path, "r").read().replace('\n', ''))[1:]
        for seq in sequences:
            splitted = seq.split('.')
            identifier = str(splitted[0]+'.'+splitted[1])
            data = splitted[-1]


            # get only complete genomes
            #if 'complete genome' in identifier:
            virus_info = fasta[0].split('/')[-1].split('_')
            virus_id.append(virus_info[0])
            virus.append(virus_info[1])
            fasta_id.append(identifier)
            entropy.append(ent(data))



def kmers_analysis(db, all_fasta, w=8, complete_genome=False):
    ''' long running time '''

    all_seq_df = []

    for root, dirs, files in tqdm(os.walk(db)):
        if files == []: # continue if there are no files
            continue
        # we want to take only protein folders
        if (('selection' in root) or ('FD' in root) or ('.' not in root.split('/')[-1])) and (complete_genome == False):
            continue
        if (('selection' in root) or ('FD' in root)) and (complete_genome == True):
            continue
        # found in the nucleotide db, ignore this level.
        if ('.DS_Store' in files) or ('Thumbs.db' in files):
            continue

        # else - get the fasta file and sequence information
        fasta = [f for f in files if f.endswith('.fasta')]
        assert len(fasta) == 1
        sequences_path = os.path.join(root, fasta[0])
        all_fasta.append(sequences_path)
        sequences = re.split(">", open(sequences_path, "r").read().replace('\n', ''))[1:]
        for seq in tqdm(sequences):
            splitted = seq.split('.')
            identifier = get_full_fasta_id(splitted[:-1])
            data = splitted[-1]

            if complete_genome:
                if 'complete genome' in identifier:
                    # create a mapping per sequence
                    mapping = {}
                    for i in tqdm(range(len(data) - w)):
                        k_mer = data[i:i + w]
                        if k_mer in mapping:
                            mapping[k_mer] += 1
                        else:
                            mapping[k_mer] = 1

                    kmers = []
                    count = []
                    entr = []
                    for kmer in tqdm(mapping):
                        kmers.append(kmer)
                        count.append(mapping[kmer])
                        entr.append(ent(kmer))
                    n = len(mapping.keys())
                    pr = [c / n for c in count]
                    virus_info = root.split('/')[-1].split('_')
                    v_id = [virus_info[0] for x in range(len(pr))]
                    virus_family = [virus_info[1] for x in range(len(pr))]
                    result = pd.DataFrame({'kmer': kmers, 'pr_of_genome': pr, 'entropy': entr, 'virus':virus_family,
                                           'virus_index':v_id})
                    all_seq_df.append(result)

            else:
                print("not complete genome!")

    concated = pd.concat(all_seq_df)

    return concated


def random_genome(n):
    string_val = "".join(np.random.choice(['A', 'C', 'T', 'G'], size=n))
    return string_val




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
        # random
        #p = 1 / total_kmers
        entropy += -(p * math.log2(p))

    return entropy


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



def sliding_window_entropy(db_path, out, w=200, complete_genome=True):

    virus_2_sequences = {}
    for root, dirs, files in tqdm(os.walk(db_path)):
        if files == []: # continue if there are no files
            continue
        # we want to take only protein folders
        if (('selection' in root) or ('FD' in root) or ('.' not in root.split('/')[-1])) and (complete_genome == False):
            continue
        if (('selection' in root) or ('FD' in root)) and (complete_genome == True):
            continue
        # found in the nucleotide db, ignore this level.
        if ('.DS_Store' in files) or ('Thumbs.db' in files):
            continue

        # else - get the fasta file and sequence information
        fasta = [f for f in files if f.endswith('.fasta')]
        assert len(fasta) == 1
        sequences_path = os.path.join(root, fasta[0])
        sequences = re.split(">", open(sequences_path, "r").read().replace('\n', ''))[1:]

        for i, seq in tqdm(enumerate(sequences)):

            splitted = seq.split('.')
            identifier = get_full_fasta_id(splitted[:-1])
            data = splitted[-1].replace('-','') # remove gaps from sequence


            if complete_genome:
                if 'complete genome' in identifier:
                    virus_info = root.split('/')[-1].split('_')
                    virus = virus_info[1]

                    # create a dictionary of sequences for each virus
                    if virus in virus_2_sequences.keys():
                        virus_2_sequences[virus].append(data)

                    else:
                        virus_2_sequences[virus] = []
                        virus_2_sequences[virus].append(data)

    # for each virus plot sliding window entropy
    for v in tqdm(virus_2_sequences.keys()):
        print(v)
        n = len(virus_2_sequences[v])
        data_seq = virus_2_sequences[v]
        colors = plt.cm.Paired(np.linspace(0, 1, n+2))

        for j, seq in tqdm(enumerate(data_seq)):
            window_entropy = []
            for i in range(len(seq) - w):
                window_entropy.append(entropy_by_kmer(seq[i:i + w], 6))
            plt.plot(window_entropy, label='seq {}'.format(j), color=colors[j])

        plt.title('{} sliding window of {}'.format(v, w))
        plt.savefig(os.path.join(out, '{}_sliding_window_{}'.format(v, w)), bbox_inches='tight', dpi=400)
        #plt.show()

    print('Done!')


if __name__ == '__main__':
    main()

