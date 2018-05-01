import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob



def unite_all_freq_files(freqs_dir, out=None):
    """
    unites all frequency file into one file. add the following:
    Replica, Degree, Sample, Time
    :param freqs_dir: a directory with frequency files in a format of pTIME-DEGREEREPLICA.freqs
    :param out: output directory to save results
    :return: a merged data frame
    """
    freqs_df = []

    # read all freq files and add the sample name to the data frame
    for f in glob.glob('*.freqs'):
        curr_df = pd.read_csv(f)
        sample = os.path.basename(f).split('.')[0]
        curr_df['Sample'] = sample
        freqs_df.append(curr_df)

    df = pd.concat(freqs_df)

    df['Time'] = df['Sample'].apply(lambda x: x.split('-')[0][1:])
    df['Degree'] = df['Sample'].apply(lambda x: x.split('-')[1][0:2])
    df['Replica'] = df['Sample'].apply(lambda x: x.split('-')[1][-1])

    if out != None:
        df.to_csv(os.path.join(out, 'all_freqs.csv'))

    return df



def filter_all_mutations(df, pos_2_remove=None):
    """
    This method filter all mutations out of a frequency file. the following will be removed:
        * insertions
        * deletions
        * positions which are not reliable (primer scopes or other problematic positions)
        * base == ref positions
    :param df: a data frame of a freq file
    :param pos_2_remove: a list of positions to remove from the data frame. default value of None
    :return: a data frame of all mutation
    """

    # remove insertions and deletions
    df['Pos'] = df[df['Pos'].astype(int) == df['Pos']]
    df = df[df.Base != '-']

    # remove problematic positions if needed
    if pos_2_remove != None:
        df = df[~df['Pos'].isin(pos_2_remove)]

    # add Mutation column
    df['Mutation'] = df.Ref + df.Base

    # filter mutation only
    df = df[~df['Mutation'].isin(['AA','CC','GG','TT'])]

    return df


def add_mutation_type(mutations_df, reference):
    """
    add the mutation type for each mutation in the data
    :return: a data frame with Type column
    """

    pass