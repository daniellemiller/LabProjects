'''

Mutation Rate Simulator

using ABC approach-
    * use a prior distribution of mutation rates - uniform distribution on interval [a,b]
    * estimate the posterior distribution using 1% of the simulations, according to distance measurement for the
    observed estimate
    * draw the posterior distribution, and extract the median value to be the mutation rate
'''

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, sparse
import os


class Simulator(object):

    def __init__(self, n, low, high, errorate, numSim=1000):
        self.n = n
        self.low = low
        self.high = high
        self.errorate = errorate
        self.numSim = numSim

    def simulate(self, estimate, output):

        # create mutation rate prior
        rates = np.array(np.random.uniform(low=self.low, high=self.high, size=self.numSim))
        # create a sparse matrix for mutation rate and for error rate and sum them
        mr_matrix = sparse.csc_matrix([stats.bernoulli.rvs(rate, size=self.n)
                                       for rate in rates])
        er_matrix = sparse.csc_matrix([stats.bernoulli.rvs(self.errorate, size=self.n) for i in range(self.numSim)])
        print(mr_matrix.shape, er_matrix.shape)
        summed = mr_matrix + er_matrix
        estimators = [int((summed[i,:] == 1).sum(1)) / self.n for i in range(self.numSim)]    # get the sum of 1's in a row

        df = pd.DataFrame({'MutationRate':rates, 'Estimators': estimators})
        df['Distance'] = df['Estimators'].apply(lambda x: (x - estimate)**2)
        df.to_csv(output, index=False)




def main(args):

    # read raw frequency file, remove insertions and deletions - currently we model only ts-tv
    df = pd.read_csv(args.freq, sep='\t')
    df = df[(df['Pos'].astype(int) == df['Pos']) & (df['Base'] != '-')]
    df['Pos'] = df['Pos'].astype(int)

    N = df.drop_duplicates('Pos')['Read_count'].sum()
    k = df[(df['Ref'] != df['Base']) & (df['Freq'] != 0)].shape[0]
    estimate = k/N

    low = args.low
    high =args.high

    if low > high:
        raise Exception("Low bound is higher then higher bound")

    out = args.out
    sim = Simulator(n = N, low=low, high=high, errorate=4.4e-4, numSim=100)
    sim.simulate(estimate, out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--freq", type=str,
                        help="a path to a raw input .freqs file", required=True)
    parser.add_argument("-l", "--low", type=float,
                        help="low bound mutation rate", default=10 ** -6)
    parser.add_argument("-u", "--high", type=float,
                        help="high bound mutation rate", default=10 ** -4)
    parser.add_argument("-o", "--out", type=str,
                        help="output file path", required=False)

    args = parser.parse_args()

    main(args)