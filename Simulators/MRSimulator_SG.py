import pandas as pd
import numpy as np
import random
import argparse
from datetime import datetime
import math
from tqdm import tqdm


''' this is a simulator for mutation rate estimations. using a known number of bases sequenced by coverage of 
each position create a simulated mutation process and see which mutation rate estimate is the closest for the known
MLE (k\n) were k is the number of mutational events and n is the number of total bases sequenced.'''


class Simulator(object):

    def __init__(self, N, numSim, mue, errorRate=10**-4):
        self.N = N
        self.numSim = numSim
        self.mue = mue
        self.errorRate = errorRate

    def get_num_sim(self):
        return self.numSim

    def get_error_rate(self):
        return self.errorRate

    def set_mue(self, mue):
        self.mue = mue

    def set_errorRate(self, err):
        self.errorRate = err

    def create_null_genome(self):
        return [0] * self.N

    def simulate(self):

        simulations = np.zeros(shape=(self.numSim, self.N)) # create a matrix with each line corresponds to a specific simulation

        # add 1's according to mutation rate threshold
        for sim in tqdm(range(self.numSim)):
            for position in tqdm(range(self.N)):
                sampled = random.random()
                if sampled >= self.mue:
                    simulations[sim, position] = 1

        # take this matrix and try again, this time with the error rate
        for sim in range(self.numSim):
            for position in range(self.N):
                sampled = random.random()
                if sampled >= self.errorRate:
                    simulations[sim, position] += 1

        return simulations

    def get_expected_mue(self, sim):
        n_ones = list(sim).count(1)
        return n_ones/self.N



def get_num_successes_from_matrix(df):

    # total bases sequenced = for each position multiply coverage by 4 (corresponds to 4 nucle.)
    bases_sequenced = df.drop_duplicates('Pos')['Read_count'].sum()
    mutations = df[(df['Ref'] != df['Base']) & (df['Freq'] != 0)].shape[0]

    return mutations / bases_sequenced


def main(args):

    # read raw frequency file, remove insertions and deletions - currently we model only ts-tv
    df = pd.read_csv(args.freq, sep='\t')
    df = df[(df['Pos'].astype(int) == df['Pos']) & (df['Base'] != '-')]
    df['Pos'] = df['Pos'].astype(int)

    N = df.drop_duplicates('Pos')['Read_count'].sum()
    estimate = get_num_successes_from_matrix(df)


    n_sim = args.num_simulations
    delta = []

    mutation_rates = [10**-4, 5*10**-5, 10**-5, 5*10**-5, 10**-6]

    with open(args.out, 'w') as o:
        o.write('{}\n'.format(datetime.now()))
        o.write('Simulations result for input file : {}\n Running {} simulations in total\n '.format(args.freq, n_sim))
        o.write('-------------------------------------------\n\n')
        o.write('number of bases sequenced = {}\nMutation rate estimator k/N = {}\n'
                .format(N, estimate))
        for mr in mutation_rates:
            estimators = []
            # create an object and run simulations
            simulator = Simulator(N=N, numSim=n_sim, mue=mr)
            simulated_mat = simulator.simulate()
            for line in simulated_mat:
                estimators.append(simulator.get_expected_mue(line))

            o.write('Mutation rate = {}\nMedian simulation estimator = {}\nMean simulation estimator = {}\n'
                    'Max simulation estimetor = {}\nMin simulation estimator = {}\n'.format(mr, np.median(estimators),
                                                                                          np.mean(estimators),
                                                                                          np.max(estimators),
                                                                                         np.min(estimators)))
            dist = math.fabs(np.mean(estimators) - estimate)
            delta.append((dist, mr))
            o.write('Distance from estimate = {}\n\n'.format(dist))

        best_mr = min(delta, key=lambda x: x[0])[1]
        o.write('Best mutation rate estimator = {}\n'.format(best_mr))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--freq", type=str,
                        help="a path to a raw input .freqs file", required=True)
    parser.add_argument("-s", "--num_simulations", type=int,
                        help="the number of simulations that will be performed", required=True)
    parser.add_argument("-o", "--out", type=str,
                        help="output file path", required=True)

    args = parser.parse_args()

    main(args)



