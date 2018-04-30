import numpy as np
from itertools import permutations
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm
import argparse


THRESHOLD = 10**-3
BASE_FITNESS = 2

class ChtrModel(object):

    def __init__(self, N, n1, G, r, B, MOI=1):
        self.N = N
        self.n1 = n1
        self.genome = G
        self.r = r
        self.b = B
        self.moi = MOI
        self.cheater = []
        self.wt = []
        self.ecoli = []
        self.time = []


    def resistantN(self):
        return self.N * (1- self.r)



def sum_2_k_pairs(k):
    #generats a list with pirs (i, j) which sum to k for all 0<=q<=k

    options = list(range(k+1))
    all_pairs = list(permutations(options, 2)) + [(x,x) for x in options]
    pairs = [pair for pair in all_pairs if sum(pair) in options]

    return pairs


class Cycle(object):

    def __init__(self, N, n1, n2, moi, k):
        self.N = N
        self.n1 = n1
        self.n2 = n2
        self.moi = moi
        self.k = k

    def setN(self, N):
        self.N = N

    def set_n1(self, n1):
        self.n1 = n1

    def set_n2(self, n2):
        self.n2 = n2

    def set_moi(self):
        self.moi = (self.n1 + self.n2) / self.N
        #self.moi = (self.n1) / self.N


    def poisson_prob(self,k1,k2):
        return ((self.n1/self.N)**k1)*((self.n2/self.N)**k2)*(math.exp(-(self.n1+self.n2)/self.N))/\
               (math.factorial(k1)*math.factorial(k2))

    def infection_proba(self, k, b):

        # init the fraction n2
        n2 = 0
        pairs = sum_2_k_pairs(k)
        P = np.zeros(shape=(k+1, k+1))

        for pair in pairs:
            i = pair[0]
            j = pair[1]

            # add the value to the matrix
            P[i, j] = self.poisson_prob(i, j)

            # co-infection, i wt's j cheaters, 3.5 arbitrarily chosen to limit cheater fitness when there are much less wt
            if i != 0 and j != 0 :
                n2 += self.poisson_prob(i, j) * BASE_FITNESS  * max(1, (j+1/i)) * b * j #we have j cheaters, not one,
                # each has a burst size b
            # only cheater infection - there are cheaters coming out
            if i == 0 and j != 0:
                n2 += self.poisson_prob(i, j) * BASE_FITNESS * j
        #print("only cheater")
        #print(sum([P[i,j] for i in range(k+1) for j in range(k+1) if i==0 and j!=0]))

        return n2 * self.N




    def simulate_cycle(self, model, passage):

        # init first round
        self.setN(model.N * (1 - model.r))
        self.set_n1(model.n1)

        if model.cheater == []:
            model.cheater.append(self.n2)   # first iteration' self.n2 is defined
        else:
            dilution_factor = model.n1 / model.wt[-1]   # in each passage we have a dilution factor.
            if dilution_factor == 1:
                dilution_factor = 0.001
            self.set_n2(model.cheater[-1] * dilution_factor + self.n1 *10**-6)
            model.cheater.append(self.n2)

        model.time.append(passage)
        model.ecoli.append(self.N)
        model.wt.append(self.n1)


        self.set_moi()

        print(self.N, self.n1, self.n2, self.moi)

        while self.N >= THRESHOLD:
            #update cheater, wt, ecoli and moi
            updatedN = (1 - model.r) * self.N * math.exp(-self.moi) * (2 ** 6)
            if updatedN < THRESHOLD:
                break
            updated_n2 = self.infection_proba(self.k, model.b)
            # we have a wt population which do not infect. the breast size is for particles that infected an ecoli.
            updated_n1 = self.n1 * (1-math.exp(-self.n1/self.N)) * model.b + self.n1 * math.exp(-self.n1/self.N)

            self.set_n2(updated_n2)
            self.setN(updatedN)
            self.set_n1(updated_n1)
            self.set_moi()

            # update models progression
            model.ecoli.append(self.N)
            model.wt.append(self.n1)
            model.cheater.append(self.n2)
            model.time.append(passage)


        #
        # print("before while - N is {}".format(self.N))
        # updatedN = self.N
        # while updatedN >= THRESHOLD:
        #     print("updatedN {}".format(updatedN))
        #     updatedN = (1 - model.r) * self.N * math.exp(-self.moi) * (2**6)
        #     updated_n2 = self.infection_proba(self.k)
        #     print("self.N {}".format(self.N))
        #     self.setN(updatedN)
        #     print("self.N {}".format(self.N))
        #     self.set_n1(self.n1 * model.b)
        #     self.set_n2(updated_n2)
        #     print("moi now is : {}, n1={} N={}".format(self.moi, self.n1, self.N))
        #     self.set_moi()
        #
        #     # update model
        #     model.ecoli.append(self.N)
        #     model.wt.append(self.n1)
        #     model.cheater.append(self.n2)
        #     model.time.append(passage)
        # print("after while - N is {}".format(self.N))
        # # remove the last almost zero result from calculation
        # model.ecoli.pop()
        # model.wt.pop()
        # model.cheater.pop()
        # model.time.pop()



def main():

    out = r'/Users/daniellemiller/Google Drive/Lab/Analysis/cheaters model/mdl_passage_cycle.csv'
    #create the model
    mdl = ChtrModel(N=10**9, n1=10**9, G=3569, r=0.3, B=1000, MOI=1)
    c = Cycle(N=mdl.N, n1=mdl.n1, n2=1000, moi=mdl.moi, k=10)

    num_passages = 20
    for p in tqdm(range(1, num_passages + 1)):
        print("passage is : {}".format(p))
        c.simulate_cycle(mdl, p)



    df = pd.DataFrame({'Passage':mdl.time, 'N': mdl.ecoli, 'n1':mdl.wt, 'n2':mdl.cheater})
    df['f1'] = df.apply(lambda row: row['n1'] / (row['n1'] + row['n2']), axis=1)
    df['f2'] = df.apply(lambda row: row['n2'] / (row['n1'] + row['n2']), axis=1)
    df.to_csv(out, index=False)

    sns.pointplot(x='Passage', y='f2', data=df.drop_duplicates('Passage'), color='#D49749')
    plt.ylim(0,1)
    plt.savefig(r'/Users/daniellemiller/Google Drive/Lab/Analysis/cheaters model/linear_model_r30_no_max.png')
    plt.show()


if __name__ == '__main__':
    main()





