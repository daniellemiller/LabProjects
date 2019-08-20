import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import statsmodels.stats.multitest as multi
from Bio import SeqIO
import os
from math import ceil
from typing import AnyStr
from utils import joint_entropy, entropy_by_kmer, get_reverse_complement
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
import networkx as nx
import glob, re




def get_critical_points(profile):
    """
    this method calculates the gradient of each one of the points in profile and returns a list of tuples containing all
    critical points (x,y)
    :param profile: a vector of numbers
    :return: a vector of the gradients of each data point
    """

    gradients = np.gradient(profile)
    critical_pts = [(x+1, profile[x]) for x, y in enumerate(gradients) if y ==0]    # add 1 to x to fit the relevant position (indices starts from 0!
    return critical_pts



def codon_scrambler(seq):
    """
    scramble the codons in seq. this method assumes that the given seq is of coding region and contains triplets.
    :param seq: a cds sequence
    :return: the scrambled codon sequence
    """

    if seq %3 != 0:
        print('seq should be a multiple of 3!')
        return

    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    random.shuffle(codons)
    return ''.join(codons)


def stretchFinder(profile, l, m=10**4):
    """
    implementation of strechFinder as described in : "Synonymous site conservation in the HIV-1 genome"
    :param profile: a vector of entropy values
    :param l: the window size
    :param m: number of permutations
    :return:
    """
    start_index = []
    p_values = []

    #create a per-profile distribution of averages, then sample
    avgs = np.array([])
    for j in range(m):
        new_profile = profile
        cur_avg = np.mean(new_profile[np.random.choice(len(new_profile), size=l, replace=False)])
        avgs = np.insert(avgs, avgs.searchsorted(cur_avg), cur_avg)

    for i in tqdm(range(0,len(profile) - l)):
        # get the current window and its average value
        w = profile[i:i+l]
        avg = np.mean(w)

        # sort average in order to get the p value
        idx = np.searchsorted(avgs, avg)
        p_value = idx/m
        p_values.append(p_value)
        start_index.append(i)

    data =  pd.DataFrame({'start':start_index, 'p_value':p_values, 'l':l})

    # correct for multiple tests
    data['corrected_pvalue'] = multi.fdrcorrection(data['p_value'])[1]

    return data


###### utils for the detection of repetitive strings #####


# taken from: https://gist.github.com/BertrandBordage/611a915e034c47aa5d38911fc0bc7df9
ASCII_TO_INT: dict = {i.to_bytes(1, 'big'): i for i in range(256)}
INT_TO_ASCII: dict = {i: b for b, i in ASCII_TO_INT.items()}


def compress(data: AnyStr):
    """
    this method compresses a string and returns its representation in bits.
    notice that one can compare between length of compressed sequences iff the sequences have the same length.
    :param data: a string
    :return: bits representing the string compressed.
    """
    if isinstance(data, str):
        data = data.encode()
    keys: dict = ASCII_TO_INT.copy()
    n_keys: int = 256
    compressed: list = []
    start: int = 0
    n_data: int = len(data)+1
    while True:
        if n_keys >= 512:
            keys = ASCII_TO_INT.copy()
            n_keys = 256
        for i in range(1, n_data-start):
            w: bytes = data[start:start+i]
            if w not in keys:
                compressed.append(keys[w[:-1]])
                keys[w] = n_keys
                start += i-1
                n_keys += 1
                break
        else:
            compressed.append(keys[w])
            break
    bits: str = ''.join([bin(i)[2:].zfill(9) for i in compressed])
    in_bytes = int(bits, 2).to_bytes(ceil(len(bits) / 8), 'big')

    return bits

def get_compression_profile(fasta, w, out=None, type='fasta'):
    """
    sliding window lzw compression profile of all sequences in a family
    :param fasta: a fasta file contatining viral sequences
    :param w: the window size
    :param out: optional. if != None a profile will be saved as a png
    :return: the vector of profile entropy
    """
    all_lzw = {}
    alias = os.path.basename(fasta).split('.')[0]

    i = 0
    for rec in SeqIO.parse(fasta, type):
        lzws = []
        # get identifier and genomic sequence

        genome = str(rec.seq)

        for j in range(len(genome) - w):
            sub_genome = genome[j:j+w]
            lzw = compress(sub_genome)
            lzws.append(lzw)


        print('Done with seq {}'.format(i))
        all_lzw['seq_{}'.format(i)] = lzws
        i += 1

    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in all_lzw.items()]))

    df.to_csv(os.path.join(out, '{}_compressed_profile.csv'.format(alias)), index=False)

    return df

def p_generator():
    """
    this method generates different distributions length 4 for each one of the nucleotides
    :param n: number of probabilities to define
    :return: list of multiple p's
    """
    ps = []

    # random
    for i in range(10**4):
        rand_a = np.random.rand(4)
        rand_a /= rand_a.sum()

        ps.append(rand_a)

    # uniforms
    for i in range(10**4):
        ps.append(np.array([0.25,0.25,0.25,0.25]))

    random.shuffle(ps)

    return ps

def sequences_generator(alphabet, w, m=10**4):
    """
    this method generates sequences by a known distribution on the alphabet content.
    :param alphabet:a list containing the wanted alphabet
    :param w: the size of the sequence generated
    :param m: number of iterations
    :return: a generator of sequences.
    """
    # get probabilities
    ps = p_generator()

    for i in range(m):
        yield ''.join(np.random.choice(alphabet,size=w, p=ps[i])), ps[i]


def create_data_matrix(alphabet, w):
    """
    create a matrix of the input data
    :param alphabet: a list containing the wanted alphabet
    :param w: the size of the sequence generated
    :return: a data frame with the input data
    """

    dfs = []
    for seq, p in sequences_generator(alphabet, w):
        entropy = entropy_by_kmer(seq, 5)
        lzw = len(compress(seq))
        df = pd.DataFrame({'entropy':entropy, 'lzw':lzw, 'a':[0], 'c':p[1], 'g':p[2], 't':p[3]}, index=[0])
        dfs.append(df)

    result = pd.concat(dfs)
    return result

def add_labels_to_matrix(X):
    """
    this method adds a label vector to the data frame X
    :param X: a data frame with the models features
    :return: a data frame with a y column describing the label of the entry - 1 is for repetitive and 0 ow
    """

    # get the threshold for the top 5 %
    p = np.percentile(X['lzw'], 10)

    # label the entries
    X['y'] = X['lzw'].apply(lambda x: 1 if x <= p else 0)
    return X


def random_forest_classifier(data):
    """
    bulid a random forest classifier and test its accuracuy
    :param data: input data frame with features and labels
    :return:
    """

    X = data[['entropy', 'lzw', 'a', 'c', 'g', 't']]  # Features
    y = data['y']  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Model Accuracy
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


    # analysis of multiple classification models taken from:
    # https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#sphx-glr-auto-examples-calibration-plot-compare-calibration-py
    # Create classifiers
    lr = LogisticRegression(solver='lbfgs')
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier(n_estimators=100)

    # #############################################################################
    # Plot calibration plots

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (gnb, 'Naive Bayes'),
                      (svc, 'Support Vector Classification'),
                      (rfc, 'Random Forest')]:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name,))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives", fontsize=14)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)', fontsize=14)

    ax2.set_xlabel("Mean predicted value", fontsize=14)
    ax2.set_ylabel("Count", fontsize=14)
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()


def test_real_data_per_seq(seq, data, w, index):
    """
    predict for each sequence in fasta its label
    :param fasta: sequence
    :param data: the data used to build the classifier
    :param w: the size of the sequence in the genome profile
    :param index: an index for the sequence. this will be used in the output df
    :return:
    """
    seq = seq.lower()
    # fit a random forest model
    X = data[['entropy', 'lzw', 'a', 'c', 'g', 't']]  # Features
    y = data['y']  # Labels

    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X, y)

    # iterate over the sequences and get all features needed for prediction
    dfs = []
    for j in tqdm(range(len(seq) - w)):
        sub_genome = seq[j:j + w]
        n = len(sub_genome)
        entropy = entropy_by_kmer(sub_genome, 5)
        lzw = len(compress(sub_genome))
        df = pd.DataFrame({'entropy':entropy, 'lzw':lzw, 'a':sub_genome.count('a') / n,'c':sub_genome.count('c') / n,
                           'g': sub_genome.count('g') / n,'t':sub_genome.count('t') / n}, index=[j])
        dfs.append(df)

    result =  pd.concat(dfs)

    # predict new labels
    y_pred = clf.predict(result)
    return pd.DataFrame({'predicted_{}'.format(index):y_pred})


def generate_all_predictions(fasta, data):
    """
    this method predicts the labels of all sequences in a fasta file
    :param fasta: a fasta file containing sequences
    :param data: a data which a model in train by
    :return: a data frame with all predictions
    """

    dfs = []
    # iteratively get all sequences and their predictions
    i = 0
    for rec in SeqIO.parse(fasta, "fasta"):
        seq = str(rec.seq)
        pred = test_real_data_per_seq(seq, data, 200, i)
        dfs.append(pred)
        i += 1

    result = pd.concat(dfs, axis=1)
    return result


# Euclidean Distance Calculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Residuals squares
def RS(a,b,ax=1):
    return np.array([(a[1] - (c[1] + a[0] * c[0])) ** 2 for c in b])


def lines_k_means(x, y, k=3):
    """
    implementation of a line based kmeans (instead of centroids)
    :param k: number of clusters
    :param x: vector of points
    :param y: vector of points (corresponds to x)
    :return: division to clusters
    """

    # store all the points in a 2d array.
    X = np.array(list(zip(x, y)))
    C_x = []
    # intercept of random centroids
    C_y = []

    init_C = np.random.choice(list(range(k)), size = len(X))
    for i in range(k):
        curX = [X[j] for j in range(len(X)) if init_C[j] == i]
        slope, intercept, r_value, p_value, std_err = stats.linregress([p[0] for p in curX], [p[1] for p in curX])
        C_x.append(slope)
        C_y.append(intercept)

    # slope of random centroids
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)
    # Loop will run till the error becomes zero
    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = RS(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by re calculating the slope and the
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            slope, intercept, r_value, p_value, std_err = stats.linregress([p[0] for p in points], [p[1] for p in points])
            C[i] = slope, intercept
        error = dist(C, C_old, None)

    return clusters



def simulate_dataset(n, size):
    """
    simulate sequences from different classes ( up to 4 : repetitive, repetitive with stem loops, random, only stem loop
    :param n: number of sequences to simulate
    :param size: the size of each sequence
    :return: a data frame containing a sequence, entropy and joint entropy, together with a type indicating the class
    """

    sequences = []
    cluster = []

    # repetitive sequences
    for i in tqdm(range(n)):
        cluster_name = 'Repetitive'
        seq = ''.join(np.random.choice(['a', 'c', 'g', 't'], p=[0.6, 0.2, 0.1, 0.1], size=size))
        sequences.append(seq)
        cluster.append(cluster_name)

    # create a data frame with all information
    df_rep = pd.DataFrame({'sequence':sequences, 'cluster':cluster})
    df_rep['entropy'] = df_rep['sequence'].apply(lambda x: entropy_by_kmer(x,5))
    df_rep['joint_entropy'] = df_rep['sequence'].apply(lambda x: joint_entropy(x, str(get_reverse_complement(x)), 5))

    # normalize bpth entropy and joint entropy to 0-1
    df_rep['entropy'] = df_rep['entropy'] /  df_rep['entropy'].max()
    df_rep['joint_entropy'] = df_rep['joint_entropy'] / df_rep['joint_entropy'].max()


    sequences = []
    cluster = []
    # repetitive sequences + structure - generate a perfect stem loop
    for i in tqdm(range(n)):
        cluster_name = 'Repetitive + Stem loop'
        seq = ''.join(np.random.choice(['a', 'c', 'g', 't'], p=[0.6, 0.2, 0.1, 0.1], size=size//2))
        seq = seq + str(get_reverse_complement(seq))
        sequences.append(seq)
        cluster.append(cluster_name)

    # create a data frame with all information
    df_rep_st = pd.DataFrame({'sequence':sequences, 'cluster':cluster})
    df_rep_st['entropy'] = df_rep_st['sequence'].apply(lambda x: entropy_by_kmer(x,5))
    df_rep_st['joint_entropy'] = df_rep_st['sequence'].apply(lambda x: joint_entropy(x, str(get_reverse_complement(x)), 5))

    # normalize bpth entropy and joint entropy to 0-1
    df_rep_st['entropy'] = df_rep_st['entropy'] /  df_rep_st['entropy'].max()
    df_rep_st['joint_entropy'] = df_rep_st['joint_entropy'] / df_rep_st['joint_entropy'].max()

    sequences = []
    cluster = []
    # only structure - generate a perfect stem loop
    for i in tqdm(range(n)):
        cluster_name = 'Stem loop'
        seq = ''.join(np.random.choice(['a', 'c', 'g', 't'], p=[0.25, 0.25, 0.25, 0.25], size=size//2))
        seq = seq + str(get_reverse_complement(seq))
        sequences.append(seq)
        cluster.append(cluster_name)

    # create a data frame with all information
    df_st = pd.DataFrame({'sequence':sequences, 'cluster':cluster})
    df_st['entropy'] = df_st['sequence'].apply(lambda x: entropy_by_kmer(x,5))
    df_st['joint_entropy'] = df_st['sequence'].apply(lambda x: joint_entropy(x, str(get_reverse_complement(x)), 5))

    # normalize bpth entropy and joint entropy to 0-1
    df_st['entropy'] = df_st['entropy'] /  df_st['entropy'].max()
    df_st['joint_entropy'] = df_st['joint_entropy'] / df_st['joint_entropy'].max()

    sequences = []
    cluster = []

    # random
    for i in tqdm(range(n)):
        cluster_name = 'Random'
        seq = ''.join(np.random.choice(['a', 'c', 'g', 't'], p=[0.25, 0.25, 0.25, 0.25], size=size))
        seq = seq + str(get_reverse_complement(seq))
        sequences.append(seq)
        cluster.append(cluster_name)

    # create a data frame with all information
    df_rand = pd.DataFrame({'sequence': sequences, 'cluster': cluster})
    df_rand['entropy'] = df_rand['sequence'].apply(lambda x: entropy_by_kmer(x, 5))
    df_rand['joint_entropy'] = df_rand['sequence'].apply(lambda x: joint_entropy(x, str(get_reverse_complement(x)), 5))

    # normalize bpth entropy and joint entropy to 0-1
    df_rand['entropy'] = df_rand['entropy'] /  df_rand['entropy'].max()
    df_rand['joint_entropy'] = df_rand['joint_entropy'] / df_rand['joint_entropy'].max()


    # combine all inputs to one df, and return it

    result = pd.concat([df_rep, df_rep_st, df_st, df_rand])
    return result



def GMM_clustering(data, k):
    """
    This method clusters 'data' using a gaussian mixture model. the motivation is that each cluster was generated from
    from a different distribution.
    :param data: data frame containing entropy and joint entropy columns
    :param k: the number of clusters
    :return: a data frame with cluster assignment for each row.
    """

    X = data[['entropy', 'joint_entropy']]
    # init model params, fit and predict

    gmm = GaussianMixture(n_components=k)
    gmm.fit(X)
    y_cluster_gmm = gmm.predict(X)

    data['GMM_cluster'] = y_cluster_gmm

    return data


def GMM_model_selection(data):
    """
    This method clusters 'data' using a gaussian mixture model. the motivation is that each cluster was generated from
    from a different distribution.
    :param data: data frame containing entropy and joint entropy columns
    :return: a data frame with clustering measurements
    """

    X = data[['entropy', 'joint_entropy']]

    n_cluster = [2,3,4,5,6]
    cov = ['full', 'tied', 'diag', 'spherical']

    dfs = []
    for k in n_cluster:
        for c in cov:
            gmm = GaussianMixture(n_components=k, covariance_type=c)
            gmm.fit(X)
            y_cluster_gmm = gmm.predict(X)

            bic = gmm.bic(X)
            score = adjusted_rand_score(data['cluster'], y_cluster_gmm)
            df = pd.DataFrame({'k':k, 'covariance':c, 'bic':bic, 'score':score}, index=[0])
            dfs.append(df)

    result = pd.concat(dfs)
    return result


def clustering_performance_evaluation(X, y_pred, y_true):
    """
    this function implement multiple evaluation metrics for clustering analysis.
    this method will be used in order to asses the quality of a clustering solution based on multiple criteria
    :param X: input matrix
    :param y_pred: predicted vector
    :param y_true: ground truth - if none - one do not have this knowledge
    :return: a dictionary with all measures
    """

    result = {}
    result['ARI'] = metrics.adjusted_rand_score(y_true, y_pred)
    result['AMI'] = metrics.adjusted_mutual_info_score(y_true, y_pred)
    result['NMI'] = metrics.normalized_mutual_info_score(y_true, y_pred)
    h,c,v = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
    result['Homo'] = h
    result['Comp'] = c
    result['V'] = v
    result['FM'] = metrics.fowlkes_mallows_score(y_true, y_pred)

    result['Sil'] = metrics.silhouette_score(X[['entropy', 'joint_entropy']], y_pred, metric='euclidean')

    return result


def generate_scores(data, model='gmm'):
    """
    run multiple models and get the score foe each one
    :param data: input data matrix
    :param model: the type of model tested
    :return: a matrix with all scores information by model type
    """

    n_clusters = [1,2,3,4,5,6]
    dfs = []

    for k in tqdm(n_clusters):
        if model =='gmm':
            model = 'GMM'
            X = GMM_clustering(data, k)
            y_pred = X['GMM_cluster']
            y_true = X['cluster']

        else:
            model = 'K-MEANS'
            y_pred = lines_k_means(data['entropy'], data['joint_entropy'], k)
            y_true = data['cluster']

        scores = clustering_performance_evaluation(data, y_pred, y_true)
        df = pd.DataFrame({'score':list(scores.keys()), 'value':list(scores.values()), 'k':k, 'model':model})
        dfs.append(df)

    result = pd.concat(dfs)
    return result


def pca_analysis(shannon, joint, deltag, seq_id, k=3, compressed=None):
    """

    :param shannon:
    :param joint:
    :param deltag:
    :param seq_id:
    :return:
    """

    X = pd.DataFrame({'x':shannon[seq_id], 'y':joint[seq_id], 'z':deltag[seq_id]})

    if compressed is not None:
        X = pd.DataFrame({'x': shannon[seq_id], 'y': joint[seq_id], 'z': deltag[seq_id], 'q':compressed[seq_id]})

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2'])
    principalDf['position'] = principalDf.index
    principalDf['position'] = principalDf['position'] + 1

    clusters = lines_k_means(principalDf['principal component 1'], principalDf['principal component 2'], k)
    principalDf['kmeans_clusters'] = [int(i) for i in clusters]

    gmm = GaussianMixture(n_components=k)
    gmm.fit(X)
    clusters_gmm = gmm.predict(X)
    principalDf['GMM_clusters'] = clusters_gmm

    return principalDf





def find_sequential_drops(shannon, joint, output, t=0.05):
    """
    given a family generate a table of all sequential drops
    :param shannon: shannon entropy matrix
    :param joint: joint entropy matrix
    :param t: p value threshold
    :return: data frame with stat and end points of every significant drop
    """

    col_dfs = []

    # for each feature in the matrix calculate the number of drops. locations and sizes

    for c in tqdm(shannon):
        if c not in joint.columns:
            continue
        x = shannon[c].dropna()
        y = joint[c].dropna()

        # get only significant drops
        x = x[x <= t]
        y = y[y <= t]

        # remember index as a new column - had bugs with splitting by index later...
        x = x.reset_index()
        y = y.reset_index()

        dx = np.diff(x['index'])
        dy = np.diff(y['index'])

        x_pos = [i + 1 for i in np.where(dx > 1)[0]]
        y_pos = [i + 1 for i in np.where(dy > 1)[0]]

        x_mod = [0] + x_pos + [len(x) + 1]
        y_mod = [0] + y_pos + [len(y) + 1]

        x_splits = [x.iloc[x_mod[n]:x_mod[n + 1]] for n in range(len(x_mod) - 1)]
        y_splits = [y.iloc[y_mod[n]:y_mod[n + 1]] for n in range(len(y_mod) - 1)]

        # fil start and end positions creating a dataframe for each column
        x_starts = []
        x_ends = []

        for s in x_splits:
            x_starts.append(s['index'].iloc[0])
            x_ends.append(s['index'].iloc[-1])

        y_starts = []
        y_ends = []

        for s in y_splits:
            y_starts.append(s['index'].iloc[0])
            y_ends.append(s['index'].iloc[-1])

        shannon_drops = pd.DataFrame({'start': x_starts, 'end': x_ends, 'type': 'shannon', 'seq': c})
        joint_drops = pd.DataFrame({'start': y_starts, 'end': y_ends, 'type': 'joint', 'seq': c})

        col_dfs.append(shannon_drops)
        col_dfs.append(joint_drops)

    drops = pd.concat(col_dfs)
    drops['size'] = drops['end'] - drops['start'] + 1

    if output != None:
        drops.to_csv(output, index=False)

    return drops


def drop_Graphs(drops, out=None):
    """
    plots a graph of shannon vs. joint drops
    :param drops: a drops data frame for a given sequence
    :return:
    """

    shannon = drops[drops['type']=='shannon']
    joint = drops[drops['type'] == 'joint']

    ss = shannon['start'].values
    se = shannon['end'].values

    js = joint['start'].values
    je = joint['end'].values

    # create two graphs, one for each measurment

    G = nx.Graph()
    G2 = nx.Graph()

    # create the two graphs
    for i in range(len(ss)):
        G.add_node(ss[i], pos=(i,10))
        G.add_node(se[i], pos=(i, 10))

    for i in range(len(js)):
        G2.add_node(js[i], pos=(i, 10.005))
        G2.add_node(je[i], pos=(i, 10.005))

    # get locations
    pos = nx.get_node_attributes(G, 'pos')
    pos2 = nx.get_node_attributes(G2, 'pos')

    # draw graph
    nx.draw(G, with_labels=True, node_size=500, node_color="skyblue", node_shape="o", alpha=0.5, linewidths=4,
            font_size=8, font_color="grey", font_weight="bold", width=2, edge_color="black", pos=pos)

    nx.draw(G2, with_labels=True, node_size=500, node_color="pink", node_shape="o", alpha=0.5, linewidths=4,
            font_size=8, font_color="grey", font_weight="bold", width=2, edge_color="black", pos=pos2)


def merge_stats_by_type(folder, data_type, joint=False):
    """
    the method merges all stats csv files into a single file
    :param family: the family name
    :param joint: indicator for joint entropy. if true run the analysis on joint entropy
    :return: saves a csv with all p values
    """

    # define the folder from which all stats are going to be generated
    stats_files = glob.glob(os.path.join(folder, '*{}*seq*stats.csv'.format(data_type)))
    mapping = {}
    for f in stats_files:
        df = pd.read_csv(f)
        seq_id = re.findall(r'seq_\d+', f)[0]
        mapping[seq_id] = df['corrected_pvalue']

    # create a complete dataframe with all p values
    stats_by_seq = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in mapping.items()]))

    output = os.path.join(folder, '{}_all_stats.csv'.format(data_type))

    stats_by_seq.to_csv(output, index=False)
    print('Done!')
    return stats_by_seq

def merge_all_drops(shannon, joint, drops):
    """
    merge all information on drops
    :param shannon: data frame with shannon entropy profiles by seq
    :param joint: data frame with joint entropy profiles by seq
    :param drops: the output of find sequential drops
    :return:
    """
    all_sequences = list(shannon.columns)
    dfs = []
    for seq in tqdm(all_sequences):
        seq_drops = drops[drops['seq'] == seq]
        seq_drops['position'] = seq_drops.apply(lambda row: list(range(row['start'], row['end'] + 1)), axis=1)

        # melt
        seq_drops = pd.DataFrame([[a, b, c, d, e, p] for a, b, c, d, e, P in \
                                  seq_drops.values for p in P], columns=seq_drops.columns)

        # merge with sequence values by position

        cur_shannon = shannon[seq].reset_index()
        cur_shannon['position'] = cur_shannon['index'] + 1
        cur_shannon['shannon'] = cur_shannon[seq]

        cur_joint = joint[seq].reset_index()
        cur_joint['position'] = cur_joint['index'] + 1
        cur_joint['joint'] = cur_joint[seq]

        cur_shannon.drop(columns=['index', seq], inplace=True)
        cur_joint.drop(columns=['index', seq], inplace=True)

        merged = pd.merge(seq_drops, cur_shannon, on=['position'])
        merged = pd.merge(merged, cur_joint, on=['position'])
        dfs.append(merged)

    result = pd.concat(dfs)
    return result