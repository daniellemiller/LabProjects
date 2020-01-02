import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

from InformationEntropy.conservation_analysis import get_size, rescale, extract_nucleotide_information, get_start_and_end

# This is a script aimed to learn the ranking function
# The main idea is to predict the conservation by the features matrix

def load_data(file_path, cluster_data=False, out=None):
    """
    this method loads the data into an input matrix for farther training
    :param file_path: a path to a training matrix
    :return: dataframe with all features
    """
    if cluster_data:
        files = glob.glob(
            r'/Volumes/STERNADILABHOME$/volume1/daniellem1/Entropy/conservation_analysis/training/*/*rank*.csv')
    else:
        files = glob.glob(file_path)

    cols_2_consider = ['Shannon_k{}'.format(k) for k in [1, 2, 3, 4, 5]] + ['Joint_k{}'.format(k) for k in
                                                                            [1, 2, 3, 4, 5]] + ['DeltaG']
    # read all files and add families to each dataframe.
    # this will be done once as this step takes time.
    dfs = []
    for f in tqdm(files):
        alias = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f)
        df = df.dropna()    # remove NA's
        df['family'] = alias

        # for some reason joint entropy values are 'object' and not float (due NA's)
        for c in tqdm(cols_2_consider):
            df[c] = df[c].astype(float)

        grouped = df.groupby(['family','seq_id', 'drop_id']).agg({np.mean, np.median, np.std}).reset_index()
        grouped.columns = [' '.join(col).strip() for col in grouped.columns.values] # flatten columns names
        dfs.append(grouped)

    data = pd.concat(dfs)
    data = data.dropna()

    if out != None:
        data.to_csv(out, index=False)
    return data

def feature_engineering(data):
    """
    rescales delta g parameter to be centered around random (~0.3)
    :param data: input data matrix for train
    :return: data mtrix scaled
    """

    cols = ['DeltaG mean', 'DeltaG median', 'DeltaG std', 'GMM_clusters mean',
       'GMM_clusters median', 'GMM_clusters std', 'Joint_k1 mean',
       'Joint_k1 median', 'Joint_k1 std', 'Joint_k2 mean', 'Joint_k2 median',
       'Joint_k2 std', 'Joint_k3 mean', 'Joint_k3 median', 'Joint_k3 std',
       'Joint_k4 mean', 'Joint_k4 median', 'Joint_k4 std', 'Joint_k5 mean',
       'Joint_k5 median', 'Joint_k5 std', 'Shannon_k1 mean',
       'Shannon_k1 median', 'Shannon_k1 std', 'Shannon_k2 mean',
       'Shannon_k2 median', 'Shannon_k2 std', 'Shannon_k3 mean',
       'Shannon_k3 median', 'Shannon_k3 std', 'Shannon_k4 mean',
       'Shannon_k4 median', 'Shannon_k4 std', 'Shannon_k5 mean',
       'Shannon_k5 median', 'Shannon_k5 std','drop_id', 'family',
       'median_conservation mean', 'position mean', 'rank mean', 'rank median', 'rank std', 'seq_id']

    # remove not needed features
    data = data[cols]

    # rename columns
    data = data.rename(columns={'position mean': 'position', 'median_conservation mean': 'median_conservation'})
    data['DeltaG median'] = data['DeltaG median'].apply(lambda x: 0 if x == -0.0 else x)
    data['scaled_DeltaG'] = data['DeltaG median'].apply(lambda x: rescale(x))
    data['conserved'] = data['median_conservation'].apply(lambda x: 0 if x <= 5 else 1)
    data['nucleotide_info'] = data.apply(lambda row: extract_nucleotide_information(row['family'], row['seq_id'],
                                                                                          row['drop_id']), axis=1)

    data = data[data['nucleotide_info'] != 'Not valid! seq_id not in fasta file']

    data['A_rel_genome'] = data['nucleotide_info'].apply(lambda lst: float(lst[0][0]))
    data['A_rel_drop'] = data['nucleotide_info'].apply(lambda lst: float(lst[0][-1]))

    data['C_rel_genome'] = data['nucleotide_info'].apply(lambda lst: float(lst[1][0]))
    data['C_rel_drop'] = data['nucleotide_info'].apply(lambda lst: float(lst[1][-1]))

    data['G_rel_genome'] = data['nucleotide_info'].apply(lambda lst: float(lst[2][0]))
    data['G_rel_drop'] = data['nucleotide_info'].apply(lambda lst: float(lst[2][-1]))

    data['T_rel_genome'] = data['nucleotide_info'].apply(lambda lst: float(lst[3][0]))
    data['T_rel_drop'] = data['nucleotide_info'].apply(lambda lst: float(lst[3][-1]))

    data['loc'] = data.apply(lambda row: get_start_and_end(row['family'], row['seq_id'], row['drop_id']), axis=1)
    data['start'] = data['loc'].apply(lambda x: x[0])
    data['end'] = data['loc'].apply(lambda x: x[-1])
    data['drop_size'] = data['end'] - data['start'] + 1

    return data


##################################### generate data matrix ###########################################

def generate_input_matrix():
    """
    run the above code for the generation of the training input matrix
    :return: saves the train data matrix into a file.
    this section should be run only once
    """
    # data = load_data(file_path='', cluster_data=True,
                     # out=r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/Entropy/ranking/train.csv')

    data = pd.read_csv(r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/Entropy/ranking/train.csv')
    added_features = feature_engineering(data)
    added_features.to_csv(r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/Entropy/ranking/train_features.csv',
                          index=False)

#######################################################################################################



##################################### learn ranking function ###########################################

# load the data to farther use
data = pd.read_csv(r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/Entropy/ranking/train_features.csv')

def factorize_all_object_features(data):
    """
    factorize non numeric data columns to fit a random forest with multiple labels
    :param data: the input matrix generated abpve
    :return: data frame with all columns
    """
    cols_to_factorize = [c for c in data.columns if c not in data.select_dtypes(np.number).columns]
    for col in cols_to_factorize:
        data[col] = pd.factorize(data[col])[0]
        # pd.factorize(data[col])[1] will hold all the definitions
    return data

def split_train_and_test(data, type='Reg', family=None):
    """
    split to train and test
    :param data: input data matrix
    :param type: splitting method
    :param family: name of a family of choice
    :return: splitted train and test
    """

    data['median_conservation'] = pd.factorize(data['median_conservation'])[0] # factorize for classes

    X = data.drop(columns=['median_conservation', 'conserved'])
    y = data['median_conservation']

    if type == 'family' and family != None:
        X_train = X[X['family'] !=  family]     # we will test on family alone
        X_test = X[X['family'] == family]

        y_train = data[data['family'] != family]['median_conservation']
        y_test = data[data['family'] == family]['median_conservation']

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """
    scale the data for training - standartization
    :param X_train: train x
    :param X_test:  test x
    :return: scaled train and test
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def run_multiple_classifiers(X_train, X_test, y_train, y_test):
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