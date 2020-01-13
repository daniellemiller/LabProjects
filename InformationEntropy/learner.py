from sklearn import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer



import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm

CLASSIFIERS = [
    ("KNN-3", KNeighborsClassifier(3)),
    ("KNN-10", KNeighborsClassifier(10)),
    ("SVC-linear", SVC(kernel="linear", C=0.025, probability=True)),
    ("SVC", SVC(gamma=2, C=1, probability=True)),
    ("GPC", GaussianProcessClassifier(1.0 * RBF(1.0))),
    ("DT", DecisionTreeClassifier(max_depth=5)),
    ("RF", RandomForestClassifier(max_depth=5, n_estimators=100)),
    ("NN", MLPClassifier(alpha=1, max_iter=1000)),
    ("Adaboost", AdaBoostClassifier()),
    ("LR", LogisticRegression(max_iter=10000)),
    ("LR-noReg", LogisticRegression(C=np.Inf, max_iter=100000)),
    ("GaussianNB", GaussianNB()),
    ("QDA", QuadraticDiscriminantAnalysis())
]


REGRESSORS = [
        ("LR", LinearRegression()),
        ("RF", RandomForestRegressor()),
        ("Ridge", Ridge(max_iter=10000)),
        ("GBR", GradientBoostingRegressor())
]


def make_preprocessing_pipeline():

    encoder =  Pipeline([
        ("encoding", ColumnTransformer([
            ("family", OrdinalEncoder(), ["family"]),
            ("drop_id", OrdinalEncoder(), ["drop_id"]),
            ("seq_id", OrdinalEncoder(), ["seq_id"]),
        ], remainder="passthrough"))
    ])
    return encoder


def load_data(fname, transform=True):
    data = pd.read_csv(fname)

    X = data.drop(columns=['median_conservation', 'conserved', 'loc', 'nucleotide_info'])
    y = data['median_conservation'].astype(int).values
    if transform:
        pipeline = make_preprocessing_pipeline()
        X = pipeline.fit_transform(X)

    return X, y



def main(args):

        # load data and preprocess
        X, y = load_data(args.file_path)

        clf_scores = []
        clf_cv = StratifiedKFold(10)
        # Iterate over classifiers
        for name, clf in CLASSIFIERS:
            print(name)
            std_clf = Pipeline([
                ("Standartize", StandardScaler()),
                ("Classification", clone(clf))
            ])
            scores = cross_val_score(std_clf, X, y, cv=clf_cv, scoring="accuracy")

            print(name, np.mean(scores), np.std(scores), scores)
            clf_scores.append(pd.DataFrame({'model':name, 'AUC':np.mean(scores), 'STD':np.std(scores)}, index=[0]))

        output_path = os.path.join(args.out_path, "accuracies_classifiers.csv")
        pd.concat(clf_scores).to_csv(output_path)

        reg_scores = []
        reg_cv = KFold(10)
        # Iterate over classifiers
        for name, reg in REGRESSORS:
            std_reg = Pipeline([
                ("Standartize", StandardScaler()),
                ("Regression", clone(reg))
            ])
            scores = cross_val_score(std_reg, X, y, cv=reg_cv, scoring="explained_variance")

            print(name, np.mean(scores), np.std(scores), scores)
            reg_scores.append(pd.DataFrame({'model':name, 'Score':np.mean(scores), 'STD':np.std(scores)}, index=[0]))

        output_path = os.path.join(args.out_path, "accuracies_regression.csv")
        pd.concat(reg_scores).to_csv(output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate conservation prediction scores using ML')
    parser.add_argument('-f', '--file_path', help="a path to the data file matrix")
    parser.add_argument('-o', '--out_path', help="where the results will be saved - directory")

    args = parser.parse_args()
    main(args)
