from sklearn import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, cross_validate

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression, Lasso
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
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
    #("KNN-3", KNeighborsClassifier(3)),
    #("KNN-10", KNeighborsClassifier(10)),
    #("SVC-linear", SVC(kernel="linear", C=0.025, probability=True)),
    #("SVC", SVC(gamma=2, C=1, probability=True)),
    #("GPC", GaussianProcessClassifier(1.0 * RBF(1.0))),
    #("DT", DecisionTreeClassifier(max_depth=5)),
    ("RF", RandomForestClassifier(max_depth=5, n_estimators=100)),
    #("NN", MLPClassifier(alpha=1, max_iter=1000)),
    ("Adaboost", AdaBoostClassifier()),
    ("LR", LogisticRegression(max_iter=10000)),
    #("LR-noReg", LogisticRegression(C=np.Inf, max_iter=100000)),
    #("GaussianNB", GaussianNB()),
    #("QDA", QuadraticDiscriminantAnalysis(),
    ("GBC", GradientBoostingClassifier()),
]


REGRESSORS = [
        ("LR", LinearRegression()),
        ("RF", RandomForestRegressor()),
        ("Ridge", Ridge(max_iter=10000)),
        ("GBR", GradientBoostingRegressor(),
         ("Lasso", Lasso()))
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
    try:
        X = data.drop(columns=['median_conservation', 'conserved', 'loc', 'nucleotide_info','TBL', 'TBL_class'])
    except:
        X = data.drop(columns=['median_conservation', 'conserved', 'loc', 'nucleotide_info'])
    y = data['median_conservation'].astype(int).values
    if transform:
        pipeline = make_preprocessing_pipeline()
        X = pipeline.fit_transform(X)

    return X, y

def features_importance(data, mdl_info):
    """
    generate the feature importance data frame
    :param data: original data frame
    :param mdl_info: output of cross_validate function
    :return:
    """
    res = []
    labels = {str(n):data.columns[n] for n in range(len(data.columns))}
    for idx, estimator in enumerate(mdl_info['estimator']):
        print("Features sorted by their score for estimator {}:".format(idx))
        feature_importances = pd.DataFrame(estimator.feature_importances_,
                                           columns=['importance']).sort_values('importance', ascending=False)

        feature_importances['type'] = feature_importances.apply(lambda row: labels[str(row.name)], axis=1)
        feature_importances['CV'] = idx
        res.append(feature_importances)
    return pd.concat(res)


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
            scores = cross_validate(std_clf, X, y, cv=clf_cv, scoring="accuracy", return_estimator =True)

            print(name, np.mean(scores['test_score']), np.std(scores['test_score']))
            clf_scores.append(pd.DataFrame({'model':name, 'ACC':scores['test_score'].mean(), 'STD':scores['test_score'].std()}, index=[0]))

        output_path = os.path.join(args.out_path, "accuracies_classifiers.csv")
        pd.concat(clf_scores).to_csv(output_path, index=False)

        reg_scores = []
        reg_cv = KFold(10)
        # Iterate over classifiers
        for name, reg in REGRESSORS:
            std_reg = Pipeline([
                ("Standartize", StandardScaler()),
                ("Regression", clone(reg))
            ])
            scores = cross_validate(std_reg, X, y, cv=reg_cv, scoring="r2", return_estimator =True)

            print(name, np.mean(scores['test_score']), np.std(scores['test_score']))
            reg_scores.append(pd.DataFrame({'model':name, 'R2':scores['test_score'].mean(), 'STD':scores['test_score'].std()}, index=[0]))

        output_path = os.path.join(args.out_path, "accuracies_regression.csv")
        pd.concat(reg_scores).to_csv(output_path, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate conservation prediction scores using ML')
    parser.add_argument('-f', '--file_path', help="a path to the data file matrix")
    parser.add_argument('-o', '--out_path', help="where the results will be saved - directory")

    args = parser.parse_args()
    main(args)
