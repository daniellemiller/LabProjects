import glob

from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, cross_validate

import numpy as np
import argparse
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
from tqdm import tqdm
from scipy import interp
import matplotlib.pyplot as plt
import seaborn as sns
from learner import load_data


### Classification
def process_data(fname):
   X, y = load_data(fname)
   # create a binary matrix of all classes
   y = label_binarize(y, classes=np.unique(y))
   return X, y


def proba_mass_split(y, folds=7):
    # taken from : https://stats.stackexchange.com/questions/65828/how-to-use-scikit-learns-cross-validation-functions-on-multi-label-classifiers

    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(obs):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)

    return index_list



def multiclass_roc(X_train, X_test, y_train, y_test, n_classes, clf=RandomForestClassifier(), fold=1, out=None, mdl='RF'):
    """
    plot a multiclass roc curve
    based on : https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    # learn to predict each class against the other
    classifier = OneVsRestClassifier(clf)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        # include only defined aucs - exclude tiny classes
        if np.isnan(auc(fpr[i], tpr[i])):
            continue
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        if i in roc_auc:        # correction for tiny classes. they are not present in test split hence tpr is not defined
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    with sns.plotting_context(rc={"font.size":14,"axes.titlesize":18,"axes.labelsize":18,
                              "xtick.labelsize":14,"ytick.labelsize":14,'y.labelsize':16}):
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average(area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average(area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = sns.color_palette('Blues', n_colors=n_classes+1)
        for i, color in zip(range(n_classes), colors):
            if i in roc_auc:
                plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                         label='class {0}(area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(f'Multi-class {mdl} ROC fold {fold}', fontsize=18)
        plt.legend(bbox_to_anchor=[1.05,1])
        if out != None:
            plt.savefig(out,format='pdf', bbox_inches='tight', dpi=400)
            plt.gcf().clear()
        else:
            plt.show()
    return pd.DataFrame({'Type':list(roc_auc.keys()), 'AUC':list(roc_auc.values()), "Classifier":mdl, "Fold":fold})


if __name__ == "__main__":
    CLASSIFIERS = [
        ("RF", RandomForestClassifier(max_depth=5, n_estimators=100)),
        # ("LR", LogisticRegression(max_iter=10000)),
        ("GBC", GradientBoostingClassifier()),
    ]
    REGRESSORS = [
        ("LR", LinearRegression()),
        ("RF", RandomForestRegressor()),
        ("Ridge", Ridge(max_iter=10000)),
        ("GBR", GradientBoostingRegressor()),
        ("Lasso", Lasso())
    ]
    files = glob.glob(r'data/train_*.csv')
    for fname in files:
        if 'Kmeans' in fname:
            continue
        print(fname)
        if os.path.exists(f"data/ACC/{os.path.basename(fname).split('.csv')[0]}CLF_AUC_BY_MODEL.csv"):
            continue
        X, y = process_data(fname)
        n_class = y.shape[1]
        auc_by_mdl = []

        #define once al train, test idxs
        test_idx_by_fold = proba_mass_split(y, folds=10)
        for name, clf in tqdm(CLASSIFIERS):
            std_clf = Pipeline([
                ("Standartize", StandardScaler()),
                ("Classification", clone(clf))
            ])
            for i, test in enumerate(test_idx_by_fold):
                # define train and test
                train = [x for x in np.arange(y.shape[0]) if x not in test]
                X_train = X[train]
                X_test = X[test]
                y_train = y[train]
                y_test = y[test]

                base = os.path.basename(fname).split('.csv')[0] + "_" + f"{name}_{i}.pdf"
                out = os.path.join('data/plots', base)
                result_auc = multiclass_roc(X_train, X_test, y_train, y_test, n_class,
                                            clf=std_clf, fold=i, out=out, mdl=name)
                auc_by_mdl.append(result_auc)

        pd.concat(auc_by_mdl).to_csv(f"data/ACC/{os.path.basename(fname).split('.csv')[0]}CLF_AUC_BY_MODEL.csv", index=False)

        X, y = load_data(fname)
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
            reg_scores.append(pd.DataFrame({'model':name, 'R2':scores['test_score'].mean(),
                                            'STD':scores['test_score'].std()}, index=[0]))

        pd.concat(reg_scores).to_csv(f"data/ACC/{os.path.basename(fname).split('.csv')[0]}_REGR_AUC_BY_MODEL.csv",
                                     index=False)


