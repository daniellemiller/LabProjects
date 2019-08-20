import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools


def generate_profile(data, drops, col, data_type):
    """
    plot the profile color coded by p-value
    :param data: data frame with entropies\ deltag
    :param stats: corresponding p values data frame
    :param col: the sequence on which the plot should be generated
    :param data_type: the type of data used (shannon\ joint\ deltaG)
    :return: plot the graph
    """
    with sns.plotting_context(rc={"font.size":14,"axes.titlesize":18,"axes.labelsize":18,
                              "xtick.labelsize":14,"ytick.labelsize":14,'y.labelsize':16}):

        lower = 0.05

        df = pd.DataFrame({'position': data[col].index + 1, 'value': data[col]})
        df = df.dropna()

        drops = drops[drops['type'] == data_type]
        drops['list'] = drops.apply(lambda row: list(range(row['start'], row['end'] + 1)), axis=1)
        merged = list(itertools.chain(*drops['list'].values))

        df['stats'] = df['position'].apply(lambda x : 0 if x in merged else 1)

        slower = np.ma.masked_where(df['stats'].values > lower, df['value'].values)
        supper = np.ma.masked_where(df['stats'].values < lower, df['value'].values)

        plt.plot(df['position'], slower, color='#32B7DE', label = 'Drop')
        plt.plot(df['position'], supper, color='#DE3232', label='Non-drop')

        sns.despine()
        plt.legend()
        plt.xlabel('Position')
        plt.ylabel('value')
        plt.title('{} Profile'.format(data_type))


def generate_pca_classification(pc_df):

    with sns.plotting_context(rc={"font.size":14,"axes.titlesize":18,"axes.labelsize":18,
                              "xtick.labelsize":14,"ytick.labelsize":14,'y.labelsize':16}):

        f, axes = plt.subplots(1, 2)
        # sns.scatterplot(x='principal component 1', y = 'principal component 2', hue='ground_truth',marker = '+', data=principalDf, ax=axes[0])
        sns.scatterplot(x='principal component 1', y='principal component 2', hue='kmeans_clusters', marker='+',
                        data=pc_df, ax=axes[0])
        sns.scatterplot(x='principal component 1', y='principal component 2', hue='GMM_clusters', marker='+', data=pc_df,
                        ax=axes[1])

        sns.despine()
