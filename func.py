import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn import tree
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid


seed = 42
np.random.seed(seed)


def load(data_in):
    print("Loading your dataset from provided file..")
    tmp = pd.read_csv(data_in)
    return tmp


def preElaborationData(data, cols):
    for i in cols:
        print(data[i].describe())
        print()


def preElaborationClass(data, target):
    data.hist(target, bins = 5, grid = False)
    plt.title("")
    plt.ylabel("Frequency")
    plt.show()


def preBoxPlotAnalysisData(data, cols, target):
    cols = cols[0:data.shape[1] - 1]  # Remove the Class column from the dataframe
    for column in cols:
        sub = '/'
        name = column
        outputBoxplot = data.boxplot(column, target, fontsize = 12, figsize = (8, 6))
        plt.title(name, fontsize = 14)
        plt.xlabel('Class', fontsize = 14)
        fig = plt.gcf()
        if (sub in name):
            name = column.replace(sub, '_')
        fig.savefig('/my_project/boxplots/boxplot_' + name + '.png')
        plt.show()


def mutualInfoRank(data, independentList, label):
    res = dict(zip(independentList,
                   mutual_info_classif(data[independentList], data[label],
                                       discrete_features = False, random_state = seed)))
    sorted_x = sorted(res.items(), key = lambda kv: kv[1], reverse = True)
    return sorted_x


def topFeatureSelect(rankMutualInfoList, number_top_attr):
    tmp = []
    for i in range(0, number_top_attr):
        tmp.append(rankMutualInfoList[i][0])
    return tmp


def pca(data):
    pca = PCA()
    pca.fit(data)
    list = []
    v = []
    for c in range(len(data.columns.values)):
        v = "pc_" + str(c + 1)
        list.append(v)
    return pca, list


def applyPCA(data, pca, pcalist):
    principal_components = pca.transform(data)
    principal_proj_data = pd.DataFrame(data = principal_components, columns = pcalist)
    return principal_proj_data


def selectPCAData(pcaData, N):
    return pcaData.iloc[:, list(range(N)) + [-1]]


def stratifiedKfold(X, y, folds, seed):
    skf = StratifiedKFold(n_splits = folds, random_state = seed, shuffle = True)

    ListXTrain = []
    ListXTest = []
    ListYTrain = []
    ListYTest = []

    print("Output from stratifiedKfold function:")
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        ListXTrain.append(X.iloc[train_index])
        ListXTest.append(X.iloc[test_index])
        ListYTrain.append(y.iloc[train_index])
        ListYTest.append(y.iloc[test_index])
    return ListXTrain, ListXTest, ListYTrain, ListYTest

def decisionTreeLearner(X, y, crit, min_split):
    T = DecisionTreeClassifier(random_state = 0, criterion = crit, min_samples_split = min_split)
    T.fit(X, y)
    return T


def showTree(T):
    n_nodes = T.tree_.node_count
    n_leaves = T.tree_.n_leaves
    plt.figure(1, figsize = (20, 20))
    # plt.figure(figsize=(1440 / 192, 900 / 192), dpi=192) - 1440x900 is the res of my screen with 192dpi on my macbook 12'
    tree.plot_tree(T, fontsize = 4)
    print("The decision tree learned have:", n_nodes, "nodes and", n_leaves, "leaves!")
    plt.show()


# compute the f-score - values on output between [0,1] - Y_True is your YTest, YPred predict the class for X so XTest (proper method of tree_)
def decisionTreeF1(XTest, YTest, T):
    YPred = T.predict(X = XTest)
    my_weighted_f1 = f1_score(YTest, YPred, average = "weighted")
    return my_weighted_f1


def determineDecisionTreekFoldConfiguration(ListXTrain, ListXTest, ListYTrain, ListYTest, feature_ranking):
    crit = ['gini', 'entropy']
    config = []

    for i in crit:
        for j in range(5, len(feature_ranking), 5):
            sum_f1 = 0
            top_val = topFeatureSelect(feature_ranking, j)
            for k in range(5):
                T = decisionTreeLearner(ListXTrain[k].loc[:, top_val], ListYTrain[k], i, 500)
                sum_f1 += decisionTreeF1(ListXTest[k].loc[:, top_val], ListYTest[k], T)
                mean_f1 = sum_f1 / 5
            config.append([i, j, mean_f1])

    best_config = max(config, key = lambda x: x[2])
    best_crit = best_config[0]
    best_N = best_config[1]
    best_eval = best_config[2]
    print(config)
    return best_crit, best_N, best_eval


def benchmark(y_true, y_pred, labels):
    conf_matx = confusion_matrix(y_true, y_pred, labels = labels)
    fig = ConfusionMatrixDisplay(confusion_matrix = conf_matx, display_labels = labels)
    print(classification_report(y_true, y_pred))
    fig.plot()
    plt.show()


def assign_clusters_classes(X_train, y_train, num_clusters, n_init):
    kmeans = KMeans(init="k-means++", n_clusters = num_clusters, n_init = n_init,
                    max_iter = 300, tol = 0.0001, random_state = seed, algorithm = "full")
    kmeans.fit(X_train)

    cluster_labels = kmeans.labels_
    class_assignments = {}

    for cluster_label in np.unique(cluster_labels):
        cluster_samples = np.where(cluster_labels == cluster_label)[0]
        cluster_classes = y_train[cluster_samples]
        most_common = np.argmax(np.bincount(cluster_classes))                   # most common class
        class_assignments[cluster_label] = most_common
    return class_assignments, kmeans.inertia_


def predict_the_classes(X_test, X_train, y_train, class_assignments):
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(X_train, y_train)
    predicted_classes = nearest_centroid.predict(X_test)

    cluster_labels = nearest_centroid.predict(X_test)

    predicted_classes = [class_assignments[label] for label in cluster_labels]
    return predicted_classes


def predict_classes_KM(X_test, X_train, class_assignments):
    kmeans = KMeans(init = "k-means++", n_clusters = len(class_assignments), random_state = seed)
    kmeans.fit(X_train)
    cluster_labels = kmeans.predict(X_test)
    predicted_classes = [class_assignments[label] for label in cluster_labels]
    return predicted_classes


def benchmark_predictions(y_test, predicted_classes, labels = None):
    conf_mtx = confusion_matrix(y_test, predicted_classes, labels = labels)
    fig = ConfusionMatrixDisplay(confusion_matrix = conf_mtx, display_labels = labels)
    fig.plot()
    plt.show()
    classification_rep = classification_report(y_test, predicted_classes, zero_division = 0)
    return conf_mtx, classification_rep