from func import *


dataset_training = "/dataset/Maldroid2020_train.csv"
target = 'Class'

# load data phase
data = load(dataset_training)
if data is not None:
    print("Just ended to load your data from the file submitted!")
    print()
else:
    print("Got an error during loading of data provided!")
    sys.exit(0)

shape = data.shape              
print("Dataset size is:", shape, "which means", shape[0], "examples x", shape[1], "attributes")
print("Legend: examples = rows")
print("Legend: attributes = columns")
print()

print("Just a quick test to see if you loaded correct data:")
print(data.head())              
print()

print("These are the columns of your dataset:")
print(data.columns)

# pre-elaboration phase
cols = list(data.columns.values)                     
preElaborationData(data, cols)                      

print("Following you can evaluate on the screen the plot of your dataset with a setup of 5 bins. (Close to go ahead).")
preElaborationClass(data, target)
print()

print("In this step you can evaluate on the screen different boxplots. These will be saved into the folder: 'ana_dati_maldroid' folder, then sub-folder 'boxplots'")
preBoxPlotAnalysisData(data, cols, target)              
print()

# feature evaluation phase with mutual info rank - feature relevance study
print("This is a list of attributes in the dataset: ")
independentList = cols[0:data.shape[1] - 1]
print()
print("Determining the feature evaluation with Mutual Info Rank of original training set provided..")
rankMI = mutualInfoRank(data, independentList, target)
print(rankMI)
print()

# feature selection phase - Select the best N features
print("Moving on feature selection step, grabbing top N features ranked according to MI.")
N = 10
toplist = topFeatureSelect(rankMI, N)
toplist.append(target)
print("The top list is above here:")
print(toplist)

# Selecting the best N features
selectedMIData = data.loc[:, toplist]              
print()
print("After last manipulation your actual dataset dimension is:", selectedMIData.shape, "which means", selectedMIData.shape[0], "examples x", selectedMIData.shape[1], "columns")
print()
print("This is the head of your data:")
print(selectedMIData.head(n = 15))
print()
print("Actual attributes are these:")
print(selectedMIData.columns)
print()

# scaling phase with minmax algo + feature evaluation with mutual info rank on the scaled !training set!
scaled_data = data.copy(deep = True)            

scaler = MinMaxScaler()
scaled_data[independentList] = scaler.fit_transform(scaled_data[independentList])
print("The MinMaxScaler is used and now it's time to determining the feature evaluation with Mutual Info Rank of scaled dataset provided..")
rank_MI_scaled = mutualInfoRank(scaled_data, independentList, target)
print(rank_MI_scaled)
print()

toplist_scaled = topFeatureSelect(rank_MI_scaled, N)
toplist_scaled.append(target)
print("The top list of scaled dataset is above here:")
print(toplist_scaled)
print()

# New dataset after applying Mutual Information (feature selection). Label is appended to the dataset as well
selectedMIdata_scaled = scaled_data.loc[:, toplist_scaled]

print("After last manipulation your actual scaled dataset dimension is:", selectedMIdata_scaled.shape, "which means", selectedMIdata_scaled.shape[0], "examples x", selectedMIdata_scaled.shape[1], "columns")
print()
print('Top list MI scaled: \n')
print(selectedMIdata_scaled.head(n = 15))
print()
print("Actual attributes are these:")
print(selectedMIdata_scaled.columns)
print()

# PCA phase - notice: remove target = "Label" from independentList
print("Let's determine Principal Components List (PCA) over your dataset:")
print()

X = data.loc[:, independentList]                      
y = data[target]
pca, pcalist = pca(X)
pcaData = applyPCA(X, pca, pcalist)
# The label column was added back
pcaData.insert(loc = len(independentList), column = target, value = y, allow_duplicates = True)

print("These are the principal components from given dataset:")
print(pcaData.columns.values)
print()
print("Quick check after computation:")
print(pcaData.head())
print()
print("After last manipulation your actual dataset dimension is:", pcaData.shape, "which means", pcaData.shape[0], "examples x", pcaData.shape[1], "columns")
print()

print("Listing", N, "top important principal components from given data frame:")
pcaDataN_top = selectPCAData(pcaData, N)
print(pcaDataN_top)
print()

# Best setup for our decision tree
# Stratified K-fold cross validation phase  : notice, seed is defined on "func.py"
folds = 5                               # Number of folds for cross-validation
print("Applying Stratified K-fold cross validation to original dataset and subdivide our dataset in", folds, "folds:")
X = data.loc[:, independentList]
y = data[target]
ListXTrain, ListXTest, ListYTrain, ListYTest = stratifiedKfold(X, y, folds, seed)
print()

independent_PCA = list(pcaData.columns.values)
independent_PCA.remove(target)

Xpca = pcaData.loc[:, independent_PCA]
Ypca = pcaData[target]

ListXPCATrain, ListXPCATest, ListYPCATrain, ListYPCATest = stratifiedKfold(Xpca, Ypca, folds,
                                                                           seed)  # Stratified 5-fold cross validation on PCA train dataset
# Decision tree learner on original training set as exercise
# Passing to this instance df_clf_X that is a copy of original dataframe - this algo use C4.5 with gini as criterion by default
print()
print("Decision tree learner on original dataset submitted and plotting on video. (Close to go ahead).")
tree = decisionTreeLearner(X, y, 'entropy', 500)
showTree(tree)
print()

bestCriterionMI, bestNMI, bestEvalMI = determineDecisionTreekFoldConfiguration(ListXTrain, ListXTest, ListYTrain, ListYTest, rankMI)
print('Feature Ranking by Mutual Info:', 'Best criterion is:', bestCriterionMI, 'and best N is:', bestNMI, 'and Best CV F1 is:', bestEvalMI)
print()

# Feature evaluation PCA phase
tmp = []
for c in pcalist:
    tmp.append(1.0)
res = dict( zip(pcalist, tmp))
rankPCA = sorted( res.items(), key = lambda kv: kv[1], reverse = True)
bestCriterionPCA, bestNPCA, bestEvalPCA = determineDecisionTreekFoldConfiguration(ListXPCATrain, ListXPCATest, ListYPCATrain, ListYPCATest, rankPCA)
print('Feature Ranking by PCA:', 'Best criterion is:', bestCriterionPCA, 'and best N is:', bestNPCA, 'and Best CV F1 is:', bestEvalPCA)
print()

# Final evaluation phase - Confusion matrix and evaluation report
dataset_testing = "/dataset/Maldroid2020_test.csv"
data_test = load(dataset_testing)

if data_test is not None:
    print("Just ended to load your data from the file submitted!")
    print()
else:
    print("Got an error during loading of data provided!")
    sys.exit(0)

preElaborationClass(data_test, target)

y_true = data_test[target]

shape = data_test.shape
print("Dataset test size is:", shape, "which means", shape[0], "examples x", shape[1], "attributes")
print()

print("Just a quick test to see if you loaded correct data:")
print(data_test.head())
print()

# Training decision tree phase - Feature selection with Mutual info
data_train_MI = data.loc[:, topFeatureSelect(rankMI, bestNMI)]
print("Shape of data_train_MI:", data_train_MI.shape, "which means", data_train_MI.shape[0], "examples x", data_train_MI.shape[1], "columns")
print()
data_test_MI = data_test.loc[:, topFeatureSelect(rankMI, bestNMI)]
print("Shape of data_test_MI:", data_test_MI.shape, "which means", data_test_MI.shape[0], "examples x", data_test_MI.shape[1], "columns")
print()

treeMI = decisionTreeLearner(data_train_MI, y, bestCriterionMI, 500)

labels_MI = treeMI.classes_
pred_MI = treeMI.predict(data_test_MI)
print("Determining confusion matrix for best configuration on feature selection ranking by Mutual Info. (Close to go ahead).")
benchmark(y_true, pred_MI, labels_MI)
print()

# Training decision tree phase - Feature selection with PCA - first the PCA was applied to the testing dataset
data_test_cols = list(data_test.columns.values)
data_test_cols.remove(target)

data_test_PCA = data_test.loc[:, data_test_cols]
data_test_PCA = applyPCA(data_test_PCA, pca, pcalist)

# Feature Selection was performed on the training dataset
data_train_PCA = pcaData.loc[:, topFeatureSelect(rankPCA, bestNPCA)]
data_test_PCA = data_test_PCA.loc[:, topFeatureSelect(rankPCA, bestNPCA)]
print("Shape of data_train_PCA:", data_train_PCA.shape, "which means", data_train_PCA.shape[0], "examples x", data_train_PCA.shape[1], "columns")
print()
print("Shape of data_test_PCA:", data_test_PCA.shape, "which means", data_test_PCA.shape[0], "examples x", data_test_PCA.shape[1], "columns")
print()
treePCA = decisionTreeLearner(data_train_PCA, y, bestCriterionPCA, 500)

labels_PCA = treePCA.classes_
pred_PCA = treePCA.predict(data_test_PCA)
benchmark(y_true, pred_PCA, labels_PCA)

print("Running K-Means algorithm for separate samples in N clusters:")
print()
X_train = data.drop(target, axis = 1).values  # Remove the Class column
y_train = data[target].values

X_test = data_test.drop(target, axis = 1).values  # Remove the Class column
y_test = data_test[target].values

max_clusters = 10  # Maximum number of clusters to try
n_init = 10  # Number of times the k-means algorithm will be run with different centroid seeds

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform the Elbow method to determine optimal number of clusters
inertias = []
for num_clusters in range(1, max_clusters + 1):
    class_assignments, inertia = assign_clusters_classes(X_train_scaled, y_train, num_clusters, n_init)
    inertias.append(inertia)

# Plot the Elbow curve
plt.figure(1, figsize = (15, 6))
plt.plot(range(1, max_clusters + 1), inertias, marker = 'o')
plt.plot(range(1, max_clusters + 1), inertias, '-', alpha = 0.5)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow curve')
plt.show()

# Calculate the best K value with the silhouette index
num_clusters = np.arange(2, 10)
results = {}
for size in num_clusters:
    model = KMeans(n_clusters=size).fit(X_train_scaled)
    predictions = model.predict(X_train_scaled)
    results[size] = silhouette_score(X_train_scaled, predictions)

best_size = max(results, key=results.get)
print()
print('The best K value based on the silhouette index is: ', best_size)
print()

n_init = 10                                             # Choose the optimal number of clusters based on the Elbow curve
optimal_num_clusters = 5                                # Choose the number based on the plot

# assign clusters to classes
class_assignments, inertia = assign_clusters_classes(X_train_scaled, y_train, optimal_num_clusters, n_init)

# predict classes for testing samples
predicted_classes = predict_the_classes(X_test_scaled, X_train_scaled, y_train, class_assignments)

# benchmark predictions
confusion_mtx, classification_rep = benchmark_predictions(y_test, predicted_classes)

print("Optimal number of clusters:", optimal_num_clusters)
print("Inertia:", inertia)
print("\nClassification report:")
print(classification_rep)

n_init = 10
optimal_num_clusters = 5

# assign clusters to the classes
class_assignments, inertia = assign_clusters_classes(X_train_scaled, y_train, optimal_num_clusters, n_init)

# predict classes for testing samples
predicted_classes = predict_classes_KM(X_test_scaled, X_train_scaled, class_assignments)

# benchmark predictions
confusion_mtx, classification_rep = benchmark_predictions(y_test, predicted_classes)

print("Optimal number of clusters:", optimal_num_clusters)
print("Inertia:", inertia)
print("\nClassification report:")
print(classification_rep)