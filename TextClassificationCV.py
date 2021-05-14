import pandas as pd

#Read the pre-processed train csv file
trainDataFrame = pd.read_csv(r"C:\Masters\DKE\Job\Scorable\TextClassification\PreProcess_train.csv", sep='|')

'Read text & class label of pre-processed training data'
X_train, y_train_label = trainDataFrame['title'], trainDataFrame['class']
print()

'2. Bag of words model' \
' with count vectorizer and corresponding feature selection'
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

#Extract all features from training data and represent words from test data using these extracted features
def featureExtraction(X_train, X_val):

    # tfIdfVectorizer = TfidfVectorizer()

    # Convert values to unicode string as per fit transform documentation
    # X_trainFeatures = tfIdfVectorizer.fit_transform(X_train.values.astype('U'))

    # X_valFeatures = tfIdfVectorizer.transform(X_val.values.astype('U'))

    # Word counts are considered
    countVectorizer = CountVectorizer()
    # # Convert values to unicode string as per fit transform documentation
    X_trainFeatures = countVectorizer.fit_transform(X_train.values.astype('U'))
    #
    X_valFeatures = countVectorizer.transform(X_val.values.astype('U'))

    # print(countVectorizer.get_feature_names())
    # print(countVectorizer.vocabulary_)

    # print(X_trainFeatures)
    # print(X_valFeatures)

    # How many features are present?
    # print("Length/Size of the vocabulary {}".format(len(countVectorizer.vocabulary_)))
    # print("Shape of train features", X_trainFeatures.shape)
    # print("Shape of test features", X_valFeatures.shape)
    return X_trainFeatures, X_valFeatures

def featureSelection(X_trainFeatures, X_valFeatures, y_train_label):
    print("Feature Selection")
    'Since the vocabulary is approximately of length 34000 after pre-processing, selecting 10% of the features'
    chiSquareFS = SelectKBest(chi2, k=3400)
    # chiSquareFS = SelectKBest(chi2, k=1000)

    #Selecting the best 3400 features on the training data and representing
    # Fit-transform on train data
    X_trainFeaturesFS = chiSquareFS.fit_transform(X_trainFeatures, y_train_label)
    print(X_trainFeaturesFS.shape)

    # Transform on test data
    X_valFeaturesFS = chiSquareFS.transform(X_valFeatures)
    print(X_valFeaturesFS.shape)
    print()

    return X_trainFeaturesFS, X_valFeaturesFS

#Method that performs both featureExtraction and featureSelection
def featureExtractionAndSelection(X_train, X_val,y_train_label):
    # Extract all features from training data and represent words from test data using these extracted features
    X_trainFeatures, X_valFeatures = featureExtraction(X_train, X_val)
    # Reduce the high dimensional space by performing feature selection
    X_trainFeaturesFS, X_valFeaturesFS = featureSelection(X_trainFeatures, X_valFeatures, y_train_label)
    # Returning all features in case they are required later
    return X_trainFeaturesFS, X_valFeaturesFS, X_trainFeatures, X_valFeatures

'3. Perform k-fold CV to determine average training accuracy'
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
import joblib

from pathlib import Path

#Create a directory to save models
Path("ModelsCV").mkdir(parents=True, exist_ok=True)

allStdAccuracyResults = []
allBalAccuracyResults = []

#Baseline clasifier that generates majority label predictions based on class distribution CV (Majority-stratified)
def baselineCV(X_train_CV_FeaturesFS, y_train_label_CV, X_val_CV_FeaturesFS, y_val_label_CV, foldIndex):
    baseline = DummyClassifier(strategy='stratified')
    baseline.fit(X_train_CV_FeaturesFS, y_train_label_CV)
    # predict the labels on validation dataset
    y_val_label_predicted_CV = baseline.predict(X_val_CV_FeaturesFS)
    #Compute standard accuracy
    print("Baseline Accuracy Score with FS-> ", accuracy_score(y_val_label_predicted_CV, y_val_label_CV) * 100)
    allStdAccuracyResults.append(accuracy_score(y_val_label_predicted_CV, y_val_label_CV) * 100)
    #Compute balanced accuracy
    print("Baseline Balanced Accuracy Score with FS-> ",balanced_accuracy_score(y_val_label_predicted_CV, y_val_label_CV)*100)
    allBalAccuracyResults.append(balanced_accuracy_score(y_val_label_predicted_CV, y_val_label_CV) * 100)
    # Save the model
    joblib.dump(baseline,r"ModelsCV\baselineCV"+ str(foldIndex) +".pkl")

#Linear SVM CV
def linearSVMCV(X_train_CV_FeaturesFS,y_train_label_CV,X_val_CV_FeaturesFS,y_val_label_CV,foldIndex):
    #Penalize misclassification of minority classes using class weight and since number of samples > number of features dual should be false.
    linearSVM = LinearSVC(dual = False, class_weight='balanced')
    linearSVM.fit(X_train_CV_FeaturesFS, y_train_label_CV)
    # predict the labels on validation dataset
    y_val_label_predicted_CV = linearSVM.predict(X_val_CV_FeaturesFS)

    # Compute standard accuracy
    print("Linear SVM Accuracy Score with FS-> ", accuracy_score(y_val_label_predicted_CV, y_val_label_CV) * 100)
    allStdAccuracyResults.append(accuracy_score(y_val_label_predicted_CV, y_val_label_CV) * 100)
    # Compute balanced accuracy
    print("Linear SVM Balanced Accuracy Score with FS-> ",
          balanced_accuracy_score(y_val_label_predicted_CV, y_val_label_CV) * 100)
    allBalAccuracyResults.append(balanced_accuracy_score(y_val_label_predicted_CV, y_val_label_CV) * 100)
    # Save the model
    joblib.dump(linearSVM,r"ModelsCV\LinearSVMCV"+ str(foldIndex) +".pkl")

#Random Forest CV
def randomForestCV(X_train_CV_FeaturesFS,y_train_label_CV,X_val_CV_FeaturesFS,y_val_label_CV,foldIndex):
    randomForest = RandomForestClassifier(random_state=50, class_weight='balanced_subsample')
    # randomForest = RandomForestClassifier(random_state=50, class_weight='balanced')
    randomForest.fit(X_train_CV_FeaturesFS, y_train_label_CV)
    # predict the labels on validation dataset
    y_val_label_predicted_CV = randomForest.predict(X_val_CV_FeaturesFS)
    # Compute standard accuracy
    print("Random Forest Accuracy Score with FS-> ",accuracy_score(y_val_label_predicted_CV, y_val_label_CV)*100)
    allStdAccuracyResults.append(accuracy_score(y_val_label_predicted_CV, y_val_label_CV)*100)
    # Compute balanced accuracy
    print("Random Forest Balanced Accuracy Score with FS-> ",balanced_accuracy_score(y_val_label_predicted_CV, y_val_label_CV)*100)
    allBalAccuracyResults.append(balanced_accuracy_score(y_val_label_predicted_CV, y_val_label_CV)*100)
    # Save the model
    joblib.dump(randomForest,r"ModelsCV\RandomForestCV"+ str(foldIndex) +".pkl")


import matplotlib.pyplot as plt
'Visualization'

def visualizeCVResults(allStdAccuracyResults, allBalAccuracyResults, classifierName):
    indices = np.arange(len(allBalAccuracyResults))
    FoldNames = ["Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"]

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy depiction for different folds of "+classifierName)
    #Respective bars to indicate accuracy
    plt.barh(indices, allBalAccuracyResults, .2, label="bal_accuracy", color='yellow')
    plt.barh(indices + 0.3, allStdAccuracyResults, .2, label="accuracy", color='c')
    plt.yticks(())
    # plt.locator_params(axis='y', nbins = 10)
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel("Score")
    plt.ylabel("Folds")
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    for i, c in zip(indices, FoldNames):
        #To control spacing
        plt.text(-.6, i, c)
    plt.savefig(r"ModelsCV\\" +classifierName+".png")
    plt.show()



def startifiedKFoldCV(allStdAccuracyResults, allBalAccuracyResults,foldIndex):
    for train_index, val_index in strat_Kfold.split(X_train, y_train_label):
        X_train_CV, X_val_CV = X_train[train_index], X_train[val_index]
        y_train_label_CV, y_val_label_CV = y_train_label[train_index], y_train_label[val_index]

        X_train_CV_FeaturesFS, X_val_CV_FeaturesFS, X_train_CV_Features, X_val_CV_Features = featureExtractionAndSelection(
            X_train_CV,
            X_val_CV,
            y_train_label_CV)

        'Uncomment the below line here based on the choice of the classifier for which CV has to be run'
        # baselineCV(X_train_CV_FeaturesFS, y_train_label_CV, X_val_CV_FeaturesFS, y_val_label_CV, foldIndex)
        # linearSVMCV(X_train_CV_FeaturesFS, y_train_label_CV, X_val_CV_FeaturesFS,y_val_label_CV,foldIndex)
        randomForestCV(X_train_CV_FeaturesFS, y_train_label_CV, X_val_CV_FeaturesFS,y_val_label_CV,foldIndex)

        foldIndex = foldIndex + 1

    allStdAccuracyResults = np.array(allStdAccuracyResults)
    print('Mean accuracy: ', np.mean(allStdAccuracyResults, axis=0))
    print('StdDev for accuracy: ', np.std(allStdAccuracyResults, axis=0))

    allBalAccuracyResults = np.array(allBalAccuracyResults)
    print('Mean balanced accuracy: ', np.mean(allBalAccuracyResults, axis=0))
    print('StdDev for balanced accuracy: ', np.std(allBalAccuracyResults, axis=0))

    'Uncomment the below line here based on the choice of the classifier for which visualizations have to be made'
    # visualizeCVResults(allStdAccuracyResults, allBalAccuracyResults, "baseline")
    # visualizeCVResults(allStdAccuracyResults, allBalAccuracyResults, "LinearSVM")
    visualizeCVResults(allStdAccuracyResults, allBalAccuracyResults, "RandomForest")


foldIndex = 0
#Perform 5-fold Stratified CV
strat_Kfold = StratifiedKFold(n_splits=5, shuffle=True)
startifiedKFoldCV(allStdAccuracyResults, allBalAccuracyResults, foldIndex)





