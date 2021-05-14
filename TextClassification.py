import pandas as pd

#Read the pre-processed csv files
trainDataFrame = pd.read_csv(r"C:\Masters\DKE\Job\Scorable\TextClassification\PreProcess_train.csv", sep='|')
testDataFrame = pd.read_csv(r"C:\Masters\DKE\Job\Scorable\TextClassification\PreProcess_test.csv", sep='|', header=None)

'Read text & class label of training data'
X_train, y_train_label = trainDataFrame['title'], trainDataFrame['class']
print()

'Read text of testing data'
X_test = testDataFrame[0]

'2. Bag of words model' \
'Count vectorizer and corresponding feature selection'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

def featureExtraction(X_train, X_test):
    # Word counts are considered
    countVectorizer = CountVectorizer()

    # Convert values to unicode string as per fit transform documentation
    X_trainFeatures = countVectorizer.fit_transform(X_train.values.astype('U'))

    X_testFeatures = countVectorizer.transform(X_test.values.astype('U'))

    # print(countVectorizer.get_feature_names())
    # print(countVectorizer.vocabulary_)

    # print(X_trainFeatures)
    # print(X_testFeatures)

    # How many features are present?
    print("Length/Size of the vocabulary {}".format(len(countVectorizer.vocabulary_)))
    print("Shape of train features", X_trainFeatures.shape)
    print("Shape of test features", X_testFeatures.shape)
    return X_trainFeatures, X_testFeatures

def featureSelection(X_trainFeatures, X_testFeatures, y_train_label):
    print("Feature Selection")
    chiSquareFS = SelectKBest(chi2, k=3400)
    # chiSquareFS = SelectKBest(chi2, k=1000)

    # Fit-transform on train data
    X_trainFeaturesFS = chiSquareFS.fit_transform(X_trainFeatures, y_train_label)
    print(X_trainFeaturesFS.shape)

    # Transform on test data
    X_testFeaturesFS = chiSquareFS.transform(X_testFeatures)
    print(X_testFeaturesFS.shape)
    print()
    return X_trainFeaturesFS, X_testFeaturesFS

def featureExtractionAndSelection(X_train, X_test,y_train_label):

    X_trainFeatures, X_testFeatures = featureExtraction(X_train, X_test)
    X_trainFeaturesFS, X_testFeaturesFS = featureSelection(X_trainFeatures, X_testFeatures, y_train_label)
    return X_trainFeaturesFS, X_testFeaturesFS, X_trainFeatures, X_testFeatures

X_trainFeaturesFS, X_testFeaturesFS, X_trainFeatures, X_testFeatures = featureExtractionAndSelection(X_train, X_test,y_train_label)
print()

'3. Train the ML algorithm on training data and compute training accuracy'
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, classification_report

def mlAlgorithm(classifier, features, classifierName):
    classifier.fit(features, y_train_label)
    y_train_predicted = classifier.predict(features)
    print("Training accuracy of "+classifierName+ " is {}".format(accuracy_score(y_train_predicted, y_train_label) * 100))
    #As the dataset is imbalanced, balanced accuracy has to be used
    print("Training balanced accuracy of "+classifierName+ " is {}".format(balanced_accuracy_score(y_train_predicted, y_train_label) * 100))
    bal_accuracy = balanced_accuracy_score(y_train_predicted, y_train_label) * 100
    #To list out precision, recall etc
    print(classification_report(y_train_predicted, y_train_label))
    plot_confusion_matrix(classifier, features, y_train_label)
    plt.title("Confusion matrix for "+classifierName)
    plt.savefig(classifierName+".png")
    #Save the model
    joblib.dump(classifier,classifierName+".pkl")
    return bal_accuracy, classifierName

np.random.seed(0)

results = {}
# #Baseline majority-startified classifier
bal_accuracy, classifierName = mlAlgorithm(DummyClassifier(strategy='stratified'), X_trainFeaturesFS, "Baseline")

results[classifierName] = bal_accuracy
#
#Linear SVM Classifier
bal_accuracy, classifierName = mlAlgorithm(LinearSVC(dual = False,class_weight='balanced'), X_trainFeaturesFS, "LinearSVM")

results[classifierName] = bal_accuracy
#
# #Random Forest Classifier
bal_accuracy, classifierName = mlAlgorithm(RandomForestClassifier(class_weight='balanced_subsample'), X_trainFeaturesFS, "RandomForest")
# bal_accuracy, classifierName = mlAlgorithm(RandomForestClassifier(class_weight='balanced'), X_trainFeaturesFS, "RandomForest")
#
results[classifierName] = bal_accuracy

print()

'4. Predict labels for the test set'

#Load the saved random forest model
randomForestClassifier = joblib.load("RandomForest.pkl")

#Get the labels for the test set features
y_test_predicted = randomForestClassifier.predict(X_testFeaturesFS)

#Conver nd array to series, so as to easily add it as a column to the dataframe.
y_test_predicted = pd.Series(y_test_predicted)

print()

#Write it back to the data frame and dump it to the original csv file
originalTestDataFrame = pd.read_csv(r"C:\Masters\DKE\Job\Scorable\TextClassification\Dataset\test.csv", sep='|', header=None)
originalTestDataFrame['predicted_class'] = y_test_predicted

#Drop column 0 as it has indexes
originalTestDataFrame = originalTestDataFrame.drop(columns=[0])
originalTestDataFrame.to_csv('test_predicted.csv', index=False, sep='|', header=None)