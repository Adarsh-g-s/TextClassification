
import pandas as pd
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer


#Read the csv files
trainDataFrame = pd.read_csv(r"./Dataset/train.csv", sep='|')
testDataFrame = pd.read_csv(r"./Dataset/test.csv", sep='|', header=None)

'1. Perform pre-processing of the text in the data frame'
def removeSpecialCharsAndLowerCase(instance):
    #Remove all special characters in the row
    cleanedInstance = re.sub(r'\W', ' ', instance)

    # Convert string to lowercase
    cleanedInstance = cleanedInstance.lower()
    # print(pre_X_doc)
    return cleanedInstance


def tokenizeAndRemoveStopwords(instance):
    tokenizedInstance = word_tokenize(instance)
    cleanedInstance = [item for item in tokenizedInstance if not item in stopwordList]
    cleanedInstance = ' '.join(cleanedInstance)
    return cleanedInstance


def getLemma(instance):
    instance = instance.split()
    instance = [lemmatizer.lemmatize(token) for token in instance]
    instance = ' '.join(instance)
    return instance

def performPreProcess(row):
    #Pre-processing each row
    #Make this row thingy a parameter in your function
    if(flag==True):
        #Current file is a test dataframe, therefore it has a number as header.
        instance = row[1]
    else:
        instance = row['title']
    # instance = row[1]
    # instance = row['title']
    print("Original Instance", instance)
    instance = removeSpecialCharsAndLowerCase(instance)
    instance = tokenizeAndRemoveStopwords(instance)
    instance = getLemma(instance)
    print("Cleaned Instance", instance)
    return instance



stopwordList = set(stopwords.words('english'))

#Adding few more words to stopword list
stopwordList.add('u')
stopwordList.add('lt')
stopwordList.add('b')
stopwordList.add('gt')

lemmatizer = WordNetLemmatizer()

'Perform pre-processing on each row of the train data frame'
#Flag to track if the dataframe passed has a header or not. Train dataframe has a header 'title'
flag=False
trainDataFrame['title'] = trainDataFrame.apply(performPreProcess, axis=1)
#Saving the results to a csv file as backup
trainDataFrame.to_csv(r"./Dataset/PreProcess/PreProcess_train.csv", index=False, sep='|')
# trainDataFrame.to_csv('PreProcess_train.csv', index=False, sep='|')

'Perform pre-processing on each row of the test data frame'
#Test dataframe does not have a header
flag=True
testDataFrame = testDataFrame.apply(performPreProcess, axis=1)
#Saving the results to a csv file as backup
testDataFrame.to_csv(r"./Dataset/PreProcess/PreProcess_test.csv", index=False, sep='|', header=None)
# testDataFrame.to_csv('PreProcess_test.csv', index=False, sep='|', header=None)




