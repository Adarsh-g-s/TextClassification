import pandas as pd
'Text classification using doc2vec'

'1. Load the pre-split dataset'
'2. Doc2Vec representation of the documents'
'3. Train the ML algo and test on the test set'

'1. Load the pre-split dataset'
#Read the pre-processed csv files
trainDataFrame = pd.read_csv(r"C:\Masters\DKE\Job\Scorable\TextClassification\PreProcess_train.csv", sep='|')
testDataFrame = pd.read_csv(r"C:\Masters\DKE\Job\Scorable\TextClassification\PreProcess_test.csv", sep='|', header=None)

'Read text & class label of training data'
X_train, y_train_label = trainDataFrame['title'], trainDataFrame['class']
print()

#Convert the train docs to a list, read them as string else an error appears asking to convert float to string
X_train = trainDataFrame.title.astype(str)
X_train = X_train.tolist()

'Read text of testing data'
X_test = testDataFrame[0]
X_test = X_test.tolist()

print()

'2. Doc2Vec representation'
from gensim.test.utils import common_texts, simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from tqdm import tqdm

'Both the documents and the class labels have to be tagged'
#Tag train set, convert the series to a list
X_data_tagged_train = [TaggedDocument(words=simple_preprocess(doc.lower()),
tags=[str(i)]) for i, doc in enumerate(X_train)]
# tags=[str(i)]) for i, doc in X_train.itertuples()]
# tags=[str(i)]) for i, doc in tqdm(X_train)]
# tags=[str(i)]) for i, doc in enumerate(X_train)]
# tags=[str(i)]) for i, _d in enumerate(X_data_train)]

print(X_data_tagged_train[1200])
print(len(X_data_tagged_train))

#Tag test set
X_data_tagged_test = [TaggedDocument(words=simple_preprocess(_d),
tags=[str(i)]) for i, _d in enumerate(X_test)]

print(len(X_data_tagged_test))
print()

'Train a doc2vec model on the training dataset'

def trainDoc2VecModel(X_data_tagged_train):
    #Each document will be represented by a vector of length 10.
    doc2VecModel = Doc2Vec(X_data_tagged_train, vector_size=10, min_count=3, epochs=20)
    return doc2VecModel

doc2VecModel = trainDoc2VecModel(X_data_tagged_train)
# doc2VecModel.build_vocab([item for item in tqdm(X_data_tagged_train)])
print(f"Word 'arnold' appeared {doc2VecModel.wv.get_vecattr('australian', 'count')} times in the training corpus.")
# doc2VecModel.train(X_data_tagged_train, total_examples=doc2VecModel.corpus_count, epochs=doc2VecModel.epochs)

vector = doc2VecModel.infer_vector(['symantec', 'set', 'acquire', 'security', 'consultant', 'stake'])
print(vector)

'Since we already have the tagged representation of the training and test set we can use it directly for classification'

from sklearn.ensemble import RandomForestClassifier

randomForest = RandomForestClassifier()

X_train_doc2Vec = []

for row in X_data_tagged_train:
    print(row)
    print(row.words)
    modelTrainVector = doc2VecModel.infer_vector(row.words)
    X_train_doc2Vec.append(modelTrainVector)

randomForest.fit(X_train_doc2Vec, y_train_label)

X_test_doc2Vec = []

for row in X_data_tagged_test:
    modelTestVector = doc2VecModel.infer_vector(row.words)
    X_test_doc2Vec.append(modelTestVector)

y_pred = randomForest.predict(X_test_doc2Vec)

# from sklearn.metrics import classification_report
#
# print(classification_report(y_classLabel_test, y_pred))

originalTestDataFrame = pd.read_csv(r"C:\Masters\DKE\Job\Scorable\TextClassification\Dataset\test.csv", sep='|', header=None)
originalTestDataFrame['predicted_class'] = y_pred

#Drop column 0 as it has indexes
originalTestDataFrame = originalTestDataFrame.drop(columns=[0])
originalTestDataFrame.to_csv('test_predicted_doc2vec.csv', index=False, sep='|', header=None)