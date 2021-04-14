All executions were performed on Python 3.7 on a Windows 10 machine.

For execution on another system, harcoded paths have to be changed in all the python files.
These hardcoded paths majorly correspond to reading the csv files to the dataframe.
e.g., trainDataFrame = pd.read_csv(r"C:\Masters\DKE\Job\Scorable\nlp_model_task\nlp_model_task\train.csv", sep='|')

TextClassification.py and TextClassificationCV.py need pre-processed train and test csv files.
Hence PreProcess.py has to be executed before the aforementioned files are run.

Yardstick 1 is evaluated in TextClassification.py and Yardstick 2 in TextClassificationCV.py.
All cross validated models appear in the ModelsCV folder.

The RandomForest.pkl model has been built on the entire training data and then used for prediction on the test set.
This execution is performed in the TextClassification.py program.

Please note random forest generally takes more time than SVM and the baseline to run.

In TextClassificationCV.py correponding classifier choice and visualization choice have to be uncommented.

e.g., For running 5-fold CV for RandomForest and visualizing its results, the corresponding code below should be uncommented in the TextClassificationCV.py file.
'Uncomment the below line here based on the choice of the classifier for which CV has to be run'
# baselineCV(X_train_CV_FeaturesFS, y_train_label_CV, X_val_CV_FeaturesFS, y_val_label_CV, foldIndex)
# linearSVMCV(X_train_CV_FeaturesFS, y_train_label_CV, X_val_CV_FeaturesFS,y_val_label_CV,foldIndex)
randomForestCV(X_train_CV_FeaturesFS, y_train_label_CV, X_val_CV_FeaturesFS,y_val_label_CV,foldIndex)

'Uncomment the below line here based on the choice of the classifier for which visualizations have to be made'
# visualizeCVResults(allStdAccuracyResults, allBalAccuracyResults, "baseline")
# visualizeCVResults(allStdAccuracyResults, allBalAccuracyResults, "LinearSVM")
visualizeCVResults(allStdAccuracyResults, allBalAccuracyResults, "RandomForest")


