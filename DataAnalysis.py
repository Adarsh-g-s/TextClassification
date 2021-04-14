
# Read the train file onto the data frame, find the class distribution in the train file

import pandas as pd
import matplotlib.pyplot as plt

trainDataFrame = pd.read_csv(r"C:\Masters\DKE\Job\Scorable\nlp_model_task\nlp_model_task\train.csv", sep='|')

print()

# Find the class distribution
def classDistribution(trainDataFrame):
    # Plot to uncover the number of instances associated with each class
    distribution = trainDataFrame['class'].value_counts().plot.bar()
    distribution.plot()
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.suptitle("Class distribution")
    plt.savefig("Class Distribution.png")
    plt.show()

classDistribution(trainDataFrame)



