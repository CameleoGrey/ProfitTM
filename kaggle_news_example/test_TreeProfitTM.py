

from profittm.save_load import *
from pprint import pprint
from profittm.DataPreproc import DataPreproc
from profittm.TreeProfitTM import TreeProfitTM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
print("Start time: {}".format(datetime.now()))

np.random.seed(45)

articles = []
for i in range(1, 4):
    tmp = pd.read_csv("../data/articles{}.csv".format(i))
    tmp = tmp["content"].values
    articles.append(tmp)
articles = np.hstack(articles)
np.random.shuffle(articles)
trainX = articles[:int(0.95 * len(articles))]
testX = articles[int(0.95 * len(articles)):]
print(articles.shape)
save("./trainTexts.pkl", trainX)
save("./testTexts.pkl", testX)

trainX = load("./trainTexts.pkl")
testX = load("./testTexts.pkl")
#trainX = trainX[:10000]
#testX = testX[:1000]
trainX = DataPreproc().prerprocNames(trainX, removeStubStrings=False)
testX = DataPreproc().prerprocNames(testX, removeStubStrings=False)
save("./preprocTrainText.pkl", trainX)
save("./preprocTestTexts.pkl", testX)

trainX = load("./preprocTrainText.pkl")
testX = load("./preprocTestTexts.pkl")

#trainX = trainX[:10000]
#testX = testX[:1000]

topicModel = TreeProfitTM( maxDepth=2 )
topicModel.fit(trainX)
topicModel.plotTopicTree()
topicModel.save("prottm", "./treetmdir/")

trainX = trainX[:10000]
topicModel = TreeProfitTM().load("prottm", "./treetmdir/")
#topicModel.plotTopicTree()
predY = topicModel.predict(trainX, returnVectors=False)
print(predY.head(10))
predY = topicModel.predict(trainX, returnVectors=True)

##############################
encodedPreds = TSNE(n_jobs=10, verbose=True).fit_transform(predY)
plt.scatter( encodedPreds[:, 0], encodedPreds[:, 1], s=1 )
plt.show()
#############################

predY = topicModel.predict(trainX, returnVectors=False)
predY.to_csv("./predY.csv", index=False)
pprint(predY)
pprint(predY[0].isna().sum())
pprint(predY[1].isna().sum())
print("done")
