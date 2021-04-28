from pprint import pprint
from profittm.ProfitTM import ProfitTM
from profittm.DataPreproc import DataPreproc
from profittm.save_load import *
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

fitVectTexts = trainX
topicModel = ProfitTM(n_jobs=10, verbose=1)
topicModel.fitTextVectorizer(fitVectTexts, size=100, window=5, n_jobs=10, min_count=2, sample=0.00001, iter=10, sg=0, seed=45)
topicModel.cacheTextVectors(trainX)
topicModel.fit(trainX, maxAggElems=10000, targetNClust=20, optParamDev=0.0, nOptSteps=1,
            batchSize=100, baseEpochs=20, taskList=[None])
topicModel.save("protm", "./")

topicModel = ProfitTM().load("protm", "./")
topicDict = topicModel.getTopicNames()
pprint(topicDict)
topicModel.plotClusters(maxPoints=10000)

topicModel.cleanCache()
topicModel.plotClusters(testX, maxPoints=10000)
print("done")
