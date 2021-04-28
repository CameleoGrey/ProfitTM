

from profittm.TfidfW2vVectorizer import TfidfW2vVectorizer
from profittm.Clusterizer import Clusterizer
from profittm.save_load import save, load
import gc

class ProfitTM():

    def __init__(self, n_jobs=10, verbose=1, name=None):
        self.vectorizer = TfidfW2vVectorizer()
        self.clusterizer = Clusterizer(SVM_C=1.0, n_jobs=n_jobs, verbose=verbose)
        self.cachedVectors = None
        self.useCacheVectors = False
        self.n_jobs = n_jobs
        self.topicCount = None
        self.name = name
        pass

    def fitTextVectorizer(self, corpus, size=100, window=5, n_jobs=10, min_count=2, sample=1e-5, iter=10, sg=0, seed=45):
        self.vectorizer = TfidfW2vVectorizer()
        self.vectorizer.fitTfidf(corpus)
        self.vectorizer.makeW2VDict(corpus, size=size, window=window, n_jobs=n_jobs,
                                    min_count=min_count, sample=sample, iter=iter, sg=sg, seed=seed)
        self.cachedVectors = None
        self.useCacheVectors = False
        return self

    def setVectorizer(self, vectorizer):
        self.vectorizer = vectorizer
        pass

    def cacheTextVectors(self, x):
        self.cachedVectors = self.vectorizer.vectorizeDocsMulticore(x, n_jobs=self.n_jobs, useTfidf=True)
        #self.cachedVectors = self.vectorizer.vectorizeDocs(x, useTfidf=True)
        self.useCacheVectors = True
        print("Text vectors cached.")
        return self.cachedVectors

    def cleanCache(self):
        self.cachedVectors = None
        self.useCacheVectors = False
        gc.collect()
        print("Cache cleared.")
        pass

    def chooseX(self, x):
        if self.useCacheVectors or x is None:
            x = self.cachedVectors
        else:
            x = self.vectorizer.vectorizeDocsMulticore(x, n_jobs=self.n_jobs, useTfidf=True)
            #x = self.vectorizer.vectorizeDocs(x, useTfidf=True)
        return x

    def fit(self, x=None, maxAggElems=10000, targetNClust=20, optParamDev=0.0, nOptSteps=1,
            batchSize=100, baseEpochs=40):

        if self.vectorizer.isFitted() is False:
            raise ValueError("Text vectorizer is not fitted.")

        x = self.chooseX(x)
        self.clusterizer.fit(x, vectorizer=self.vectorizer, maxAggElems=maxAggElems,
                             targetNClust=targetNClust, optParamDev=optParamDev,nOptSteps=nOptSteps,
                             batchSize=batchSize, baseEpochs=baseEpochs)
        self.topicCount = self.clusterizer.classCount
        pass

    def predict(self, x=None):
        x = self.chooseX(x)
        y = self.clusterizer.predict(x)
        return y

    def get_features(self, x=None):
        x = self.chooseX(x)
        estimates = self.clusterizer.get_features(x)
        return estimates

    def get_class_estimates(self, x=None):
        x = self.chooseX(x)
        estimates = self.clusterizer.get_class_estimates(x)
        return estimates

    def getTopicNames(self, x=None):
        x = self.chooseX(x)
        topicDict = self.clusterizer.getTopicNames(x, self.vectorizer)
        return topicDict

    def drawDists(self, x=None):
        x = self.chooseX(x)
        y = self.clusterizer.predict(x)
        self.clusterizer.drawDists(x, y, self.vectorizer)
        pass

    def plotClusters(self, x=None, maxPoints=10000):
        x = self.chooseX(x)

        if len(x) > maxPoints:
            x = x[:maxPoints]

        self.clusterizer.plotClusters(x)
        pass

    def save(self, name, tmDir):
        self.clusterizer.save( "clusterizer_" + name, tmDir)
        tmp = self.clusterizer
        self.clusterizer = None
        save(tmDir + "vectorizer_" + name + ".pkl", self)
        self.clusterizer = tmp
        pass

    def load(self, name, tmDir):
        loadedTM = load(tmDir + "vectorizer_" + name + ".pkl")
        loadedTM.clusterizer = self.clusterizer.load("clusterizer_" + name, tmDir)

        self.clusterizer = loadedTM.clusterizer
        self.vectorizer = loadedTM.vectorizer
        self.cachedVectors = loadedTM.cachedVectors
        self.useCacheVectors = loadedTM.useCacheVectors

        return self