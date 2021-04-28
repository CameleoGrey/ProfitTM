
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from profittm.save_load import save, load
from sklearn.metrics import calinski_harabasz_score
from sklearn.svm import SVC
from scipy.spatial.distance import cosine
import numpy as np
from copy import deepcopy
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from profittm.CenterLossCompressor import CenterLossCompressor


class Clusterizer():

    def __init__(self, SVM_C=1.0, n_jobs=10, verbose=1):

        self.featExtractor = CenterLossCompressor()
        self.classifier = SVC(C=SVM_C) #SVC(C=0.2)
        self.compressor = TSNE(n_jobs=n_jobs, verbose=verbose)
        self.classCount = None

        pass

    def findOptimalClusters(self, x, targetNClust, targetNClustDev, nOptSteps):

        startNClust = int( (1 - targetNClustDev) * targetNClust )
        endNClust = int( (1 + targetNClustDev) * targetNClust )
        nClust = np.linspace(startNClust, endNClust, nOptSteps, dtype=int)
        bestClusterizer = None
        bestScore = -1e20
        bestNClust = None
        for i in range(len(nClust)):
            clusterizer = AgglomerativeClustering(n_clusters=nClust[i], linkage="ward")
            clusterizer.fit(x)
            labels = clusterizer.labels_
            if len(np.unique(labels)) == 1:
                bestClusterizer = deepcopy(clusterizer)
                bestNClust = nClust[i]
                break
            score = calinski_harabasz_score(x, labels=labels)
            print("{} | n_clusters = {} | score = {}".format("Stub", nClust[i], score))
            if score > bestScore:
                bestClusterizer = deepcopy(clusterizer)
                bestScore = score
                bestNClust = nClust[i]
        print("Best score at {}: {}".format(bestNClust, bestScore))

        self.clusterizer = deepcopy(bestClusterizer)
        pass

    def fit(self, x, vectorizer, maxAggElems=30000, targetNClust=20, optParamDev=0.0, nOptSteps=1,
            batchSize=20, baseEpochs=20):

        if len(x) > maxAggElems:
            x = x[:maxAggElems]
        if len(x) <= targetNClust:
            targetNClust = 1
        if len(x) == 1:
            x = np.vstack([x, x])

        self.findOptimalClusters(x, targetNClust, optParamDev, nOptSteps)
        clustTrainY = self.clusterizer.labels_
        self.printClustDict(x, clustTrainY, vectorizer)
        clustTrainY = self.distanceBazedClusterMerge(x, clustTrainY, metric="cosine", nQuantiles=20)

        if len(np.unique(clustTrainY)) == 1:
            clustTrainY[0] = 1 + clustTrainY[1]

        tmp = []
        for i in range(len(x)):
            tmp.append(np.vstack([x[i], x[i], x[i], x[i]]))
        trainX = np.array(tmp)

        self.featExtractor.fit(trainX, clustTrainY, batchSize=batchSize, epochs=baseEpochs)
        trainX = self.featExtractor.predict(trainX)
        self.classifier.fit(trainX, clustTrainY)

        clustTrainY = self.classifier.predict(trainX)
        self.printClustDict(x, clustTrainY, vectorizer)
        clustTrainY = self.sizeBazedClusterMerge(trainX, clustTrainY, smallClustThreshold= 0.04, nQuantiles=20)
        self.printClustDict(x, clustTrainY, vectorizer)

        tmp = []
        for i in range(len(x)):
            tmp.append(np.vstack([x[i], x[i], x[i], x[i]]))
        trainX = np.array(tmp)

        if len(np.unique(clustTrainY)) == 1:
            return self

        self.classCount = len(np.unique(clustTrainY))
        self.featExtractor.fit(trainX, clustTrainY, batchSize=batchSize, epochs=baseEpochs)
        trainX = self.featExtractor.predict(trainX)
        self.classifier.fit(trainX, clustTrainY)
        pass

    def printClustDict(self, x, y, vectorizer):
        # get centers of each cluster as mean of top N words closest to center
        clustDict = {}
        x = np.array(x)
        uniqY = np.unique(y)
        centers = []
        for i in range(len(uniqY)):
            clustX = x[y == uniqY[i]]
            clustCenter = np.mean(clustX, axis=0)
            centers.append(clustCenter)
            mostSim = vectorizer.w2vModel.most_similar([clustCenter], topn=10)
            clustDict[uniqY[i]] = mostSim
        pprint(clustDict)

    def sizeBazedClusterMerge(self, x, y, smallClustThreshold = 0.04, nQuantiles=20):

        # get size threshold
        x = np.array(x)
        uniqY = np.unique(y)
        clustSizes = []
        for i in range(len(uniqY)):
            clustX = x[y == uniqY[i]]
            clustSize = len(clustX)
            clustSizes.append(clustSize)
        clustSizes = np.array(clustSizes)
        sizeQuantiles = pd.DataFrame({"clustSize": clustSizes})
        print("size quantiles")
        pprint(pd.qcut(sizeQuantiles["clustSize"], nQuantiles, duplicates="drop").value_counts().index)
        sizeQuantiles = list(sorted(list(pd.qcut(sizeQuantiles["clustSize"], nQuantiles, duplicates="drop").value_counts().index)))
        #sizeQuantiles[0].left = abs(sizeQuantiles[0].left)

        sizeThreshold = None
        relativeBorders = []
        # last max change can be at the end of sorted quantiles
        # define optimal threshold as the max change
        for i in range(len(sizeQuantiles)-1):
            relativeBorder = sizeQuantiles[i].right / abs(sizeQuantiles[i].left)
            relativeBorders.append(relativeBorder)
        # if no changes then don't merge
        if len(relativeBorders) == 0:
            return y
        maxRBInd = np.argmax(relativeBorders)
        maxRelBorder = relativeBorders[maxRBInd]
        #if there was no big change between sorted quantile sizes
        #then there are no small trash clusters
        print("Max relative border: {}".format(maxRelBorder))
        sizeThreshold = sizeQuantiles[maxRBInd].right

        # find small clusters
        mergeDict = {}
        maxClustSize = max(clustSizes)
        relativeSizes = []
        for i in range(len(uniqY)):
            clustX = x[y == uniqY[i]]
            clustSize = len(clustX)
            relativeSize = clustSize / maxClustSize
            relativeSizes.append(relativeSize)
            if clustSize <= sizeThreshold:
            #if relativeSize <= smallClustThreshold:
                mergeDict[uniqY[i]] = []

        print("relative sizes")
        pprint(relativeSizes)
        print("Max cluster size: {}".format(maxClustSize))
        # find nearest big cluster for small cluster
        for smallClustInd in mergeDict.keys():
            smallClust = x[y == smallClustInd]
            smallClustCenter = np.mean(smallClust, axis=0)
            minDist = 1e30
            bestBigInd = None
            for bigClustInd in uniqY:
                if smallClustInd == bigClustInd:
                    continue

                bigClust = x[y == bigClustInd]
                bigClustSize = len(bigClust)
                if bigClustSize <= sizeThreshold: #don't merge with other small
                    continue

                bigClustCenter = np.mean(bigClust, axis=0)
                dist = cosine(smallClustCenter, bigClustCenter)
                if dist < minDist:
                    bestBigInd = bigClustInd
            mergeDict[smallClustInd].append(bestBigInd)
        pprint(mergeDict)

        optimalY = self.mergeClusters(y, mergeDict)
        return optimalY

    def distanceBazedClusterMerge(self, x, y, metric="cosine", nQuantiles=20):

        # get centers of each cluster as mean of top N words closest to center
        x = np.array(x)
        uniqY = np.unique(y)
        centers = []
        for i in range(len(uniqY)):
            clustX = x[y == uniqY[i]]
            clustCenter = np.mean(clustX, axis=0)
            centers.append(clustCenter)
        centers = np.array(centers)

        # get distance threshold for merging
        distances = []
        for i in range(len(centers)):
            for j in range(len(centers)):
                if i == j: continue
                if metric == "cosine":
                    dist = cosine(centers[i], centers[j])
                else:
                    dist = euclidean(centers[i], centers[j])
                distances.append(dist)
        distances = pd.DataFrame({"dist": distances})
        pprint(pd.qcut(distances["dist"], nQuantiles, duplicates="drop"))
        distQuantiles = list(sorted(list(pd.qcut(distances["dist"], nQuantiles, duplicates="drop").value_counts().index)))

        #if no quantiles then don't merge
        if len(distQuantiles) == 0:
            return y

        #####################################
        distThreshold = None
        relativeBorders = []
        # last max change can be at the end of sorted quantiles
        # define optimal threshold as the max change
        for i in range(len(distQuantiles) - 1):
            relativeBorder = distQuantiles[i].right / abs(distQuantiles[i].left)
            relativeBorders.append(relativeBorder)
        # if no changes then don't merge
        if len(relativeBorders) == 0:
            return y
        maxRBInd = np.argmax(relativeBorders)
        maxRelBorder = relativeBorders[maxRBInd]
        # if there was no big change between sorted quantile sizes
        # then there are no small trash clusters
        print("Max relative distance border: {}".format(maxRelBorder))
        distThreshold = distQuantiles[maxRBInd].right
        #####################################

        #####################################
        #distThreshold = distQuantiles[0].right
        #####################################

        # get clusters which centers are closer than distance threshold
        mergeDict = {}
        for i in range(len(centers)):
            mergeDict[uniqY[i]] = []
            for j in range(len(centers)):
                if i == j: continue
                if metric == "cosine":
                    dist = cosine(centers[i], centers[j])
                else:
                    dist = euclidean(centers[i], centers[j])
                if dist <= distThreshold:
                    mergeDict[uniqY[i]].append(uniqY[j])
            if len(mergeDict[uniqY[i]]) == 0:
                mergeDict[uniqY[i]].append(-1)

        optimalY = self.mergeClusters(y, mergeDict)
        return optimalY

    def mergeClusters(self, y, mergeDict):
        # get initial merge components list
        mergeList = []
        for key in mergeDict.keys():
            if -1 not in mergeDict[key]:
                mergeComponent = []
                mergeComponent.append(key)
                for clustToMerge in mergeDict[key]:
                    mergeComponent.append(clustToMerge)
                mergeComponent = list(sorted(mergeComponent))
                mergeList.append(mergeComponent)
        uniqComponents = []
        for mergeComponent in mergeList:
            if mergeComponent not in uniqComponents:
                uniqComponents.append(mergeComponent)
        mergeList = uniqComponents
        mergeList = list(sorted(mergeList, key=len, reverse=True))
        print(mergeList)

        # clean merge list
        componentsLens = set()
        for mergeComponent in mergeList:
            componentsLens.add(len(mergeComponent))
        componentsLens = list(sorted(list(componentsLens), reverse=True))
        for k in range(len(componentsLens[:len(componentsLens)-1])):
            targetComps = []
            targetInds = []
            subComps = []
            subSetInds = []
            for i in range(len(mergeList)):
                if len(mergeList[i]) == componentsLens[k]:
                    targetComps.append(set(mergeList[i]))
                    targetInds.append(i)
                elif len(mergeList[i]) <= 1:
                    pass
                elif len(mergeList[i]) in componentsLens[k+1:]:
                    subComps.append(set(mergeList[i]))
            for i in range(len(targetComps)):
                for j in range(len(subComps)):
                    intersect = subComps[j].intersection(targetComps[i])
                    if subComps[j] == intersect:
                        subSetInds.append(j)
            subSetInds = list(set(subSetInds))
            for i in range(len(targetComps)):
                for j in range(len(subSetInds)):
                    targetComps[i] = targetComps[i].difference(subComps[subSetInds[j]])
            for i in range(len(targetInds)):
                mergeList[ targetInds[i] ] = targetComps[i]

        for i in range(len(mergeList)):
            mergeList[i] = set(mergeList[i])
        tmp = []
        for i in range(len(mergeList)):
            if mergeList[i] not in tmp and len(mergeList[i]) > 1:
                tmp.append(mergeList[i])
        mergeList = tmp
        print(mergeList)

        # get final clean components
        cleanComponents = []
        for i in range(len(mergeList)):
            curComp = set()
            for j in range(i, len(mergeList)):
                if curComp == set():
                    curComp = curComp.union(mergeList[j])
                else:
                    if curComp.intersection(mergeList[j]) != set():
                        curComp = curComp.union(mergeList[j])
            notIn = True
            for i in range(len(cleanComponents)):
                if curComp.intersection(cleanComponents[i]) != set():
                    notIn = False
            if notIn:
                cleanComponents.append(curComp)
        print(cleanComponents)

        # set new labels
        for optSetY in cleanComponents:
            minY = min(optSetY)
            for i in range(len(y)):
                if y[i] in optSetY:
                    y[i] = minY
        uniqY = np.unique(y)
        oldNewDict = {}
        for i in range(len(uniqY)):
            oldNewDict[uniqY[i]] = i
        for i in range(len(y)):
            y[i] = oldNewDict[y[i]]
        optimalY = y

        uniqY = np.unique(optimalY)
        print("Optimal profittm = {}".format(len(uniqY)))

        return optimalY


    def get_features(self, x):
        estimates = []
        for i in range(len(x)):
            estimates.append(np.vstack([x[i], x[i], x[i], x[i]]))
        estimates = np.array(estimates)
        estimates = self.featExtractor.predict(estimates)
        return estimates

    def get_class_estimates(self, x):
        features = self.get_features(x)
        estimates = self.classifier.decision_function(features)
        return estimates

    def predict(self, x):

        estimates = self.get_features(x)
        predY = self.classifier.predict(estimates)

        return predY

    def compressFeatures(self, x):

        compressedX = self.compressor.fit_transform(x)

        return compressedX

    def getTopicNames(self, x, vectorizer):

        x = np.array(x)
        topicDict = {}
        y = self.predict(x)
        uniqY = np.unique(y)
        centers = []
        for i in range(len(uniqY)):
            clustX = x[y == uniqY[i]]
            clustCenter = np.mean(clustX, axis=0)
            centers.append(clustCenter)
            mostSim = vectorizer.w2vModel.most_similar([clustCenter], topn=5)
            topicDict[uniqY[i]] = mostSim

        return topicDict


    def drawDists(self, x, y, vectorizer):
        clustDict = {}
        x = np.array(x)
        uniqY = np.unique(y)
        centers = []
        for i in range(len(uniqY)):
            clustX = x[y == uniqY[i]]
            clustCenter = np.mean(clustX, axis=0)
            centers.append(clustCenter)
            mostSim = vectorizer.w2vModel.most_similar([clustCenter], topn=10)
            clustDict[uniqY[i]] = mostSim
        pprint(clustDict)
        centers = np.array(centers)

        distances = []
        for i in range(len(centers)):
            for j in range(len(centers)):
                if i == j: continue
                dist = cosine(centers[i], centers[j])
                distances.append(dist)
        print(len(distances))
        distances = pd.DataFrame({"dist": distances})
        pprint(pd.qcut(distances["dist"], 20))

        compressedX = []
        minDistLabels = []
        for i in range(len(centers)):
            minDist = 1e30
            min_j = -1
            for j in range(len(centers)):
                if i == j: continue
                dist = cosine(centers[i], centers[j])
                if dist < minDist:
                    minDist = dist  # + 0.1 * np.random.standard_normal(1)[0] * dist #noise is only for visualization
                    min_j = j
            compressedX.append([minDist, minDist])
            minDistLabels.append(str(uniqY[i]) + " | " + str(uniqY[min_j]))
        compressedX = np.array(compressedX)

        for i in range(len(minDistLabels)):
            plt.text(compressedX[i, 0], compressedX[i, 1], minDistLabels[i], fontsize=12)
            plt.scatter(compressedX[i, 0], compressedX[i, 1], s=1)
        plt.show()

        optimalY = None
        return optimalY

    def plotClusters(self, x):

        featX = self.get_features(x)
        #featX = self.get_class_estimates(x)
        predY = self.predict(x)

        compressedX = self.compressFeatures(featX)
        uniqLabels = np.unique(predY)
        for i in uniqLabels:
            #for x_, y_ in zip(compressedX[predY == i, 0], compressedX[predY == i, 1]):
            #    plt.text(x_, y_, str(i), fontsize=8)
            plt.scatter(compressedX[predY == i, 0], compressedX[predY == i, 1], s=1)
            plt.scatter(np.mean(compressedX[predY == i, 0]), np.mean(compressedX[predY == i, 1]), s=50, c="r")
            plt.text(np.mean(compressedX[predY == i, 0]), np.mean(compressedX[predY == i, 1]), str(i), fontsize=15, c="r")

        plt.show()

        pass

    def save(self, name, path):
        save(path + name + ".pkl", self)
        pass

    def load(self, name, path):
        loadedClusterizer = load(path + name + ".pkl")
        return loadedClusterizer