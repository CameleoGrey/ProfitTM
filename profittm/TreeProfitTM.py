from profittm.ProfitTM import ProfitTM
import numpy as np
import pandas as pd
from profittm.save_load import save, load
import networkx as nx
import uuid
from profittm.graph_draw import plotGraph
from sklearn.preprocessing import OneHotEncoder


class TreeProfitTM():

    def __init__(self, maxDepth=None, curLevel=0, parentsName=None):
        self.node = None
        self.childs = {}
        self.topicNames = None
        self.topicCount = None
        self.maxDepth = maxDepth
        self.curLevel = curLevel
        self.treeName = str(uuid.uuid4())

        if curLevel == 0 and parentsName is None:
            self.isRoot = True
        else:
            self.isRoot = False

        pass

    def fit(self, x):

        if self.curLevel == 0:
            #x = np.array(x)
            self.node = ProfitTM()
            self.node.fitTextVectorizer(x)
            self.node.cacheTextVectors(x)
            self.node.fit(x)
            self.topicNames = self.node.getTopicNames(x)
            self.topicCount = self.node.topicCount
        else:
            self.node.cacheTextVectors(x)
            self.node.fit(x)
            self.topicNames = self.node.getTopicNames(x)
            self.topicCount = self.node.topicCount

        if self.curLevel + 1 < self.maxDepth:
            y = self.node.predict(x)
            #self.node.cleanCache()
            uniqY = np.unique(y)
            topicDocs = {}
            for topic in uniqY:
                #topicDocs[topic] = x[y == topic]
                ########
                topicDocs[topic] = []
                for i in range(len(y)):
                    if y[i] == topic:
                        topicDocs[topic].append(x[i])
                ########
                self.childs[topic] = TreeProfitTM(self.maxDepth, self.curLevel + 1)
                self.childs[topic].node = ProfitTM()
                self.childs[topic].node.setVectorizer(self.node.vectorizer)
                #if self.isRoot is False:
                #    self.node.vectorizer = None

        self.node.cleanCache()

        if self.curLevel + 1 < self.maxDepth:
            for topic in topicDocs.keys():
                self.childs[topic].fit(topicDocs[topic])
        pass

    def predict(self, x, returnVectors=False):

        sharedPredicts = self.prepareToPredict(x)
        self.hPredict(sharedPredicts, predInds=None)

        if returnVectors:
            sharedPredicts = self.convertPredictsToVectors(sharedPredicts)

        return sharedPredicts

    def prepareToPredict(self, x):
        sharedPredicts = pd.DataFrame({"docs": x})
        for i in range(self.maxDepth):
            sharedPredicts[i] = None
        return sharedPredicts

    def hPredict(self, sharedPredicts, predInds=None):

        if self.curLevel < self.maxDepth:

            if predInds is None:
                predInds = sharedPredicts.index

            docs = sharedPredicts["docs"]
            y = self.node.predict(docs[predInds].values)
            sharedPredicts.iloc[predInds, self.curLevel + 1] = y
            uniqY = np.unique(y)
            for topic in uniqY:
                nextInds = docs[predInds][y == topic].index
                if len(self.childs.keys()) == 0:
                    self.leafPredict(sharedPredicts, nextInds)
                else:
                    self.childs[topic].hPredict(sharedPredicts, nextInds)
        pass

    def leafPredict(self, sharedPredicts, predInds):
        docs = sharedPredicts["docs"]
        y = self.node.predict(docs[predInds].values)
        sharedPredicts.iloc[predInds, self.curLevel + 1] = y
        pass

    def plotTopicTree(self):
        topicGraph = nx.OrderedGraph()
        self.buildTopicGraph(topicGraph)
        plotGraph(topicGraph)
        pass

    def buildTopicGraph(self, graph):

        if self.isRoot:
            for parentTopic in self.topicNames.keys():
                parentTopicName = self.extractTopicName(self.topicNames[parentTopic])
                graph.add_edge("docs", parentTopicName)

        if len(self.childs.keys()) != 0:
            for parentTopic in self.topicNames.keys():
                parentTopicName = self.extractTopicName(self.topicNames[parentTopic])
                for childTopic in self.childs[parentTopic].topicNames.keys():
                    childTopicName = self.extractTopicName(self.childs[parentTopic].topicNames[childTopic])
                    graph.add_edge(parentTopicName, childTopicName)
            for key in self.childs.keys():
                self.childs[key].buildTopicGraph(graph)
        pass

    def extractTopicName(self, topicTuple):
        topic = []
        for i in range(len(topicTuple)):
            topic.append(topicTuple[i][0])
        topic = "\n".join(topic)
        return topic

    def buildTreeGraph(self, graph):
        if len(self.childs.keys()) != 0:
            for key in self.childs.keys():
                graph.add_edge(self.treeName, self.childs[key].treeName, weight=key)
            for key in self.childs.keys():
                self.childs[key].buildTreeGraph(graph)
        pass

    def getTopicDict(self):

        sharedTopicDict = {}
        self.enrichTopicDict(sharedTopicDict)
        return sharedTopicDict

    def enrichTopicDict(self, sharedTopicDict, parentTopicID=None):
        if self.isRoot:
            for parentTopic in self.topicNames.keys():
                sharedTopicDict[parentTopic] = self.extractTopicName(self.topicNames[parentTopic])

        if len(self.childs.keys()) != 0:
            for parentTopic in self.topicNames.keys():
                for childTopic in self.childs[parentTopic].topicNames.keys():
                    if self.isRoot:
                        parentTopicID = str(parentTopic)
                    childID = str(parentTopicID) + "." + str(childTopic)
                    sharedTopicDict[childID] = self.extractTopicName(self.childs[parentTopic].topicNames[childTopic])
                    self.childs[parentTopic].enrichTopicDict(sharedTopicDict, childID)
        pass

    def getTopicVectorsDict(self):
        topicDict = self.getTopicDict()
        topicIds = np.array(list(topicDict.keys())).reshape((-1, 1))
        topicVecs = OneHotEncoder(dtype=int, sparse=False).fit_transform(topicIds)
        topicIds = topicIds.reshape((-1, ))

        topicVecsDict = {}
        for i in range(len(topicVecs)):
            topicVecsDict[topicIds[i]] = topicVecs[i]

        return topicVecsDict

    def convertPredictsToVectors(self, sharedPredicts):
        labels = np.zeros((len(sharedPredicts), self.maxDepth), dtype=int)

        labels = labels.T
        for i in range(self.maxDepth):
            labels[i] = sharedPredicts[i].values
        labels = labels.T
        print(labels[:10])

        labels = list(labels)
        for i in range(len(labels)):
            labels[i] = list(labels[i])
            for j in range(len(labels[i])):
                labels[i][j] = str(labels[i][j])
        print(labels[:10])

        ids = []
        for i in range(len(labels)):
            ids.append( ".".join(labels[i]) )
        print(ids[:10])

        topicVectorsDict = self.getTopicVectorsDict()
        vectors = []
        for i in range(len(ids)):
            vectors.append( topicVectorsDict[ids[i]] )
        vectors = np.array(vectors)
        print(vectors[:10])

        return vectors

    def save(self, name, dir):
        print("Saving whole hierarchy topic model to {}...".format(dir + name))
        treeGraph = nx.OrderedGraph()
        self.buildTreeGraph(treeGraph)
        save(dir + name + "_treegraph.pkl", treeGraph)
        vectorizer = self.node.vectorizer
        save(dir + name + "_vectorizer.pkl", vectorizer)
        self.removeVectorizers()
        self.saveTrees(name, dir)
        self.placeBackVectorizers(vectorizer)
        print("The whole hierarchy topic model saved.")
        pass

    def removeVectorizers(self):
        self.node.vectorizer = None
        if len(self.childs.keys()) != 0:
            for key in self.childs.keys():
                self.childs[key].removeVectorizers()

    def placeBackVectorizers(self, vectorizer):
        self.node.vectorizer = vectorizer
        if len(self.childs.keys()) != 0:
            for key in self.childs.keys():
                self.childs[key].node.vectorizer = vectorizer


    def saveTrees(self, name, dir):

        print("Saving tree {}".format(dir + name + "_{}".format(self.treeName)))

        self.node.save(name + "_{}".format(self.treeName), dir)
        node = self.node
        self.node = None

        childs = self.childs
        self.childs = None
        save(dir + name + "_{}_metadata.pkl".format(self.treeName), self)
        self.node = node
        self.childs = childs

        for key in self.childs.keys():
            self.childs[key].saveTrees(name, dir)

    def load(self, name, dir):

        print("Loading whole hierarchy topic model from {}...".format(dir + name))
        treeGraph = load(dir + name + "_treegraph.pkl")
        vectorizer = load(dir + name + "_vectorizer.pkl")

        trees = {}
        for node in treeGraph.nodes:
            treeName = node
            trees[node] = self.loadTree(name, dir, treeName)

        edges = list(treeGraph.edges.data("weight"))
        for edge in edges:
            trees[edge[0]].childs[edge[2]] = trees[edge[1]]

        loadedTree = None
        for key in trees.keys():
            if trees[key].isRoot:
                loadedTree = trees[key]
                break

        loadedTree.placeBackVectorizers(vectorizer)
        print("The whole hierarchy topic model loaded.")
        return loadedTree

    def loadTree(self, name, dir, treeName):

        tree = load(dir + name + "_{}_metadata.pkl".format(treeName))
        tree.node = ProfitTM()
        tree.node.load(name + "_{}".format(treeName), dir)
        tree.childs = {}
        return tree