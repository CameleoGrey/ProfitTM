from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from profittm.save_load import save, load
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
import multiprocessing as mp
import os
import gc

def identity_tokenizer(text):
    return text

class TfidfW2vVectorizer():
    def __init__(self):
        self.w2vModel = None
        self.w2vDict = None
        self.tfidfVectorizer = TfidfVectorizer() #kaggle_all_the_news
        #self.tfidfVectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, stop_words=None)  # kaggle_survey_2020
        pass

    def makeW2VDict(self, docs, size=128, window=5, n_jobs=10, min_count=1, sample=0, iter=100, sg=0, seed=45):

        #docs = deepcopy(docs)
        docs = list(docs)
        for i in range(len(docs)):
            docs[i] = docs[i].split()

        self.w2vModel = Word2Vec(docs, size=size, window=window, workers=n_jobs, min_count=min_count, sample=sample, iter=iter, sg=sg, seed=seed)
        self.w2vDict = dict(zip(self.w2vModel.wv.index2word, self.w2vModel.wv.syn0))
        docs = None
        gc.collect()
        pass

    def isFitted(self):
        if self.w2vModel is None:
            return False
        return True

    def fitTfidf(self, docs):
        print("Fitting tfidf vectorizer")
        self.tfidfVectorizer.fit(docs)
        print("Tfidf vectorizer fitted.")
        pass

    def vectorizeDocs(self, docs, verbose=True, useTfidf=True):

        #docs = deepcopy(docs)
        docs = list(docs)
        tfidfFeats = self.tfidfVectorizer.transform(docs)
        for i in range(len(docs)):
            docs[i] = docs[i].split()

        docVectors = []
        if verbose:
            procRange = tqdm(range(len(docs)), desc="Vectorizing docs")
        else:
            procRange = range(len(docs))

        tfidfVocab = self.tfidfVectorizer.vocabulary_
        for i in procRange:
            tmpVector = []
            sentenceTfidf = tfidfFeats[i].toarray()
            for j in range(len(docs[i])):
                if docs[i][j] in self.w2vDict:
                    if useTfidf:
                        if docs[i][j] not in tfidfVocab:
                            continue
                        tfidfInd = tfidfVocab[docs[i][j]]
                        tfidf = sentenceTfidf[0][tfidfInd]
                        tmpVector.append(tfidf * self.w2vDict[docs[i][j]])
                    else:
                        tmpVector.append(self.w2vDict[docs[i][j]])
            if len(tmpVector) != 0:
                tmpVector = np.array(tmpVector)
                tmpVector = np.mean(tmpVector, axis=0)
            else:
                tmpVector = np.zeros(list(self.w2vDict.values())[0].shape)
            docVectors.append(tmpVector)
        return docVectors

    def vectorizeDocsMulticore(self, docs, n_jobs=2, verbose=True, useTfidf=True):

        def vectorizeBatch(docs, tfidfVectorizer, w2vDict, proc_id, verbose=True, useTfidf=True):

            #docs = deepcopy(docs)
            docs = list(docs)
            tfidfFeats = tfidfVectorizer.transform(docs)
            for i in range(len(docs)):
                docs[i] = docs[i].split()

            docVectors = []
            if verbose:
                procRange = tqdm(range(len(docs)), desc="Vectorizing docs")
            else:
                procRange = range(len(docs))

            tfidfVocab = tfidfVectorizer.vocabulary_
            for i in procRange:
                tmpVector = []
                sentenceTfidf = tfidfFeats[i].toarray()
                for j in range(len(docs[i])):
                    if docs[i][j] in w2vDict:
                        if useTfidf:
                            if docs[i][j] not in tfidfVocab:
                                continue
                            tfidfInd = tfidfVocab[docs[i][j]]
                            tfidf = sentenceTfidf[0][tfidfInd]
                            tmpVector.append(tfidf * w2vDict[docs[i][j]])
                        else:
                            tmpVector.append(w2vDict[docs[i][j]])
                if len(tmpVector) != 0:
                    tmpVector = np.array(tmpVector)
                    tmpVector = np.mean(tmpVector, axis=0)
                else:
                    tmpVector = np.zeros(list(w2vDict.values())[0].shape)
                docVectors.append(tmpVector)
            save( "./procRes_{}.pkl".format(proc_id), {proc_id: docVectors}, verbose=0)

        if len(docs) // n_jobs == 0:
            n_jobs = 1

        batchSize = len(docs) // n_jobs
        docsBatches = []
        for i in range(n_jobs - 1):
            docsBatches.append(docs[i*batchSize : (i+1)*batchSize])
        docsBatches.append( docs[(n_jobs-1)*batchSize:] )

        w2vDicts = []
        for i in range(n_jobs):
            w2vDicts.append( self.w2vDict )
        tfidfVecs = []
        for i in range(n_jobs):
            tfidfVecs.append(self.tfidfVectorizer)

        processes = []
        for i in range(n_jobs):
            processes.append(mp.Process(target=vectorizeBatch, args=(docsBatches[i], tfidfVecs[i], w2vDicts[i], i)))
            processes[i].start()
        for i in range(n_jobs):
            processes[i].join()

        tmp = []
        for i in range(n_jobs):
            procRes = load("./procRes_{}.pkl".format(i), verbose=0)
            os.remove("./procRes_{}.pkl".format(i))
            tmp.append(procRes[i])
        docVectors = tmp

        docVectors = np.vstack(docVectors)
        print(docVectors.shape)
        docVectors = list(docVectors)

        ################
        w2vDicts = None
        tfidfVecs = None
        docsBatches = None
        processes = None
        tmp = None
        gc.collect()
        ################

        return docVectors