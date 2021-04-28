
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from joblib import Parallel, delayed
import gensim.downloader as api
from profittm.save_load import save, load
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


class W2VVectorizer():
    def __init__(self):
        self.w2vModel = None
        self.w2vDict = None
        pass

    def makeW2VDict(self, docs, size=128, window=5, n_jobs=10, min_count=1, sample=0, iter=100, sg=0, seed=45):

        docs = deepcopy(docs)
        docs = list(docs)
        for i in range(len(docs)):
            docs[i] = docs[i].split()

        self.w2vModel = Word2Vec(docs, size=size, window=window, workers=n_jobs, min_count=min_count, sample=sample, iter=iter, sg=sg, seed=seed)
        self.w2vDict = dict(zip(self.w2vModel.wv.index2word, self.w2vModel.wv.syn0))
        pass

    def setModel(self, model):
        self.w2vModel = model
        self.w2vDict = dict(zip(model.index2word, model.vectors))


    def vectorizeDocs(self, grammsDocs, n_jobs=1, tfidfVectorizer = None, verbose=True):

        def process_batch(batchGrammsDocs, tfidfFeats, tfidfVocab, w2vDict, verbose=True):
            batchVectors = []

            if verbose:
                procRange = tqdm(range(len(batchGrammsDocs)), desc="Vectorizing docs")
            else:
                procRange = range(len(batchGrammsDocs))

            for i in procRange:
                tmpVector = []

                sentenceTfidf = tfidfFeats[i].toarray()
                for j in range(len(batchGrammsDocs[i])):
                    if batchGrammsDocs[i][j] in w2vDict:
                        if tfidfVectorizer is not None:
                            if batchGrammsDocs[i][j] not in tfidfVocab:
                                continue
                            tfidfInd = tfidfVocab[batchGrammsDocs[i][j]]
                            tfidf = sentenceTfidf[0][tfidfInd]
                            tmpVector.append(tfidf * w2vDict[batchGrammsDocs[i][j]])
                        else:
                            tmpVector.append(w2vDict[batchGrammsDocs[i][j]])
                    #else:
                    #    print(batchGrammsDocs[i][j])
                if len(tmpVector) != 0:
                    tmpVector = np.array(tmpVector)
                    tmpVector = np.mean(tmpVector, axis=0)
                else:
                    tmpVector = np.zeros(list(w2vDict.values())[0].shape)
                batchVectors.append(tmpVector)
            return batchVectors

        splittedDocs = deepcopy(grammsDocs)
        splittedDocs = list(splittedDocs)

        if n_jobs > 1:
            for i in range(len(splittedDocs)):
                splittedDocs[i] = splittedDocs[i].split()
            splittedDocs = np.array_split(splittedDocs, n_jobs)

            w2vDicts = []
            for i in range(n_jobs):
                w2vDicts.append(deepcopy(self.w2vDict))

            self.w2vDict = None
            del self.w2vDict
            self.idfGrammDict = None
            del self.idfGrammDict

            docVectors = Parallel(n_jobs)(delayed(process_batch)(batchGrammsDocs, w2vDict) \
                                      for batchGrammsDocs, w2vDict in zip(splittedDocs, w2vDicts))
            self.w2vDict = deepcopy(w2vDicts[0])
            docVectors = np.vstack(docVectors)
        else:
            tfidfFeats = tfidfVectorizer.transform(splittedDocs)
            #tfidfFeats = tfidfFeats.toarray()
            for i in range(len(splittedDocs)):
                splittedDocs[i] = splittedDocs[i].split()
            docVectors = process_batch(splittedDocs, tfidfFeats, tfidfVectorizer.vocabulary_, self.w2vDict, verbose=verbose)

        return docVectors

    def setPretrainedModel(self, name):
        info = api.info()  # show info about available models/datasets
        model = api.load("glove-wiki-gigaword-100")
        save("./gwg.pkl", model)
        model = load("./gwg.pkl")
        self.setModel(model)
        pass