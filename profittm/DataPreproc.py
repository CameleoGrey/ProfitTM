
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import Stemmer
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

class DataPreproc():

    def __init__(self):
        self.stopWords = [ "that", "a", "to", "of", "which", "and", "while", "in", "for", "those", "their", "these",
                           "this", "but", "howev", "it", "also", "the", "onli", "have", "one", "t", "s", "v", "d", "at", "has", "what"]
        self.stopWords = self.stopWords + list(set(stopwords.words('english')))
        for i in range(len(self.stopWords)):
            self.stopWords[i] = Stemmer.Stemmer("english").stemWord(self.stopWords[i])
        self.stopWords = list(set(self.stopWords))
        #self.translit = Transliterator()
        self.lemmatizer = WordNetLemmatizer()

        #self.uselessWords = []

    def preprocDocString(self, sample):
        docStr = str(sample)
        docStr = docStr.lower()

        #docStr = list(docStr)
        #for i in range(len(docStr)):
        #    docStr[i] = self.translit.transliterate(docStr[i])
        #docStr = "".join(docStr)

        docStr = re.sub("[^A-Za-zА-Яа-я\s\t]+", " ", docStr)

        docStr = docStr.split()
        for i in range(len(docStr)):
            docStr[i] = self.lemmatizer.lemmatize(docStr[i])
        docStr = " ".join(docStr)

        docStr = docStr.split()
        for i in range(len(docStr)):
            docStr[i] = Stemmer.Stemmer("english").stemWord(docStr[i])
        docStr = " ".join(docStr)

        #remove stopwords
        docStr = docStr.split()
        docStr = " ".join([i for i in docStr if i not in self.stopWords])
        #checkWords = deepcopy(docStr)
        #for word in checkWords:
        #    if word in self.stopWords:
        #        docStr.remove(word)
        #docStr = " ".join(docStr)

        docStr = re.sub("\n+", " ", docStr)
        docStr = re.sub(" +", " ", docStr)
        docStr = docStr.strip()
        if docStr == "" or docStr == " ":
            docStr = "$$$STUB$$$"

        return docStr

    def prerprocMulticore(self, df, n_jobs, remove_stub_strings=True):
        def preprocBatch(batch, remove_stub_strings):
            validStringsInd = []
            for i in tqdm(range(batch.shape[0]), desc="Prerpoc data"):
                if remove_stub_strings:
                    preprocRow = self.preprocDocString(batch[i])
                    if preprocRow != "$$$STUB$$$":
                        validStringsInd.append(preprocRow)
                else:
                    preprocRow = self.preprocDocString(batch[i])
                    validStringsInd.append(preprocRow)
            return validStringsInd

        df = np.array_split(df, n_jobs)
        df = Parallel(n_jobs)(delayed(preprocBatch)(batch, remove_stub_strings) for batch in df)
        df = np.hstack(df)

        return df

    def prerprocNames(self, strList, removeStubStrings=True, verbose=True):

        preprocCorpus = []

        if verbose:
            procRange = tqdm(strList)
        else:
            procRange = strList

        for row in procRange:
            if removeStubStrings:
                preprocRow = self.preprocDocString(row)
                if preprocRow != "$$$STUB$$$":
                    preprocCorpus.append( preprocRow )
            else:
                preprocRow = self.preprocDocString(row)
                preprocCorpus.append(preprocRow)

        return preprocCorpus

    def getUniqTextList(self, preprocTexts):
        uniqTexts = np.hstack([preprocTexts[:, 0], preprocTexts[:, 1]])
        uniqTexts = np.unique(uniqTexts)
        uniqTexts = list(sorted(list(uniqTexts)))
        return uniqTexts
