
from pprint import pprint
import numpy as np
from tqdm import tqdm


class SummaryMaker():

    def __init__(self, preprocessor, vectorizer):

        self.preprocessor = preprocessor
        self.vectorizer = vectorizer

        pass

    def transform(self, x):

        summaries = []
        for i in tqdm(range(len(x)), "Making summary"):
            sentences = x[i].split(".")
            sentences = self.preprocessor.prerprocNames(sentences, removeStubStrings=True, verbose=False)

            if len(sentences) == 0:
                print("No summary at {}".format(i))
                summaries.append("No summary")
                continue

            sentenceVecs = self.vectorizer.vectorizeDocs(sentences, verbose=False)

            if len(sentenceVecs) == 0:
                summaries.append("no summary")
                print("No summary news ({}): {}".format(i, x[i]))
                continue

            sentenceVecs = np.array(sentenceVecs)
            centerVec = np.mean(sentenceVecs, axis=0)
            sentenceVecs = sentenceVecs[:4]
            diffs = sentenceVecs - centerVec
            distances = np.sqrt( np.linalg.norm(diffs, axis=1) )
            summary = sentences[ np.argmin(distances) ]
            summary = summary[0].upper() + summary[1:] + "."
            summaries.append(summary)

        return summaries