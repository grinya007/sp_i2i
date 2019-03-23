import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# fuzzy finder class
class Search():
    # expects DataFrame and a name of column which becomes a learning corpus
    def __init__(self, df, column, analyzer='word', ngram_range=(1, 1)):
        self.df = df
        # TfidfVectorizer is so cool that it does all the work here
        # maximum document frequency set to 0.28 is just enough to eliminate "the"s out
        self.vectorizer = TfidfVectorizer(analyzer = analyzer, ngram_range = ngram_range, max_df = 0.28)
        # creates feature/document matrix where features are distinct words or ngrams
        self.matrix = self.vectorizer.fit_transform(df[column].values)
        self.features = set(self.vectorizer.get_feature_names())

    # performs query matching against the matrix
    def search(self, string):
        # scores features of query in accordance with corpus
        r = self.vectorizer.transform([string])
        # adds query as a new row to the matrix
        r = sp.vstack((r, self.matrix))
        # removes all columns where query row has zeros
        r = r[:, list(set(range(r.shape[1])) - set(np.where(r[0, :].todense() == 0)[1]))]
        # calculates similarities
        r = (r * r.T)[1:, 0].toarray()
        # turns result into DataFrame having original index
        r = pd.DataFrame(r, index=self.df.index, columns=['score'])
        # removes all irrelevant rows
        r = r[r['score'] != 0]
        # inner joins result to original DataFrame
        r = self.df.join(r, on=self.df.index, how='inner')
        # drops leftovers of join
        r = r.drop(labels='key_0', axis=1)
        # ranks results by relevance
        r = r.sort_values(by='score', ascending=False)
        return r

