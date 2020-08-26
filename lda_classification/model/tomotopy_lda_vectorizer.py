from sklearn.base import BaseEstimator, TransformerMixin
from tomotopy import LDAModel
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

"""
tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k=1, alpha=0.1, eta=0.01, 
seed=None, corpus=None, transform=None
"""


class TomotopyLDAVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_of_topics, workers=2, train_iter=10, infer_iter=100,
                 train_steps=100, return_dense=True, sparse_threshold=0.01,
                 min_df=5, rm_top=10):
        self.num_of_topics = num_of_topics
        self.min_df = min_df
        self.rm_top = rm_top
        self.workers = workers
        self.train_steps = train_steps
        self.train_iter = train_iter
        self.infer_iter = infer_iter
        self.return_dense = return_dense
        self.sparse_threshold = sparse_threshold

    def fit(self, docs):
        self.lda = LDAModel(k=self.num_of_topics, min_df=self.min_df,
                            rm_top=self.rm_top)
        for d in docs:
            self.lda.add_doc(d)
        for _ in range(0, self.train_steps * self.train_iter, self.train_iter):
            self.lda.train(iter=self.train_iter, workers=self.workers)
        return self

    def transform(self, docs):
        trans_docs = [self.lda.make_doc(d) for d in docs]
        inferred = self.lda.infer(trans_docs, iter=self.infer_iter,
                                  workers=self.workers)[0]
        if self.return_dense:
            inferred = np.array(inferred)
        else:
            inferred = [i[i > self.sparse_threshold] for i in inferred]
            inferred = csr_matrix(inferred)
        return inferred

    def fit_transform(self, docs, y=None, **fit_params):
        return self.fit(docs).transform(docs)
