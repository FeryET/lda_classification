from sklearn.base import TransformerMixin, BaseEstimator
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense, corpus2csc
import numpy as np


def get_kwargs(**kwargs):
    return kwargs


class GensimLDAVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_topics, alpha="symmetric", beta=None, workers=2,
                 lda_iterations=100, dense_matrix=True):
        super().__init__()
        self.lda: LdaMulticore = None
        self.kwargs = get_kwargs(num_topics=num_topics, workers=workers,
                                 iterations=lda_iterations, alpha=alpha,
                                 eta=beta)
        self.is_dense = dense_matrix

    def fit(self, docs):
        """
        :param docs: List of split strings.
        :return: GensimLDAVectorizer
        """
        id2word = Dictionary(docs)
        id2word.filter_extremes()
        corpus = [id2word.doc2bow(d) for d in docs]
        self.lda = LdaMulticore(corpus=corpus, id2word=id2word, **self.kwargs)
        return self

    def transform(self, docs):
        """
        :param docs: List of split strings.
        :return: numpy.ndarray
        """
        cur_bow = [self.lda.id2word.doc2bow(d) for d in docs]
        lda_bag_of_topics = [self.lda[c] for c in cur_bow]
        num_terms = self.lda.num_topics
        return corpus2dense(lda_bag_of_topics,
                            num_terms) if self.is_dense else corpus2csc(
                lda_bag_of_topics, num_terms)

    def fit_transform(self, docs, y=None, **fit_params):
        return self.fit(docs).transform(docs)
