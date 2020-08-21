from sklearn.base import TransformerMixin, BaseEstimator
from gensim.models import LdaMulticore, CoherenceModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense, corpus2csc
import numpy as np


class GensimLDAVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_topics, return_dense=True, max_df=0.5, min_df=5,
                 **lda_params):
        """
        :param num_topics: number of topics for the LDA model
        :param return_dense: transform function returns dense or not
        :param max_df: maximum word documentfrequency. Should be given as
        :param min_df: minimum word documentfrequency. Similar to max_df.
        :param lda_params: parameters for the constructor of
        gensim.model.Ldamulticore
        """
        super().__init__()
        self.lda: LdaMulticore = None
        self.corpus = None
        self.lda_params = lda_params
        self.lda_params["num_topics"] = num_topics
        self.is_dense = return_dense
        self.max_df = max_df
        self.min_df = min_df

    def fit(self, docs):
        """
        :param docs: List of split strings.
        :return: GensimLDAVectorizer
        """
        id2word = Dictionary(docs)
        id2word.filter_extremes(self.min_df, self.max_df)
        self.corpus = [id2word.doc2bow(d) for d in docs]
        self.lda = LdaMulticore(corpus=self.corpus, id2word=id2word,
                                **self.lda_params)
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
                            num_terms).T if self.is_dense else corpus2csc(
                lda_bag_of_topics, num_terms).T

    def fit_transform(self, docs, y=None, **fit_params):
        return self.fit(docs).transform(docs)

    def evaluate_coherence(self, docs, coherence="c_v"):
        """
        :param docs: List[List[str]]
        :param coherence: one of the coherence methods stated in
        gensim.models.CoherenceModel
        :return: gensim.models.CoherenceModel
        """
        return CoherenceModel(model=self.lda, texts=docs, corpus=self.corpus,
                              coherence=coherence,
                              processes=self.lda_params["workers"])

    def save(self, fname, *args, **kwargs):
        self.lda.save(fname=fname, *args, **kwargs)

    @classmethod
    def load(self, fname, return_dense=True, max_df=0.5, min_df=5, *args,
             **kwargs):
        lda = LdaMulticore.load(fname, *args, **kwargs)
        lda = LdaMulticore()
        alpha = lda.alpha
        eta = lda.eta
        iterations = lda.iterations
        random_seed = lda.random_state
        workers = lda.workers
        num_topics = lda.num_topics
        return GensimLDAVectorizer(num_topics, alpha, eta, workers, iterations,
                                   return_dense, max_df, min_df, random_seed)
