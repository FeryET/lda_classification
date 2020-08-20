from sklearn.base import TransformerMixin, BaseEstimator
from gensim.models import LdaMulticore, CoherenceModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense, corpus2csc
import numpy as np


def get_kwargs(**kwargs):
    return kwargs


class GensimLDAVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_topics, alpha="symmetric", beta=None, workers=2,
                 lda_iterations=100, return_dense=True, max_df=0.5, min_df=5,
                 random_seed=0):
        """
        :param num_topics: number of topics for the LDA model
        :param alpha: the alpha parameter in gensim.models.LdaMulticore
        :param beta: the eta parameter in gensim.models.LdaMulticore
        :param workers: number of cpu workers
        :param lda_iterations: number of training iterations
        :param return_dense: transform function returns dense or not
        :param max_df: maximum word documentfrequency. Should be given as
        either float in (0,1) or an integer.
        :param min_df: minimum word documentfrequency. Similar to max_df.
        :param random_seed: 0 means no pre-determined seed.
        """
        super().__init__()
        self.lda: LdaMulticore = None
        self.corpus = None
        self.kwargs = get_kwargs(num_topics=num_topics, workers=workers,
                                 iterations=lda_iterations, alpha=alpha,
                                 eta=beta, random_state=random_seed)
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
                                **self.kwargs)
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
                              processes=self.kwargs["workers"])

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
        return GensimLDAVectorizer(num_topics, alpha, eta, workers,
                                         iterations, return_dense, max_df,
                                         min_df, random_seed)