import matplotlib.pyplot as plt
from tqdm import tqdm

from lda_classification.model.gensim_lda_vectorizer import GensimLDAVectorizer


class LDACoherenceEvaluator:
    def __init__(self, min_topic, max_topic, step, **kwargs):
        self.params = kwargs
        self.topic_range = list(range(min_topic, max_topic, step))

    def evaluate(self, docs, coherence_method="c_v", plot=True, ax=None,
                 mark_max=False, grid=True):
        coherence_scores = []
        for n in tqdm(self.topic_range, desc="computing coherences"):
            vectorizer = GensimLDAVectorizer(num_topics=n, **self.params).fit(
                    docs)
            c = vectorizer.evaluate_coherence(docs,
                                              coherence=coherence_method).get_coherence()
            coherence_scores.append(c)
        if plot is True:
            if ax is None:
                ax = plt.subplot()
            ax.plot(self.topic_range, coherence_scores, label="coherence",
                    marker=".")
            ax.set_xticks(self.topic_range)
            ax.set_xlabel("Number of Topics")
            ax.set_ylabel("Coherence Values")
            ax.set_title("Evaluation of LDA Model  (metric = {})".format(
                    coherence_method))
            if grid is True:
                ax.grid()
            if mark_max is True:
                idx = coherence_scores.index(max(coherence_scores))
                ax.annotate("{:.3f}".format(coherence_scores[idx]),
                            (self.topic_range[idx], coherence_scores[idx]),
                            textcoords="offset points",
                            # how to position the text
                            xytext=(3, 3), size=8)
                ax.axvline(self.topic_range[idx], alpha=0.5, color="r",
                           linestyle="--")
            return coherence_scores, ax
        else:
            return coherence_scores, None
