import logging

import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models.hdpmodel import HdpModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lda_classification.evaluation.lda_coherence_evaluation import \
    LDACoherenceEvaluator
from lda_classification.model import GensimLDAVectorizer
from lda_classification.preprocess.spacy_cleaner import SpacyCleaner

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
n_workers = 6

# Change this to false if you want to search for the
# number of topics via c_v score (super slow)
IS_HIERARCHICAL = True
cats = ["rec.autos", "rec.motorcycles", "rec.sport.baseball",
        "rec.sport.hockey"]
data, target = fetch_20newsgroups(subset='all', return_X_y=True,
                                  categories=cats)
y_true = LabelEncoder().fit_transform(target)

data_train, data_test, y_train, y_test = train_test_split(data, y_true,
                                                          test_size=0.1,
                                                          shuffle=True)

processor = SpacyCleaner(chunksize=1000)

docs = processor.transform(data_train)

if IS_HIERARCHICAL:
    id2word = Dictionary(docs)
    id2word.filter_extremes(no_above=5, no_below=0.5)
    corpus = [id2word.doc2bow(d) for d in docs]
    hdp_model = HdpModel(corpus, id2word)
    num_of_topics = len(hdp_model.get_topics())
    print(num_of_topics)
else:
    num_of_topics, ax = LDACoherenceEvaluator(min_topic=5, max_topic=56, step=5,
                                              workers=4, min_df=5, max_df=0.5,
                                              eval_every=1, passes=50,
                                              random_state=500).evaluate(docs,
                                                                         mark_max=True)
    plt.savefig("results/20NG_sports_number_of_topics.png")
    plt.show()

vectorizer = GensimLDAVectorizer(num_topics=num_of_topics, max_df=0.5, min_df=5,
                                 return_dense=False, eval_every=1, passes=50,
                                 workers=n_workers)
clf = SVC()
svd = TruncatedSVD(n_components=30)

# It's recommended to do data-agnostic preprocessing only one time to all
# data, but I've added it here just for the sake of fewer lines of code

pipe = Pipeline([("preprocessor", processor), ("vectorizer", vectorizer),
                 ("scalar", StandardScaler(with_mean=False)),
                 ("dimension reductor", svd), ("classifier", clf)])

pipe.fit(data_train, y_train)
print(pipe.score(data_test, y_test))
