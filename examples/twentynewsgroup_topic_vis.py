import logging

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.hdpmodel import HdpModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tomotopy import HDPModel
from lda_classification.model import GensimLDAVectorizer
from lda_classification.preprocess.gensim_cleaner import GensimCleaner
from lda_classification.preprocess.spacy_cleaner import SpacyCleaner
import matplotlib.pyplot as plt


# Courtesy of Gensim's documentation
def plot_difference(mdiff, title="", annotation=None):
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)


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

# processor = SpacyCleaner(chunksize=1000)
processor = GensimCleaner(6)
docs = processor.transform(data_train)

id2word = Dictionary(docs)
id2word.filter_extremes(no_above=5, no_below=0.5)
corpus = [id2word.doc2bow(d) for d in docs]
hdp_model = HDPModel(min_cf=5, rm_top=10)
for d in docs:
    hdp_model.add_doc(d)
hdp_model.burn_in = 100
hdp_model.train(0)
for i in range(0, 1000, 10):
    hdp_model.train(100)
    print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i,
                                                                         hdp_model.ll_per_word,
                                                                         hdp_model.live_k))

num_of_topics = hdp_model.live_k
print(num_of_topics)

vectorizer = GensimLDAVectorizer(num_topics=num_of_topics, max_df=0.5, min_df=5,
                                 return_dense=False, eval_every=1, passes=50,
                                 workers=n_workers)

X = vectorizer.fit_transform(docs)

mdiff, annotation = vectorizer.lda.diff(vectorizer.lda, distance='jaccard',
                                        num_words=50)
plot_difference(mdiff, title="Topic difference (one model) [jaccard distance]",
                annotation=annotation)
plt.show()
