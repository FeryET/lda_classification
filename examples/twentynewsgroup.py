import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score, )
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from lda_classification.model import TomotopyLDAVectorizer
from lda_classification.preprocess.spacy_cleaner import SpacyCleaner

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
n_workers = 6

# Change this to false if you want to search for the
# number of topics via c_v score (super slow)
cats = ["rec.autos", "rec.motorcycles", "rec.sport.baseball",
        "rec.sport.hockey"]
docs, target = fetch_20newsgroups(subset='all', return_X_y=True,
                                  categories=cats)

y_true = LabelEncoder().fit_transform(target)

processor = SpacyCleaner(chunksize=1000, workers=4)

docs = processor.transform(docs)

folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)

vectorizer = TomotopyLDAVectorizer(num_of_topics=17, workers=2,
                                   min_cf=5, rm_top=10)
clf = SVC()
pca = PCA(n_components=0.95)

# It's recommended to do data-agnostic preprocessing only one time to all
# data, but I've added it here just for the sake of fewer lines of code

pipe = Pipeline([("vectorizer", vectorizer),
                 ("scalar", StandardScaler(with_mean=False)),
                 ("dimension reductor", pca), ("classifier", clf)])

results = cross_val_score(pipe, docs, y_true, cv=folds, n_jobs=4, verbose=1,
                          scoring="accuracy")
print("Accuracy -> mean: {}\tstd: {}".format(results.mean(), results.std()))
