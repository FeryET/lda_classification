import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     cross_val_score, )
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from lda_classification.model import TomotopyLDAVectorizer
from lda_classification.preprocess.spacy_cleaner import SpacyCleaner

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
workers = 6

# Change this to false if you want to search for the
# number of topics via c_v score (super slow)
cats = ["rec.autos", "rec.motorcycles", "rec.sport.baseball",
        "rec.sport.hockey"]
docs, target = fetch_20newsgroups(subset='all', return_X_y=True,
                                  categories=cats)

y_true = LabelEncoder().fit_transform(target)

processor = SpacyCleaner(chunksize=1000, workers=workers)

docs = processor.transform(docs)

folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)

vectorizer = TomotopyLDAVectorizer(num_of_topics=15, workers=workers, min_df=5,
                                   rm_top=5)
clf = SVC()

pipe = Pipeline([("vectorizer", vectorizer), ("scalar", StandardScaler()),
                 ("classifier", clf)])

results = cross_val_score(pipe, docs, y_true, cv=folds, n_jobs=2, verbose=2,
                          scoring="accuracy")
print("Accuracy -> mean: {}\tstd: {}".format(results.mean(), results.std()))
