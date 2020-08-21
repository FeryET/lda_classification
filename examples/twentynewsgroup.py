from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lda_classification.model import GensimLDAVectorizer
from lda_classification.preprocess import SpacyCleaner

data, target = fetch_20newsgroups(subset='all', return_X_y=True)
y_true = LabelEncoder().fit_transform(target)
processor = SpacyCleaner()
vectorizer = GensimLDAVectorizer(200, return_dense=False)
clf = SVC()
svd = TruncatedSVD(n_components=30)

# It's recommended to do data-agnostic preprocessing only one time to all
# data, but I've added it here just for the sake of fewer lines of code

pipe = Pipeline([("preprocessor", processor), ("vectorizer", vectorizer),
                 ("scalar", StandardScaler(copy=False, with_mean=False)),
                 ("dim_reduction", svd), ("classifier", clf)])

data_train, data_test, y_train, y_test = train_test_split(data, y_true,
                                                          test_size=0.1,
                                                          shuffle=True)
pipe.fit(data_train, y_train)
print(pipe.score(data_test, y_test))
