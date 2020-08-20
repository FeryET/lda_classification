# lda_classifcation

Instantly train an LDA model with a scikit-learn compatible wrapper around gensim's LDA model.


* Preprocess Your Documents
* Train an LDA 
* Evaluate Your LDA Model
* Extract Document Vectors 
* Select the Most Informative Features
* Classify Your Documents

All in a few lines of code, completely compatible with `sklearn`'s Transformer API.


####Installation:
If you want to install via Pypi:
```pip install lda_classification```
If you want to install from the sourcefile:
```
git clone https://github.com/FeryET/lda_classification.git
cd lda_classification/
python setup.py install
```
####Requirements:

```
gensim == 3.8.0
matplotlib == 3.1.2
numpy == 1.19.1
pandas == 1.1.0
scikit_learn == 0.23.1
setuptools == 49.6.0.post20200814
spacy == 2.3.1
tqdm == 4.48.2
xgboost == 1.1.1
```

 
####Example: 
```python
from model import GensimLDAVectorizer
from preprocess.spacy_cleaner import SpacyCleaner
from utils.model_selection.xgboost_features import XGBoostFeatureSelector

# docs, labels = FETCH YOUR DATASET 
# y = ENCODED_LABELS
docs = SpacyCleaner().transform(docs)
X = GensimLDAVectorizer(200, return_dense=False).fit_transform(docs)
X_transform = XGBoostFeatureSelector().fit_transform(X, y)
```

There is also a `dataloader` class and a `BaseData` class in
order to automate reading your data files from disk. Extend
`BaseData` and implement the abstractmethods in the subclass and
feed it to `DataReader` to simplify fetching your dataset.