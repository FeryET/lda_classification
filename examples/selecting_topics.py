from sklearn.preprocessing import LabelEncoder

from lda_classification.dataloader import CogSciData, DataReader
from lda_classification.model.gensim_lda_vectorizer import GensimLDAVectorizer
from lda_classification.preprocess import SpacyCleaner
from lda_classification.utils import XGBoostFeatureSelector
import argparse


def run(path):
    range_n_topics = list(range(2, 51, 6))
    reader = DataReader(path, CogSciData)
    df = reader.to_dataframe()
    texts = SpacyCleaner().process_documents_multithread(list(df["text"]))
    x = GensimLDAVectorizer(num_topics=20).fit_transform(texts)
    # print(x.shape)
    y = LabelEncoder().fit_transform(list(df["label"]))
    selector = XGBoostFeatureSelector()
    x_select = selector.fit_transform(x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="store",
                        help="path to the directory of the dataset", type=str,
                        required=True)
    path = parser.parse_args().path
    run(path)
