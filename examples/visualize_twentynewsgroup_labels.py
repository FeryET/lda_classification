import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tomotopy import HDPModel

from lda_classification.model import TomotopyLDAVectorizer
from lda_classification.preprocess.spacy_cleaner import SpacyCleaner
from lda_classification.preprocess.gensim_cleaner import GensimCleaner

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

workers = 6
min_df = 5
rm_top = 5


def plot_topic_clusters(ax, x2d, y, labels):
    ax.set_aspect("equal")
    colors = cm.get_cmap("Spectral", len(labels))
    for i, l in enumerate(labels):
        c = colors(i / len(labels))
        ax.scatter(x2d[y == i, 0], x2d[y == i, 1], color=c, label=l, alpha=0.7)
    ax.grid()
    ax.legend()
    ax.set(adjustable='box', aspect='equal')
    return ax


def main(use_umap=True):
    labels = ["rec.autos", "rec.motorcycles", "rec.sport.baseball",
              "rec.sport.hockey"]

    raw_docs, y = fetch_20newsgroups(subset='all', return_X_y=True,
                                     categories=labels)

    processor = SpacyCleaner(workers=workers, chunksize=1000)
    docs = processor.transform(raw_docs)
    docs_train, docs_test, y_train, y_test = train_test_split(docs, y,
                                                              test_size=0.1,
                                                              shuffle=True)
    hdp_model = HDPModel(min_df=min_df, rm_top=rm_top)
    hdp_model.optim_interval = 5
    for d in docs_train:
        hdp_model.add_doc(d)
    hdp_model.burn_in = 100
    hdp_model.train(0, workers=workers)
    for i in range(0, 2000, 10):
        hdp_model.train(10, workers=workers)
        print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i,
                                                                             hdp_model.ll_per_word,
                                                                             hdp_model.live_k))

    print(hdp_model.summary())
    num_of_topics = hdp_model.live_k

    vectorizer = TomotopyLDAVectorizer(num_of_topics=num_of_topics,
                                       workers=workers, min_df=min_df,
                                       rm_top=rm_top)

    x_train = vectorizer.fit_transform(docs_train)
    x_test = vectorizer.transform(docs_test)

    print(vectorizer.lda.summary())

    title = "PCA Visualization of the Dataset using {}"
    if use_umap is True:
        from umap import UMAP
        dim_reducer = UMAP(n_components=2)
        title = title.format("UMAP")
    else:
        from sklearn.manifold import TSNE
        dim_reducer = TSNE(n_components=2)
        title = title.format("TSNE")

    x_transform = np.concatenate((x_train, x_test))
    x_transform = StandardScaler().fit_transform(x_transform)
    x_transform = dim_reducer.fit_transform(x_transform)

    x2d_train = x_transform[:x_train.shape[0], :]
    x2d_test = x_transform[x_train.shape[0]:, :]

    print(x2d_test.shape)
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    plot_topic_clusters(axes[0], x2d_train, y_train, labels)
    plot_topic_clusters(axes[1], x2d_test, y_test, labels)
    axes[0].set_title("Train Subset")
    axes[1].set_title("Test Subset")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog="This is an example for visualizing twentynewsgroup "
                 "dataset.")
    parser.add_argument("--enable-umap", action="store_true", required=False,
                        default=False)
    args = parser.parse_args()
    main(args.enable_umap)
