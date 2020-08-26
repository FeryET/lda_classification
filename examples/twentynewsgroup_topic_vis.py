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
from lda_classification.preprocess.gensim_cleaner import GensimCleaner

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

n_workers = 6


def plot_topic_clusters(ax, x2d, y, labels):
    ax.set_aspect("equal")
    colors = cm.get_cmap("Spectral", len(labels))
    for i, l in enumerate(labels):
        c = colors(i / len(labels))
        ax.scatter(x2d[y == i, 0], x2d[y == i, 1], color=c, label=l, alpha=0.7)
    ax.grid()
    ax.legend(prop={'size': 6})
    ax.autoscale()
    return ax


def main(use_umap=True):
    labels = ["rec.autos", "rec.motorcycles", "rec.sport.baseball",
              "rec.sport.hockey"]
    data = fetch_20newsgroups(subset='all', categories=labels)

    data, target = data.data, data.target
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(target)
    data_train, data_test, y_train, y_test = train_test_split(data, y_true,
                                                              test_size=0.1,
                                                              shuffle=True)
    processor = GensimCleaner(6)
    train_docs = processor.transform(data_train)
    test_docs = processor.transform(data_test)
    hdp_model = HDPModel(min_cf=5, rm_top=10)
    hdp_model.optim_interval = 5
    for d in train_docs:
        hdp_model.add_doc(d)
    hdp_model.burn_in = 100
    hdp_model.train(0, workers=n_workers)
    for i in range(0, 1000, 10):
        hdp_model.train(10, workers=n_workers)
        print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i,
                                                                             hdp_model.ll_per_word,
                                                                             hdp_model.live_k))

    num_of_topics = hdp_model.live_k
    print(num_of_topics)

    vectorizer = TomotopyLDAVectorizer(num_of_topics=num_of_topics,
                                       workers=n_workers)

    x_train = vectorizer.fit_transform(train_docs)
    x_test = vectorizer.transform(test_docs)

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
    dpi = 300
    fig, axes = plt.subplots(ncols=2, figsize=(3000 / dpi, 1500 / dpi), dpi=dpi)
    plot_topic_clusters(axes[0], x2d_train, y_train, labels)
    plot_topic_clusters(axes[1], x2d_test, y_test, labels)
    axes[0].set_title("Train Subset")
    axes[1].set_title("Test Subset")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog="This is an example for visualizing twentynewsgroup "
                 "dataset.")
    parser.add_argument("--enable-umap", action="store_true", required=False,
                        default=False)
    args = parser.parse_args()
    main(args.enable_umap)
