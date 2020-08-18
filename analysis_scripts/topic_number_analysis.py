from dataloader import CogSciData, DataReader
from model.gensim_lda_vectorizer import GensimLDAVectorizer
from preprocess.gensim_cleaner import GensimCleaner
from preprocess.spacy_cleaner import SpacyCleaner
import matplotlib.pyplot as plt
from tqdm import tqdm


def run():
    path = "/home/farhood/Projects/datasets_of_cognitive/Data/Unprocessed Data"
    range_n_topics = list(range(2, 51, 6))
    reader = DataReader(path, CogSciData)
    df = reader.to_dataframe()
    # texts = GensimCleaner().process_documents_multithread(list(df["text"]))
    texts = SpacyCleaner().process_documents_multithread(list(df["text"]))
    values = []
    for n in tqdm(range_n_topics):
        vectorizer = GensimLDAVectorizer(num_topics=n)
        vectorizer.fit(texts)
        values.append(vectorizer.evaluate_coherence(texts).get_coherence())

    l1 = plt.plot(range_n_topics, values, label="coherence")
    # plt.legend((l1), ("coherence"))
    plt.gca().set_xlabel("Number of Topics")
    plt.gca().set_ylabel("Coherence Value")
    plt.title("Number of Topics vs Coherence Values")
    plt.show()


if __name__ == '__main__':
    run()
