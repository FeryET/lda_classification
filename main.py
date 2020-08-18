from dataloader import CogSciData, DataReader
from preprocess.spacy_cleaner import SpacyCleaner
from preprocess.gensim_cleaner import GensimCleaner
from model.gensim_lda_vectorizer import GensimLDAVectorizer
from evaluation.lda_coherence_evaluation import LDACoherenceEvaluator
import matplotlib.pyplot as plt


def main():
    path = "/home/farhood/Projects/datasets_of_cognitive/Data/Unprocessed Data"
    datareader = DataReader(path, data_type=CogSciData)
    df = datareader.to_dataframe()
    texts = list(df["text"])
    texts = GensimCleaner().process_documents_multithread(texts)
    evaluator = LDACoherenceEvaluator(min_topic=2, max_topic=10, step=2,
                                      alpha="symmetric", beta=None, workers=2,
                                      lda_iterations=100, return_dense=True,
                                      max_df=0.5, min_df=5, random_seed=0)
    ax = evaluator.evaluate(texts, mark_max=True)
    plt.show()  # vectorizer = GensimLDAVectorizer(5)  # X =
    # vectorizer.fit_transform(texts)  # print(vectorizer.evaluate_coherence(
    # texts).get_coherence())


if __name__ == '__main__':
    main()
