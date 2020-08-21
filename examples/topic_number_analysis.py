from lda_classification.dataloader import CogSciData, DataReader
from lda_classification.preprocess import SpacyCleaner
import matplotlib.pyplot as plt
from lda_classification.evaluation.lda_coherence_evaluation import \
    LDACoherenceEvaluator
import argparse


def run(path):
    range_n_topics = list(range(2, 51, 6))
    reader = DataReader(path, CogSciData)
    df = reader.to_dataframe()
    texts = SpacyCleaner().process_documents_multithread(list(df["text"]))
    evaluator = LDACoherenceEvaluator(min_topic=2, return_dense=True,
                                      max_topic=100, step=10, alpha="symmetric",
                                      beta=None, workers=4, lda_iterations=300,
                                      max_df=0.5, min_df=5, random_state=300)
    ax = evaluator.evaluate(texts, mark_max=True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="store",
                        help="path to the directory of the dataset", type=str,
                        required=True)
    path = parser.parse_args().path
    run(path)
