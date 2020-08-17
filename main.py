from dataloader import CogSciData, DataReader
from preprocess.spacy_cleaner import SpacyCleaner
from preprocess.gensim_cleaner import GensimCleaner


def main():
    path = "/home/farhood/Projects/datasets_of_cognitive/Data/Unprocessed Data"
    datareader = DataReader(path, data_type=CogSciData)
    df = datareader.to_dataframe()
    texts = list(df["text"])
    texts = SpacyCleaner().process_documents_multithread(texts)
    texts = list(df["text"])
    texts = GensimCleaner().process_documents_multithread(texts)
    df["processed"] = texts
    print(df)


if __name__ == '__main__':
    main()
