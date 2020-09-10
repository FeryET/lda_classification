import os

from lda_classification.preprocess.spacy_cleaner import SpacyCleaner
import pandas as pd

path = "/home/farhood/Projects/datasets_of_cognitive/Data/S2ORC_linguistics"

dfs = []
for root, __, files in os.walk(path):
    for f in files:
        if f.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(root, f), index_col=False))
for df in dfs[1:]:
    dfs[0] = dfs[0].append(df)

df = dfs[0].copy()
del dfs
df.dropna(inplace=True)
df = df[df.columns[1]]
# df.columns = ["text"]
# df.reset_index(inplace=True)
df = list(df)
df = df[:5000]
cleaner = SpacyCleaner(workers=4, chunksize=300)

texts = cleaner.process_documents(df)
texts = cleaner.fit_transform(df)