import setuptools

with open("README.md") as fh:
    long_desc = fh.read()

setuptools.setup(name="lda_classification", version="0.0.2",
                 author="Farhood Etaati", author_email="farhoodet@gmail.com",
                 long_description=long_desc,
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 url="https://github.com/FeryET/lda_classification",
                 classifiers=["Programming Language :: Python :: 3",
                              "License :: OSI Approved :: MIT License",
                              "Operating System :: OS Independent"],
                 install_requires=["gensim == 3.8.0", "matplotlib == 3.1.2",
                                   "numpy == 1.19.1", "pandas == 1.1.0",
                                   "scikit_learn == 0.23.1", "spacy == 2.3.1",
                                   "tqdm == 4.48.2", ],
                 python_requires='>=3.7.5')
