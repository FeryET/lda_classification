import setuptools

with open("README.md") as fh:
    long_desc = fh.read()

setuptools.setup(name="lda_classifcation", version="0.0.1",
                 author="Farhood Etaati", author_email="farhoodet@gmail.com",
                 long_description=long_desc,
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 classifiers=["Programming Language :: Python :: 3",
                              "License :: OSI Approved :: MIT License",
                              "Operating System :: OS Independent"],
                 python_requires='>=3.7.5')
