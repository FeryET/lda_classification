import setuptools

with open("README.md") as fh:
    long_desc = fh.read()

with open("requirements.txt") as req:
    requirements = req.readlines()

setuptools.setup(name="lda_classification", version="0.0.26",
                 author="Farhood Etaati", author_email="farhoodet@gmail.com",
                 long_description=long_desc,
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 url="https://github.com/FeryET/lda_classification",
                 classifiers=["Programming Language :: Python :: 3",
                              "License :: OSI Approved :: MIT License",
                              "Operating System :: OS Independent"],
                 install_requires=requirements, python_requires='>=3.5')
