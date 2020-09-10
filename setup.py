import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
        name="bert-for-relation-extraction-jvasilakes",
        version="0.0.1",
        author="Jake Vasilakes",
        author_email="jvasilakes@gmail.com",
        description="BERT for relation extraction in Tensorflow 2",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="",
        packages=setuptools.find_packages() + ["bert_re"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
)
