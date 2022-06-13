# BERT Database

The RETRO model uses a KNN database that stores (key, value) pairs of text and is indexed with BERT encodings. The (key, value) pair consists of a neighbor entry and its continuation where the raw text of both consists of the value and the BERT embedding of the neighbor entry is the key.


See [The Illustrated Retrieval Transformer](https://jalammar.github.io/illustrated-retrieval-transformer/) from Jay Alammar for more information.


To test the core database functions, build the docker image and run it.
`docker build -t test-bert -f Dockerfile .`


BERT model is composed of the [BERT preprocessing model](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3) and [BERT encoder model](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3) from Tensorflow Hub.
Tokenizer is the [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer) tokenizer from Huggingface Transformers.
Dataset is the [Wikitext](https://huggingface.co/datasets/wikitext) dataset from Huggingface Datasets.


### Faiss Implementation

The subfolder called `faiss_db/` contains an implementation of the BERT database using the faiss (Facebook AI Similarity Search) library from Facebook/Meta AI for faster vector similarity search. While this implementation still has a bit more to go in terms of development and flushing out the features of the database, faiss also allows for faster search (at the expense of some accuracy) and is much more efficient (in speed and memory usage) compared to the base implementation in the root directory of this repository. Note that the faiss module is best installed with anaconda, however when using pip to install faiss, use `pip install faiss-cpu` or `pip install faiss-gpu`. The specific version used in the requirements.txt file in the `faiss_db/` folder.


To test the core database functions, navigate to the `faiss_db/` folder and build the docker image before running it.
`docker build -t test-faiss-db -f Dockerfile .`


It is also important to note that the documentation for faiss is isolated to the module's [github](https://github.com/facebookresearch/faiss) (and is very hard to navigate).


### Scann Implementation

The subfolder called `scann_db/` contains an implementation of the BERT database using the ScaNN (Scalable Nearest Neighbors) library from Google for faster vector similarity search.

This implementation is still a WIP. Will update more when functionality is similar to base implementation or `faiss_db/` implementation.