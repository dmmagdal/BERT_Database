# BERT Database

The RETRO model uses a KNN database that stores (key, value) pairs of text and is indexed with BERT encodings. The (key, value) pair consists of a neighbor entry and its continuation where the raw text of both consists of the value and the BERT embedding of the neighbor entry is the key.


See [The Illustrated Retrieval Transformer](https://jalammar.github.io/illustrated-retrieval-transformer/) from Jay Alammar for more information.


To test the core database functions, build the docker image and run it.
`docker build -t test-bert -f Dockerfile .`


BERT model is composed of the [BERT preprocessing model](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3) and [BERT encoder model](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3) from Tensorflow Hub.
Tokenizer is the [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer) tokenizer from Huggingface Transformers.
Dataset is the [Wikitext](https://huggingface.co/datasets/wikitext) dataset from Huggingface Datasets.


### Annoy Implementation

The subfolder called 'annoy_db/' contains an implementation of the BERT database using the ANNOY (Approximate Nearest Neighbors Oh Yeah) library from Spotify for fast vector similarity search. 

----To add once completed----
While this implementation still has a bit more to go in terms of development and flushing out the features of the database, annoy is supposed to be efficient (in speed and memory usage) compared to the base implementation in the root directory of this repository. The specific version of annoy used can be found in the requirements.txt file in the `annoy_db/` folder.


To test the core database functions, navigate to the `annoy_db/` folder and build the docker image before running it.
`docker build -t test-annoy-db -f Dockerfile .`


It is also important to note that the documentation for annoy is isolated to the module's [github](https://github.com/spotify/annoy) (and is very rudimentary in terms of the features it supports).


There are a few caveats to the annoy database implementation:
 - The first major difference is that annoy does not support deleting items from the index. There is the option to zero out values at a position `i` in the index where the vector `v` is stored, however that is a great waste of space so only adding items is supported. 
 - The next major caveat is that once the `build()` function is called, no new items can be added. So calling that function must happen **ONLY AFTER** all relevant data has been added to the database.
----End of entry----


This implementation is still a WIP. Will update more when functionality is simlar to base implementation or `faiss_db/` implementation.


### Faiss Implementation

The subfolder called `faiss_db/` contains an implementation of the BERT database using the faiss (Facebook AI Similarity Search) library from Facebook/Meta AI for faster vector similarity search. While this implementation still has a bit more to go in terms of development and flushing out the features of the database, faiss also allows for faster search (at the expense of some accuracy) and is much more efficient (in speed and memory usage) compared to the base implementation in the root directory of this repository. Note that the faiss module is best installed with anaconda, however when using pip to install faiss, use `pip install faiss-cpu` or `pip install faiss-gpu`. The specific version used in the requirements.txt file in the `faiss_db/` folder.


To test the core database functions, navigate to the `faiss_db/` folder and build the docker image before running it.
`docker build -t test-faiss-db -f Dockerfile .`


It is also important to note that the documentation for faiss is isolated to the module's [github](https://github.com/facebookresearch/faiss) (and is very hard to navigate).


### Scann Implementation

The subfolder called `scann_db/` contains an implementation of the BERT database using the ScaNN (Scalable Nearest Neighbors) library from Google for faster vector similarity search.

This implementation is still a WIP. Will update more when functionality is similar to base implementation or `faiss_db/` implementation.
