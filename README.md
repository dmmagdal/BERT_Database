# BERT Database

The RETRO model uses a KNN database that stores (key, value) pairs of text and is indexed with BERT encodings. The (key, value) pair consists of a neighbor entry and its continuation where the raw text of both consists of the value and the BERT embedding of the neighbor entry is the key.


See [The Illustrated Retrieval Transformer](https://jalammar.github.io/illustrated-retrieval-transformer/) from Jay Alammar for more information.


To test the core database functions, build the docker image and run it.
`docker build -t test-bert -f Dockerfile .`


BERT model is composed of the [BERT preprocessing model](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3) and [BERT encoder model](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3) from Tensorflow Hub.
Tokenizer is the [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer) tokenizer from Huggingface Transformers.
Dataset is the [Wikitext](https://huggingface.co/datasets/wikitext) dataset from Huggingface Datasets.