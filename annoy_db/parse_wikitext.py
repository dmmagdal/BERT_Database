# parse_wikitext.py
# Load wikitext dataset and parse it before loading it to a BERT
# database instance.
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import re
import os
import tensorflow as tf
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
from database_annoy import *


# Divide a tensor of shape (max_seq_len,) into chunks of size n.
# @param: text_tensor, a tensor of tokens (int32) that is going to be
#	divided into even chunks.
# @param: n, the max size of the chunk.
# @return: returns (yields) a tensor of shape (n,).
def divide_chunks(text_tensor, n):
	# Verify that the input text_tensor is of shape (max_seq_len,).
	assert len(tf.shape(text_tensor)) == 1,\
		f"Invalid tensor cardinality: expected 1, recieved {len(tf.shape(text_tensor))}"
	
	# Divide up the list of tokens in the text to equal shapes of size
	# n.
	for i in range(0, tf.shape(text_tensor)[0], n):
		yield text_tensor[i:i + n]


# Tokenize text list and split tokens list into chunks of 64 tokens
# long and reassemble the text.
# @param: text_list, list of texts to be parsed.
# @param: tokenizer, (pre-trained) tokenizer.
# @return: returns a list of list of string tensors, each tensor 
#	containing chunks of the original string.
def process(text_list, tokenizer):
	# List to contain all tokenized.
	text_chunks = []

	# Iterate through each line item in the list of text.
	for text in tqdm(text_list):#[:15]:
		# Use regex to find lines that are titles (formatted as
		# " = [title] = ", " = = [title] = = ", or
		# " = = = [title] = = = "). In particular, use search()
		# instead of match() because the latter starts its search only
		# at the beginning of the string instead of the whole string
		# like the former (search()) does.
		title1_re = re.compile(' = [A-Za-z ]+ =')
		title2_re = re.compile(' = = [A-Za-z ]+ = =')
		title3_re = re.compile(' = = = [A-Za-z ]+ = = =')
		search1 = title1_re.search(text)
		search2 = title2_re.search(text)
		search3 = title3_re.search(text)
		searches = [search1, search2, search3]

		# Skip all title lines and lines that are empty.
		if searches != [None, None, None] or text == "":
			continue

		# Tokenize the text. The result is a tensorflow tensor of shape
		# (batch_size, max_seq_len), where batch_size is 1 in this
		# case. Since only one sample is passed in the batch (hence
		# batch_size = 1), then tf.squeeze() the tensor to reduce the
		# dimensionality to just (max_seq_len). Note that there is no
		# need to specify max_length or padding arguments for the
		# tokenizer as we are only processing 1 string at a time. If
		# the initial tokenized output is less than 64 tokens, the
		# continuation text for the entry will be a blank string ("").
		tokens = tf.squeeze(
			tokenizer.encode(
				text,
				return_tensors="tf", # return tensors
			)
		)

		# Check the length of the tokenized output. Each set of tokens
		# will be broken in to chunks of 64. These chunks will become
		# the neighbor and continuation in their un-tokenized form for
		# the BERT database. 
		local_text_chunks = list(divide_chunks(tokens, 64))
		local_text_chunks = [
			tokenizer.decode(chunk) for chunk in local_text_chunks
		]

		# Append local text chunks to list of all text chunks.
		text_chunks.append(local_text_chunks)
			
	# Returned the list of processed text chunks. 
	return text_chunks


def main():
	# Initialize tokenizer.
	tokenizer = GPT2Tokenizer.from_pretrained(
		"gpt2", cache_dir="./gpt2-tokenizer"
	)
	tokenizer.add_special_tokens({
		'pad_token': "[PAD]"
	})

	# Load wikitext dataset. Both have test, train, and validation
	# splits. Note that the test and validation splits between wiki-103
	# and wiki-2 are the same size but wiki-103 has 60x larger train
	# split.
	wiki_103 = load_dataset(
		"wikitext", "wikitext-103-v1"
	)
	wiki_2 = load_dataset( 
		"wikitext", "wikitext-2-v1"
	)
	print("wikitext 103:")
	print(wiki_103)
	print("wikitext 2:")
	print(wiki_2)

	# Iterate through all datasets and splits.
	chunked_dataset = []
	for dataset in [wiki_103, wiki_2]:
		for split in ["train", "test", "validation"]:
			# TODO: Remove after verifying code works on smaller splits
			# of the dataset.
			if split == "train":
				continue

			# Process the text.
			wiki = "Wikitext-103" if dataset == wiki_103 else "Wikitext-2"
			print(f"Processing {wiki} {split}:")
			chunked_dataset_output = process(
				dataset[split]["text"], tokenizer
			)
			chunked_dataset += chunked_dataset_output
			print(f"Number of sections in dataset: {len(chunked_dataset_output)}")

	# Initialize BERT database. Load dataset to databse.
	db = BERTDatabase(initialize_model=True)
	db = load_dataset_to_db(chunked_dataset, db, "./wikitext-BERT_DB")
	db.save("./wikitext-BERT_DB")

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()