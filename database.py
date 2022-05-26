# database.py
# BERT KNN database.
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow_text as text
from tqdm import tqdm


# Also note that for the design of the database, instead of using a
# dict where the key is the tensorflow tensor of the neighbor's BERT
# embedding and the value is the tuple of string tensors with the
# actual neighbor and it's completion,
# ie {Neighbor BERT Embedding: (Neighbor, Continuation)}
# there is an option to make the database a list of dict objects such
# that each values in the dict where the keys are the respective
# components,
# ie [
#	{
#		Neighbor: Neighbor,
#		Continuation: Continuation,
#		Embedding: Neighbor BERT Embedding,
#	},
# ]
# which may result in faster operation runtimes (I havent tested it).
# This was inspired by the wiki_dpr dataset hosted on the
# Huggingface Datasets website:
# https://huggingface.co/datasets/wiki_dpr
# This allows for natural json storage instead of using the pickle
# module.


class BERTDatabase:
	def __init__(self, initialize_model=False, path="./Embedding_BERT"):
		# Standard save path for BERT model.
		self.save_path = path

		# Initialize or load BERT model.
		if initialize_model:
			print("Initializing new BERT model for text embedding...")
			self.build_bert()
			print("New BERT model initialized successfully.")
		else:
			print("Loading saved BERT model for text embedding...")
			self.bert = load_model(self.save_path)
			self.bert.summary()
			print("BERT model loaded successfully.")

		# Data storage.
		self.data = {}

		# Save file name for data.
		self.file = "bert_db_data.pkl"


	# Initialize a new BERT model and save it.
	# @param: takes no arguments.
	# @return: returns nothing.
	def build_bert(self):
		# Tensorflow Hub links for the BERT preprocessor and encoder
		# model.
		BERT_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3" # this is marked as a text embedding model
		PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3" # this is marked as a text preprocessing model

		# Create a BERT model with the Keras Functional API.
		text_input = keras.layers.Input(shape=(), dtype=tf.string)
		preprocessor_layer = hub.KerasLayer(PREPROCESS_MODEL)
		encoder_inputs = preprocessor_layer(text_input)
		encoder = hub.KerasLayer(BERT_MODEL, trainable=False)
		outputs = encoder(encoder_inputs)
		pooled_output = outputs["pooled_output"] # [batch_size, 768] (represent each input sequence as a whole)
		sequence_output = outputs["sequence_output"] # [batch_size, seq_length, 768] (represent each input token in context)
		model = keras.Model(
			inputs=text_input, outputs=pooled_output, name="Bert"
		)
		model.trainable = False
		model.summary()

		# Save the model.
		model.save(self.save_path)

		# Set BERT model.
		self.bert = model


	# Embed text with BERT.
	# @param: text_tensor, string tensor of shape [batch_size,]
	#	containing the texts to be embedded with the BERT model.
	# @return: returns an embedding tensor of shape [batch_size, 768].
	def encode_text(self, text_tensor):
		# Assertion to check input tensor shape.
		assert len(text_tensor.shape) == 1, f"Invalid shape to BERT model {text_tensor.shape}"

		# Convert string tensor to BERT embedding.
		return self.bert(text_tensor)


	# Notes on the (key, value) store database:
	# Can only add tf.Tensor if it is hashable with .ref() (can deref
	# with deref()). To pass a tensor to a dict as the key:
	# dict[tensor.ref()] = value. re() returns a hashable reference
	# object to the tensor. Even if another variable is the exact same
	# as the first tensor, ref() will result in different hashes. This
	# is why a = tf.constant([1, 2, 3, 4], dtype=tf.int32) and
	# b = tf.constant([1, 2, 3, 4], dtype=tf.int32), even though they
	# are the exact same, they will result in different hashes in the
	# dict so dict[a.ref()] = 12, dict[b.ref()] = 11 will result in two
	# entries in the dict.
	# Do NOT use tf.lookup.KeyValueTensorInitializer and
	# tf.lookup.StaticHashTable because the hash table becomes
	# immutable after it is initialized. Experimental modules such as
	# tf.lookup.experimental.MutableHashTable or
	# tf.lookup.experimental.DenseHashTable also do not work because
	# they do not support having the key_dtype=tf.float32.


	# Retrieve the value of a (key, value) pair from the database given
	# the keys.
	# @param: keys, string tensor containing the text entries to be
	#	retrieved from the database.
	# @param: verbose, boolean that tells whether the "key not found" 
	#	message should be printed out. Default is True. All class 
	#	functions that call get() (ie add(), remove(), etc) will pass 
	#	in their verbos value.
	# @return: returns a list of (key, value) pair tensors from the database.
	def get(self, keys, verbose=True):
		# Type and tensor shape checking.
		if not isinstance(keys, tf.Tensor) or keys.dtype != tf.string:
			raise TypeError(f"Entry {keys} is not a Tensorflow tensor of type tf.string.")
		elif len(tf.shape(keys)) != 1:
			raise ValueError(f"Entry {keys} shape cardinality is {len(tf.shape(keys))} instead of 1. Input entry shape {tf.shape(keys)}.")

		# Load keys.
		database_keys = list(self.data.keys())

		# Return list of None (use None for all input keys that do not
		# exist within the dictionary) if the database is empty.
		if len(database_keys) == 0:
			return [None] * tf.shape(keys)[0].numpy().item(0)

		# Retrieve embedding and the entries from database. Current
		# runtime is O(n^2). Could be better but I can't figure it out.
		database_items = []
		for i in range(tf.shape(keys)[0]):
			# Embed entry with BERT model.
			emb_key = self.encode_text(tf.expand_dims(keys[i], axis=0)) # Needs to be shape (1,)

			# Retrieve indices where matching search key is matched
			# (true). This will be used to extract the copy of the
			# desired key. As noted earlier, the original copy of the
			# key is required to retrieve it's value from the
			# dictionary.
			key_found = tf.where(
				[
					# Reduce all true values within the comparison
					# between the input key and the dictionary key. Do
					# this for all keys in the dictionary.
					tf.math.reduce_all(
						# Check that all values of the input key match
						# with the dereferenced dictionary key. This
						# will create a bool tensor of shape [768,].
						# tf.math.equal(key.deref(), emb_key[i, :])
						tf.math.equal(key.deref(), emb_key)
					)#.numpy()
					for key in database_keys
				]
			)#.numpy()

			# If there is a matching tensor (there should be only 1
			# match), the tensor shape will be (1, 1). If there is no
			# match, the tensor shape is (0, 1).
			if key_found.shape.as_list() == [1, 1]:
				# Isolate the index of the matching key. Use it to
				# retrieve an original copy of the key from the list of
				# dictionary keys.
				key_index = key_found.numpy().item(0)
				key = database_keys[key_index]
				database_items.append(
					(key, self.data[key])
				)
			else:
				# Print an error message that the key string does not
				# exist in the database. Append a None to the return
				# list.
				if verbose:
					print(f"Key {keys[i]} does not exist in database.")
				database_items.append(None)

		# Return the list of items retrieved from the database.
		assert len(database_items) == tf.shape(keys)[0]
		return database_items


	# Add (key, value) entry pairs to the database.
	# @param: input_pairs, a tuple of string tensors containing the
	#	primary entry and its continuation.
	# @param: verbose, boolean that tells whether the "key not found" 
	#	message in get() should be printed out. Default is False. 
	# @return: returns nothing.
	def add(self, input_pairs, verbose=False):
		# Unpack input pairs. Input pairs consist of an entry as well
		# as a continuation text. The BERT embedding of the initial
		# text entry is the key to the KNN database.
		entry, continuation = input_pairs
		
		# Type and tensor shape checking.
		if not isinstance(entry, tf.Tensor) or entry.dtype != tf.string:
			raise TypeError(f"Entry {entry} is not a Tensorflow tensor of type tf.string.")
		elif len(tf.shape(entry)) != 1:
			raise ValueError(f"Entry {entry} shape cardinality is {len(tf.shape(entry))} instead of 1. Input entry shape {tf.shape(entry)}.")
		if not isinstance(continuation, tf.Tensor) or continuation.dtype != tf.string:
			raise TypeError(f"Entry {continuation} is not a Tensorflow tensor of type tf.string.")
		elif len(tf.shape(continuation)) != 1:
			raise ValueError(f"Entry {continuation} shape cardinality is {len(tf.shape(continuation))} instead of 1. Inpue entry shape {tf.shape(continuation)}.")
		
		# Query the database to see if any of the values already exist.
		existing_keys = self.get(entry, verbose=verbose)

		# Save to database (key, value) store.
		for i in range(tf.shape(entry)[0]):
			# Embed entry with BERT model.
			emb_key = self.encode_text(
				tf.expand_dims(entry[i], axis=0) # Needs to be shape (1,)
			)

			# If key doesnt exist assign data in database. Otherwise,
			# override existing value.
			if existing_keys[i] is None:
				self.data[emb_key.ref()] = (entry[i], continuation[i])
			else:
				self.data[existing_keys[i][0]] = (entry[i], continuation[i])


	# Delete (key, value) pairs from the database given the keys.
	# @param: keys, string tensor containing the text entries to be
	#	removed from the database.
	# @param: verbose, boolean that tells whether the "key not found" 
	#	message in get() should be printed out. Default is False. 
	# @return: returns nothing.
	def remove(self, keys, verbose=False):
		# Type and tensor shape checking.
		if not isinstance(keys, tf.Tensor) or keys.dtype != tf.string:
			raise TypeError(f"Entry {keys} is not a Tensorflow tensor of type tf.string.")
		elif len(tf.shape(keys)) != 1:
			raise ValueError(f"Entry {keys} shape cardinality is {len(tf.shape(keys))} instead of 1. Entry shape {tf.shape(keys)}.")

		# Query the database to see if any of the values already exist.
		existing_keys = self.get(keys, verbose=verbose)

		# Delete embedding from database.
		for i in range(tf.shape(keys)[0]):
			# If key doesnt exist print error message. Otherwise,
			# delete the entry from the database.
			if existing_keys[i] is not None:
				del self.data[existing_keys[i][0]]
			else:
				print(f"Key {keys[i]} does not exist in database.")


	# Retrieve the k nearest neighbors from the given input.
	# @param: input_tensor, tensor containing the text entries to be
	#	retrieved from the database.
	# @param: k, the number of neighbors to return. Default is 2.
	# @return: returns the neighbors containing the neighbor entry and
	#	continuation pairs.
	def get_knn(self, input_tensor, k=2):
		# Load keys.
		database_keys = list(self.data.keys())

		# Return database (key, value) pairs if the number of items in
		# the database is less than k.
		if len(database_keys) < k:
			return [(k, v) for k, v in self.data.items()]

		# Get the euclidian distance between the input and the other
		# embeddings in the database.
		top_k_matches = []
		for i in range(tf.shape(input_tensor)[0]):
			# Embed entry with BERT model.
			emb_key = self.encode_text(input_tensor)

			deltas = [
				tf.norm(
					emb_key - key.deref(), ord="euclidean"
				)
				for key in database_keys
			]

			# The K Nearest Neighbors (KNN) is the embeddings with the
			# smallest distance to the input. Here, top_k retrieves the
			# largest k values and their indices from a tensor/list. To
			# retrieve the smallest k values, negate the euclidian
			# distance values and call top_k on that list.
			top_k = tf.math.top_k(tf.math.negative(deltas), k=k)
			for k in top_k.indices.numpy():
				top_k_matches.append(
					(database_keys[k], self.data[database_keys[k]])
				)

		# Return the embeddings as well as the 
		return top_k_matches


	# Load the database.
	# @param: path, string path to save the data.
	# @return: returns nothing.
	def load(self, path):
		# Print error if path directory does not already exist.
		if not os.path.exists(path) or not os.path.isdir(path):
			print(f"Could not find path folder to load from {path}. Failed to load database.")
			return

		# Load from full path with pickle.
		full_path = os.path.join(path, self.file)
		self.data = pickle.load(open(full_path, "rb"))


	# Save the database.
	# @param: path, string path to load the data.
	# @return: returns nothing.
	def save(self, path):
		# Create path directory if it doesnt exist already.
		if not os.path.exists(path):
			os.makedirs(path, exist_ok=True)

		# Write to full path with pickle.
		full_path = os.path.join(path, self.file)
		pickle.dump(self.data, open(full_path, "wb+"))


# Given a list of list of tensors containing the chunked tokens (from
# the text).
# @param: dataset, List[List[Tensor(string)]], each element is a list
#	of string tensors from the input dataset. This is the expected
#	format for all data that is to be loaded into the BERT database.
# @param: db, the BERT database that this data is going to be loaded
#	into.
# @return: returns the BERT database (db) instance now loaded with all
#	the data passed in from the list of string tensors (dataset).
def load_dataset_to_db(dataset, db):
	# Iterate through each item in the main list. These items can be
	# considered sections (usually they're a paragraph of text).
	print("Loading dataset to BERT database...")
	for i in tqdm(range(len(dataset))):
		section = dataset[i]

		if len(section) == 1:
			# In the event that there was only 1 chunk within a
			# section, the continuation is just a blank string.
			db.add((
				tf.expand_dims(section[0], axis=0), 
				tf.convert_to_tensor([""], dtype=tf.string)
			))
		else:
			# Iterate through each chunk within the section. The
			# neighbor and continuation entries are the two immediate
			# chunks next to eachother in the section.
			for j in range(len(section) - 1):
				db.add((
					tf.expand_dims(section[j], axis=0),
					tf.expand_dims(section[j + 1], axis=0),
				))

	# Return the database.
	print("Dataset loaded.")
	return db


if __name__ == '__main__':
	# Print tensorflow version (should be 2.7) and initialize a new
	# database object (as well as a new BERT model within it).
	print(tf.__version__)
	db = BERTDatabase(initialize_model=True)

	# Input to model is tensor of shape [batch_size,]. Each entry, both
	# the neighbor and its continuation, should no longer than 64
	# tokens in length (different from what their raw string will look
	# like). An input with batch_size 4 should look like this:
	# tf.Tensor([sentence1, sentence2, sentence3, sentence4],
	# dtype=tf.string) where each sentence is a string.
	entries = [
		"Hello there.", "I am the Senate.", 
		"I don't like sand.", "Lightsabers make a fine",
		"Lambda class shuttle", "This is the way.",
		"You are on the council,", "master yoda",
		"Help me obi wan kenobi", "It's not the jedi way.",
	]
	values = [
		"General Kenobi!", "It's treason then.", 
		"It's coarse and rough.", "addition to my collection.",
		"is on approach from scarif", "This is the way.",
		"but we do not grant you the rank of master", "you survived",
		"you're my only hope.", "Dewit.",
	]
	invalid_entries = [
		"Short for a storm trooper.", "You were the chosen one!",
		"I have no such weaknesses."
	]

	# Input data in a batch.
	neighbors = tf.convert_to_tensor(
		entries[:6],
		dtype=tf.string
	)
	continuations = tf.convert_to_tensor(
		values[:6],
		dtype=tf.string
	)
	pairs = (neighbors, continuations)
	db.add(pairs)

	# Input data one value at a time.
	neighbors = tf.convert_to_tensor(
		[entries[7]],
		dtype=tf.string
	)
	continuations = tf.convert_to_tensor(
		[values[7]],
		dtype=tf.string
	)
	pairs = (neighbors, continuations)
	db.add(pairs)

	# Input data using a key that already exists.
	neighbors = tf.convert_to_tensor(
		entries[4:7],
		dtype=tf.string
	)
	continuations = tf.convert_to_tensor(
		values[4:7],
		dtype=tf.string
	)
	pairs = (neighbors, continuations)
	db.add(pairs)

	print("-" * 72)

	# Attempt to retrieve data with a valid key (batch retrieval).
	valid_response = db.get(
		tf.convert_to_tensor([entries[0], entries[3]], dtype=tf.string)
	)
	print(valid_response)

	# Attempt to retrieve data with a valid key (single retrieval).
	valid_response = db.get(
		tf.convert_to_tensor([entries[4]], dtype=tf.string)
	)
	print(valid_response)

	# Attempt to retrieve data with an invalid key (batch retrieval).
	invalid_response = db.get(
		tf.convert_to_tensor(invalid_entries, dtype=tf.string)
	)
	print(invalid_response)

	# Attempt to retrieve data with an invalid key (single retrieval).
	invalid_response = db.get(
		tf.convert_to_tensor([invalid_entries[1]], dtype=tf.string)
	)
	print(invalid_response)

	# Attempt to retrieve data with both invald and valid key (batch
	# retrieval only).
	mixed_response = db.get(
		tf.convert_to_tensor(
			[invalid_entries[0], entries[3]], dtype=tf.string
		)
	)
	print(mixed_response)

	# Another Note: Similar to the above note regarding how
	# initializing two of the same tensors will yeild different hashes
	# even if they are initialized to be exactly the same, the same
	# extends to the input tensors being passed into the database. Note
	# how passing in an entry 1 at a time will not find entries that
	# were entered a batch at a time. This appears to be because
	# passing the tensor a batch at a time to the BERT encoding will
	# result in different rounding compared to passing a tensor with a
	# single value in. This conclusion was reached by taking the
	# difference between the same entry encoded in a batch tensor vs
	# by itself in another tensor. The embeddings were all values that
	# went to the power of e-2. The non-zero values in the difference
	# were at the power of e-8. The solution to this problem will be to
	# pass in values as a list of strings or force tensors to be
	# sent into another tensor, one sample at a time, before encoding.

	print("-" * 72)

	# Attempt to remove an entry with a valid key (batch removal).
	valid_response = db.remove(
		tf.convert_to_tensor([entries[2]], dtype=tf.string)
	)

	# Attempt to remove an entry with a valid key (single removal).
	valid_response = db.remove(
		tf.convert_to_tensor(entries[3:5], dtype=tf.string)
	)

	# Attempt to remove an entry with an invalid key (batch removal).
	invalid_response = db.remove(
		tf.convert_to_tensor(
			[invalid_entries[0]], dtype=tf.string
		)
	)

	# Attempt to remove an entry with an invalid key (single removal).
	invalid_response = db.remove(
		tf.convert_to_tensor(
			invalid_entries[:2], dtype=tf.string
		)
	)

	# Attempt to remove an entry with both a valid and an invalid key
	# (batch removal only).
	mixed_response = db.remove(
		tf.convert_to_tensor(
			[invalid_entries[-1], entries[-1]], 
			dtype=tf.string
		)
	)

	print("-" * 72)

	# Retrieve the K Nearest Neighbors from the database given the
	# input text. KNN doesnt require the input text to be a part of the
	# database.

	# Start by populating the entire database with all (key, value)
	# pairs.
	db.add(tf.convert_to_tensor((entries, values), dtype=tf.string))
	
	# Get KNN entries from batch.
	knn = db.get_knn(
		tf.convert_to_tensor(
			[
				"I don't like sand.", 
				"I am a jedi, like my father before me."
			], 
			dtype=tf.string
		)
	)
	print(knn)

	# Get KNN entries from a single sample.
	knn = db.get_knn(
		tf.convert_to_tensor(
			["The senate will decide your fate."], dtype=tf.string
		)
	)
	print(knn)

	print("-" * 72)

	# Test database save function.
	db.save("./BERT_DB")

	# Test database load function.
	db_copy = BERTDatabase()
	db_copy.load("./BERT_DB")

	# Verify database save and load match.
	values1 = [v for k, v in db.data.items()]
	values2 = [v for k, v in db_copy.data.items()]
	value_comparisions = [v1 == v2 for v1, v2 in zip(values1, values2)]

	keys1 = [k.deref() for k in db.data.keys()]
	keys2 = [k.deref() for k in db_copy.data.keys()]
	key_comparisions = [
		tf.reduce_all(tf.equal(k1, k2)) 
		for k1, k2 in zip(keys1, keys2)
	]

	# This is a weak way of comparing the databases to see if they
	# match, but it's the best I've got.
	all_keys_match = tf.reduce_all(value_comparisions).numpy()
	all_values_match = tf.reduce_all(key_comparisions).numpy()
	print(f"Saved database matches loaded database: {all_keys_match == all_values_match}")

	# Exit the program.
	exit(0)