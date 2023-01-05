# database_annoy.py
# Implement BERT KNN database with annoy.
# Source on annoy: https://github.com/spotify/annoy
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
from annoy import AnnoyIndex


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

		# Data storage. Will implement strategy 1 (the 
		# List[Tuple(str, str)] format) for now.
		self.data = []

		# Ideas for self.data format/structure
		# 1) self.data = List[Tuple(str, str)]
		#	-> tuple item in self.data is (entry, continuation).
		# 	-> self.data is updated in linear time (average case).
		#	-> index trained by calling train(). Will train on existing
		#		self.data list.
		#	-> updates to self.data do NOT call training on index
		#		unless specified in arguments
		#	-> updating a tuple in self.data (ie entry already exists
		#		but continuation is changed) will cause no updates/
		#		changes to index because embedding is still the same.
		#	-> removing a tuple from self.data will require index to be
		#		updated. index.add() adds vectors to index. 
		#		index.search() searches for k nearest neighbors for all
		#		queried indices. index.remove_ids() will remove the
		#		specified indices from the index.
		#	-> create embeddings requires converting entry strings in
		#		tuples to tensor. Do this on an as-needed basis? OR
		#		add that as a part of the tuple (entry: str,
		#		continuation: str, embedding: tf.Tensor). First causes
		#		a spike in computing whenever retraining the index. The
		#		latter causes more memory strain.
		# 2) self.data = Dict(key: str, value: str)
		#	-> key is the entry and value is the continuation.
		#	-> self.data is updated in linear time (worst case).
		#	-> updates to self.data are faster (ideally).
		#	-> regarding embeddings, if we decide to store embeddings
		#		in self.data, value is now a Tuple(str, tf.Tensor)
		#		containing the continuation and embedding.
		#	-> downside of using a dict for self.data is that the
		#		indices may change with changes to they keys in 
		#		self.data. So updates to self.data may be faster but 
		#		tracking may be more complicated.
		# 3) self.data = tf.data.Dataset({"entry", "continuation"})
		#	-> similar properties to self.data = List
		#	-> reduced memory footprint due to batch(), cache(), and
		#		prefetch() functions.
		#	-> map() function applies targeted function (can be used
		#		for creating embeddings quickly).
		#	-> can load/store as TFRecords file (may have smaller 
		#		storage).

		'''
		from annoy import AnnoyIndex
		import random

		# Set the number of features (length of the vector).
		f = 40  # Length of item vector that will be indexed

		# Initialize an index. Pass in the number of features as well
		# as the distanace metric. Initialize random vectors (lists)
		# and add them to the index with a given value i.
		t = AnnoyIndex(f, 'angular')
		for i in range(1000):
			v = [random.gauss(0, 1) for z in range(f)]
			t.add_item(i, v)

		# Build the index. Be sure to specify the number of trees to
		# build the index with. Then save the index.
		t.build(10) # 10 trees
		t.save('test.ann')

		# Initialize a new index with the same features as the last one
		# as well as the same distance metric. Load the previous index.
		# Then get the nearest 1,000 neighbors to the 0th item in the
		# index.
		u = AnnoyIndex(f, 'angular')
		u.load('test.ann') # super fast, will just mmap the file
		print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors

		'''

		# Create index with annoy using AnnoyIndex. Set the dimension 
		# (ndims) to be the same as the BERT embeddings (768). The 
		# nlist parameter (used for specifying the number of trees in
		# the index on build time) can be something else, but for 1M to
		# 1T tokens, nlist of 10,000 sounds reasonable. May tweak later.
		self.ndims = self.bert.outputs[0].shape[-1]
		self.nlist = 10_000
		self.metric = "dot" # [angular, euclidean, manhattan, hamming, dot]
		self.index = AnnoyIndex(self.ndims, self.metric) 
		self.k = 2
		self.built = False

		# Save file name for data.
		self.file = "bert_db_data.pkl"
		self.index_file = "bert_db_index.ann"


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


	# Build annoy index.
	# @param: takes no arguments.
	# @return: returns nothing.
	def train_index(self):
		# Because calling `build()` on the index will freeze the index
		# in annoy, assume all items that are meant to be in the index
		# have been added to the index.
		print(f"Building index...")
		self.index.build(self.nlist)
		self.built = True
		print(f"Index compiled successfully.")


	# Retrieve (entry, continuation) pair from the database given the
	# keys (entry).
	# @param: keys, list of strings containing the text entries to be
	#	retrieved from the database.
	# @param: verbose, boolean that tells whether the "key not found" 
	#	message should be printed out. Default is True.
	# @return: returns a list of (entry, continuation) pairs from the 
	#	database.
	def get(self, keys, verbose=False):
		# If keys passed in is just a string, store the value in a
		# list.
		if isinstance(keys, str):
			keys = [keys]

		# A few assertions to for type checking and validating the keys
		# list is not empty.
		assert isinstance(keys, list), f"Queries expected to be a list of strings."
		assert len(keys) > 0, f"Require number of queries to be greater than 0. Recieved: {len(keys)}"
		assert all(isinstance(query, str) for query in keys), f"All queries in list are expected to be strings."

		# Create a temporary list of all the entries to be parallel
		# with the (entry, continuation) tuples in self.data.
		entries = [entry for (entry, continuation) in self.data]

		# Iterate through each key, searching for their corresponding
		# tuple in self.data
		retrieved_pairs = []
		for key in keys:
			if key in entries:
				# Retrieve (entry, continuation) from self.data.
				key_index = entries.index(key)
				retrieved_pairs.append(self.data[key_index])
			else:
				# Append None to the list (keeps things in order with
				# respect to the original queries).
				retrieved_pairs.append(None)
				if verbose:
					# Output that the key could not be found in
					# self.data if verbose is True.
					print(f"Entry {key} was not found in database.")

		# Return the list of retrieved (entry, continuation) pairs.
		return retrieved_pairs


	# Add/updates (entry, continuation) pairs to the database.
	# @param: input_pairs, a tuple of strings containing the primary 
	#	entry and its continuation.
	# @param: verbose, boolean that tells whether the "updated entry" 
	#	message should be printed out. Default is False.
	# @return: returns nothing.
	def add(self, input_pairs, verbose=False):
		# If input_pairs passed in is just a single tuple, store the
		# value in a list.
		all_text_is_str = all(
			isinstance(text, str) for text in input_pairs
		)
		if isinstance(input_pairs, tuple) and all_text_is_str:
			input_pairs = [input_pairs]

		# A few assertions to for type checking and validating the
		# input_pairs list is not empty.
		assert isinstance(input_pairs, list), f"Input pairs expected to be a list of string tuples."
		assert len(input_pairs) > 0, f"Require number of input pairs to be greater than 0. Recieved: {len(input_pairs)}"
		assert all(
			all(isinstance(text, str) for text in pair) 
			for pair in input_pairs
		), f"Not all inputs are strings in {input_pairs}."

		if self.built:
			print(f"Warning: Database index has already been built. Any new items wont be added to the index.")

		# Create a temporary list of all the entries to be parallel
		# with the (entry, continuation) tuples in self.data.
		entries = [entry for (entry, continuation) in self.data]

		# Add to data storage. Override any (entry, continuation)
		# values if an entry already exists.
		for pair in input_pairs:
			# Extract the entry from the tuple.
			entry = pair[0]
			if entry in entries:
				# Update the continuation in the (entry, continuation)
				# from self.data. No need to update embeddings in the
				# index because the entry is still the same.
				entry_index = entries.index(entry)
				self.data[entry_index] = pair

				if verbose:
					# Output that the key was updated in self.data if
					# verbose is True.
					print(f"Updated {entry} entry to be {pair}")
			else:
				# Skip if this is a new item to add to the
				# index/database and the index is already built.
				if self.built:
					print(f"Warning: New item ({entry}) was detected. Could not add to database.")
					continue

				# Add new (entry, continuation) pair to self.data.
				self.data.append(pair)

				# Convert entry to tensor and get embedding.
				tensor = tf.convert_to_tensor(
					[entry], dtype=tf.string
				)
				embedding = self.bert(tensor)
				embedding = tf.squeeze(embedding, axis=0)

				# The index of the entry is the length of self.data.
				entry_index = len(self.data) - 1

				# Add new embeddings to index.
				self.index.add_item(entry_index, embedding)

			# Important assertion that the number of items in the index
			# is equal to the number of (entry, continuation) tuple
			# strings in self.data.
			assert len(self.data) == self.index.get_n_items(), f"Number of embeddings in index should match dataset size. Index embeddings {self.index.get_n_items()}, Dataset size: {len(self.data)}"


	# Retrieve the k nearest neighbors from the given input.
	# @param: input_texts, list of strings containing the text entries
	#	to used as a query to the database.
	# @param: k, the number of neighbors to return. Default is 2.
	# @return: returns the neighbors containing the neighbor entry and
	#	continuation pairs.
	def get_knn(self, input_texts):
		# If input_texts passed in is just a string, store the value in
		# a list.
		if isinstance(input_texts, str):
			input_texts = [input_texts]

		# A few assertions to for type checking and validating the
		# input_texts list is not empty.
		assert isinstance(input_texts, list), f"Queries expected to be a list of strings."
		assert len(input_texts) > 0, f"Require number of queries to be greater than 0. Recieved: {len(input_texts)}"
		assert all(isinstance(query, str) for query in input_texts), f"All queries in list are expected to be strings."

		# Convert string list to tensor.
		input_tensor = tf.convert_to_tensor(
			input_texts, dtype=tf.string
		)
		
		# Calculate embeddings embeddings.
		embeddings = self.bert(input_tensor)

		# Perform KNN search on each embedding.
		nearest_neighbors = []
		for embedding in embeddings:
			# Conduct search on indices. Note that what is returned is
			# a list of the indices in the index (and also self.data
			# since that was how the data was added to the index). So
			# for n = self.k = 2, results will be a list of len 2
			# containing the indices which are int values.
			results = self.index.get_nns_by_vector(embedding.numpy(), n=self.k)

			# Return the k nearest neighbors (entry, continuation)
			# tuple string.
			nearest_neighbors.append(
				[self.data[idx] for idx in results]
			)

		return nearest_neighbors


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
		full_index_path = os.path.join(path, self.index_file)

		# Load data.
		with open(os.path.join(path, "data.json"), "r") as f:
			self.data = json.load(f)
		self.data = [tuple(pair) for pair in self.data]

		# Load index. Any loaded index is automatically assumed to be
		# already built/trained.
		self.index.load(full_index_path)
		self.built = True


	# Save the database.
	# @param: path, string path to load the data.
	# @return: returns nothing.
	def save(self, path):
		# Create path directory if it doesnt exist already.
		if not os.path.exists(path):
			os.makedirs(path, exist_ok=True)

		# Write to full path with pickle.
		full_path = os.path.join(path, self.file)
		full_index_path = os.path.join(path, self.index_file)

		# Another note here. Currently, on a small dataset (see test
		# code below under if __name__ == '__main__'), .pkl and .index
		# files are the same size for the index. It appears neither is
		# more "efficient" in terms of space usage. The same also can
		# be observed for the .pkl and .json files for the data. Will
		# currently use native .json and .index file formats for the
		# respective data unless there are updates or new findings to
		# show a more efficient way to store each.

		# Save data.
		with open(os.path.join(path, "data.json"), "w+") as f:
			json.dump(self.data, f, indent=4)

		# Save index.
		self.index.save(full_index_path)


# Given a list of list of strings containing the chunked tokens (from
# the text), populate the specified database.
# @param: dataset, List[List[str]], each element is a list of strings
#	from the input dataset. This is the expected format for all data
#	that is to be loaded into the BERT database.
# @param: db, the BERT database that this data is going to be loaded
#	into.
# @param: build_index, boolean, whether to build the index after
#	loading the data to the databse.
# @return: returns the BERT database (db) instance now loaded with all
#	the data passed in from the list of string tensors (dataset).
def load_dataset_to_db(dataset, db, build_index=False):
	# Iterate through each item in the main list. These items can be
	# considered sections (usually they're a paragraph of text).
	print("Loading dataset to BERT database...")
	for i in tqdm(range(len(dataset))):
		section = dataset[i]

		if len(section) == 1:
			# In the event that there was only 1 chunk within a
			# section, the continuation is just a blank string.
			db.add((section[0], ""))
		else:
			# Iterate through each chunk within the section. The
			# neighbor and continuation entries are the two immediate
			# chunks next to eachother in the section.
			data = [
				(section[j], section[j + 1]) 
				for j in range(len(section) - 1)
			]
			db.add(data)

	# Train the index if specified.
	if build_index:
		db.train_index()

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
	# [sentence1, sentence2, sentence3, sentence4] where each sentence
	# is a string.
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

	print("-" * 72)

	# Add data to index (and database).
	print("Adding (entry, continuation) pairs to index...")

	# Input data in a batch.
	neighbors = entries[:6]
	continuations = values[:6]
	pairs = list(zip(neighbors, continuations))
	db.add(pairs)

	# Input data one value at a time.
	neighbors = [entries[7]]
	continuations = [values[7]]
	pairs = list(zip(neighbors, continuations))
	db.add(pairs)

	# Input data using a key that already exists.
	neighbors = entries[4:6]
	continuations = values[4:6]
	pairs = list(zip(neighbors, continuations))
	db.add(pairs)

	# Add the rest of the data (if there are no issues so far).
	db.add(list(zip(entries, values)))
	print("Data ((entry, continuation) pairs) added to index.")

	print("-" * 72)

	# Build/train index.
	print("Building index...")

	db.train_index()
	print("Index built.")

	print("-" * 72)

	# Try and add data to the index/database after build (should print
	# warning message).
	print("Attempt to add a new (entry, continuation) pair to index after build...")

	entry = "I know who the best president of the united states is"
	continuation = "Joe Bama"
	db.add([(entry, continuation)])
	
	print("-" * 72)

	# Retrieve data from the database (not the index).
	print("Retrieving (entry, continuation) pairs from the database...")

	# Attempt to retrieve data with a valid key (batch retrieval).
	valid_response = db.get([entries[0], entries[3]])
	print(valid_response)

	# Attempt to retrieve data with a valid key (single retrieval).
	valid_response = db.get([entries[4]])
	print(valid_response)

	# Attempt to retrieve data with an invalid key (batch retrieval).
	invalid_response = db.get(invalid_entries)
	print(invalid_response)

	# Attempt to retrieve data with an invalid key (single retrieval).
	invalid_response = db.get([invalid_entries[1]])
	print(invalid_response)

	# Attempt to retrieve data with both invald and valid key (batch
	# retrieval only).
	mixed_response = db.get([invalid_entries[0], entries[3]])
	print(mixed_response)
	print("All valid entries retrieved.")

	print("-" * 72)

	# Retrieve the K Nearest Neighbors from the database given the
	# input text. KNN doesnt require the input text to be a part of the
	# database.
	print("Querying KNN (entry, continuation) pairs from the index...")
	
	# Get KNN entries from batch.
	print("Query:")
	print(
		[
			"I don't like sand.", 
			"I am a jedi, like my father before me."
		]
	)
	knn = db.get_knn(
		[
			"I don't like sand.", 
			"I am a jedi, like my father before me."
		]
	)
	print("Results:")
	print(knn)

	# Get KNN entries from a single sample.
	print("Query:")
	print(["The senate will decide your fate."])
	knn = db.get_knn(["The senate will decide your fate."])
	print("Results:")
	print(knn)
	print("KNN queries completed.")

	print("-" * 72)

	# Test database save function.
	print("Saving database and index...")
	db.save("./BERT_DB")
	print("Database and index saved successfully.")

	# Test database load function.
	print("Loading database and index...")
	db_copy = BERTDatabase()
	db_copy.load("./BERT_DB")
	print("Database and index loaded successfully.")

	# Verify database content is the same. This is not the best way
	# to validate loading because the index can be in a different
	# "trained" state (index in the loaded database is always retrained
	# on the full dataset which may not be true for the initial
	# database).
	print(f"Database contents align: {db.data == db_copy.data}")

	# Query the copy database. Should expect to see the exact same
	# results as the original.
	print("Query to copy:")
	knn = db_copy.get_knn(
		[
			"I don't like sand.", 
			"I am a jedi, like my father before me."
		]
	)
	print("Results:")
	print(knn)

	# Try and add data to the index/database after load (should print
	# warning message).
	print("Attempt to add a new (entry, continuation) pair to index after build...")

	entry = "I know who the best president of the united states is"
	continuation = "Joe Bama"
	db_copy.add([(entry, continuation)])
	
	print("-" * 72)

	# Exit the program.
	exit(0)