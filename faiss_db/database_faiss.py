# database_faiss.py
# Implement BERT KNN database with faiss.
# Source on faiss: https://www.youtube.com/watch?v=sKyvsdEv6rk
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import os
import json
import pickle
import faiss
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow_text as text
from tqdm import tqdm


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
		# Dimensionality of the vectors/embeddings. 768 for BERT.
		dims = 768

		# Flat index (vectors using L2 or euclidean distance for
		# similarity).
		index = faiss.IndexFlatL2(dims)

		# Number of embeddings to return.
		k = 2

		# Set training on index. We train index on things such as
		# clustering. We can check if an index needs training or is
		# trained with the is_trained attribute. Flat L2 index doing
		# nothing special doesnt need to be trained (is_trained =
		# True).
		print(index.is_trained)

		embedding = self.bert(tf.convert_to_tensor(
			["good morning"], dtype=tf.string
		))

		# Add embeddings to index (batch_size, 768). Can add a batch at
		# a time.
		index.add(embedding)

		# Can check if embeddings were added properly by looking at
		# the ntotal values.
		print(index.ntotal)

		# Query vector.
		query_vector = self.bert(tf.convert_to_tensor(
			["hello there"], dtype=tf.string
		))

		# Search index. I is the indices of the input lines and D is
		# the distances. To get the values you can use:
		# [f"{i} for {lines[i]} in I[0]"]
		# Downside is that this takes a long time.
		D, I = index.search(query_vector, k=k)

		# Can use approximate search by adding partitioning
		# (number of voroni cells)
		nlist = 50

		# Define quantizer (endless step in the indexing process),
		# still using the L2 distance
		quantizer = faiss.IndexFlatL2(dims)

		# Define the index. This is an index that needs to be trained
		# because it does clustering to create the voroni cells.
		index = faiss.IndexIVFFlat(quantizer, dims, nlist)

		# Should see is_trained = False
		print(index.is_trained)

		# To train the index, call index.train and pass all the
		# sentence embeddings.
		index.train(embedding)

		# Should see is_trained = True now.
		print(index.is_trained)

		# Add sentence embeddings (same as before).
		index.add(embedding)
		print(index.ntotal)

		# Query the index. This should be MUCH faster (but may not be
		# as accurate, hence it's an "approximate" search and the
		# results may vary between the exhaustive search).
		D, I = index.search(query_vector, k=k)

		# Setting the nprobe value will expand the search from 1
		# centroid in a voroni cell (default) to multiple (nprobe)
		# centriods. This should increase accuracy at the expense of
		# taking a hit on time.
		index.nprobe = 10

		# There is one more index called the product quantization 
		# index. Note that dims must be a multiple of m.
		m = 8 # Number of sub vectors
		bits = 8 # Number of bits within each centriod for each sub-vector
		quantizer = faiss.IndexFlatL2 # Quantizer
		index = faiss.IndexIVFPQ(quantizer, dims, nlist, m, bits)

		# Will also have to train this kinds of index.
		index.train(embedding)
		index.add(embedding)

		# Query the index. This should be even FASTER (but is not as
		# accurate).
		D, I = index.search(query_vector, k=k)
		'''

		# Create index with faiss using IVFFlat (best combination of
		# speed and accuracy). Set the dimension (ndims) to be the same
		# as the BERT embeddings (768). nlist and nprobe parameters can
		# be set to something else, but for 1M to 1T tokens, nlist of
		# 10,000 and nprobe 100 sounds reasonable. May tweak later.
		self.ndims = self.bert.outputs[0].shape[-1]
		self.nlist = 10_000
		self.quantizer = faiss.IndexFlatL2(self.ndims)
		self.index = faiss.IndexIVFFlat(
			self.quantizer, self.ndims, self.nlist
		)
		self.nprobe = 100
		self.index.nprobe = self.nprobe
		self.k = 2

		# Batch size to pass break up data when training the index.
		self.batch_size = 256 # 512

		# Save file name for data.
		self.file = "bert_db_data.pkl"
		self.index_file = "bert_db_index.pkl"


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


	# Train faiss index.
	# @param: takes no arguments.
	# @return: returns nothing.
	def train_index(self):
		# Store all entries in self.data in a list and convert that
		# list to a string tensor. Tensor shape = (len(self.data),).
		input_tensor = tf.convert_to_tensor(
			[entry for entry, continuation in self.data],
			dtype=tf.string
		)

		# Break up inputs into chunks of 512 or 256 if len(self.data)
		# is larger than the self.batch_size threshold. Pass the inputs
		# (chunks or otherwise) to the BERT model for embedding.
		if len(self.data) > self.batch_size:
			input_tensors = list(
				divide_chunks(input_tensor, self.batch_size)
			)
			embeddings_list = [
				self.bert(input_tensor)
				for input_tensor in input_tensors
			]
			embeddings = tf.concat(embeddings_list, axis=0)
		else:
			embeddings = self.bert(input_tensor)

		# Convert embeddings tensor to list? numpy array? Keep as
		# tf.Tensor?
		# Confirmed, need to convert embeddings to 2D numpy array.
		# Note that faiss index accepts 2D numpy array of shape
		# (batch_size, ndims) or in this case (len(embeddings), 768).

		# Reset the nlist and nprobe values to adapt appropriately to
		# the size of the dataset for training. Faiss will throw errors
		# if len(embeddings) < self.index.nlist (the dataset is smaller
		# than the number of clusters assigned to it).
		# self.index.ntotal = number of entries/vectors in index.
		# len(self.data) = number of entries/pairs in known dataset.
		# len(embeddings) = embeddings.shape[0] = len(self.data).
		if len(embeddings) <= self.nlist:
			if len(embeddings) < 100:
				self.index.nlist = 1
				self.index.nprobe = 1
			else:
				self.index.nlist = len(embeddings) // 50\
					if len(embeddings) // 50 > 0 else 1
				self.index.nprobe = self.index.nlist // 100\
					if self.index.nlist // 100 > 0 else 1
		else:
			self.index.nlist = self.nlist
			self.index.nprobe = self.nprobe

		# Train index on embeddings.
		self.index.train(embeddings.numpy())


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
	# @param: retrain, boolean that tells whether to retrain the index
	#	after applying changes. Default is False.
	# @return: returns nothing.
	def add(self, input_pairs, verbose=False, retrain=False):
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

		# Create a temporary list of all the entries to be parallel
		# with the (entry, continuation) tuples in self.data.
		entries = [entry for (entry, continuation) in self.data]

		# Add to data storage. Override any (entry, continuation)
		# values if an entry already exists.
		embeddings = []
		for pair in input_pairs:
			# Extract the entry from the tuple.
			entry = pair[0]
			if entry in entries:
				# Update the continuation in the (entry, continuation)
				# from self.data. No need to update embeddings in the
				# index because the entry is still the same.
				key_index = entries.index(entry)
				self.data[key_index] = pair

				if verbose:
					# Output that the key was updated in self.data if
					# verbose is True.
					print(f"Updated {entry} entry to be {pair}")
			else:
				# Add new (entry, continuation) pair to self.data.
				self.data.append(pair)

				# Convert entry to tensor and get embedding.
				tensor = tf.convert_to_tensor(
					[entry], dtype=tf.string
				)
				embedding = self.bert(tensor)
				embeddings.append(tf.squeeze(embedding, axis=0))

		# Retrain the index if specified. Note that the index MUST be
		# trained on the data embeddings if it is empty BEFORE adding
		# the data.
		if retrain:
			self.train_index()

		# Add new embeddings to index. Be sure to stack the tensor
		# embeddings to create a new tensor of shape
		# (len(embeddings), 768).
		if len(embeddings) != 0:
			self.index.add(tf.stack(embeddings, axis=0).numpy())
			assert len(self.data) == self.index.ntotal, f"Number of embedings in index should match dataset size. Index embeddings {self.index.ntotal}, Dataset size: {len(self.data)}"


	# Delete (entry, continuation) pairs from the database given the
	# keys (entry).
	# @param: keys, list of strings containing the text entries to be
	#	removed from the database.
	# @param: verbose, boolean that tells whether the "key not found" 
	#	message should be printed out. Default is False. 
	# @param: retrain, boolean that tells whether to retrain the index
	#	after applying changes. Default is False.
	# @return: returns nothing.
	def remove(self, keys, verbose=False, retrain=False):
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

		# Remove entries from data storage where key (entry) is found
		# in the data.
		embeddings = []
		for key in keys:
			if key in entries:
				# Convert entry to tensor and get embedding.
				tensor = tf.convert_to_tensor([key], dtype=tf.string)
				embedding = self.bert(tensor)

				# Delete (entry, continuation) from self.data and the
				# temporary entries list.
				key_index = entries.index(key)
				self.data.pop(key_index)
				entries.pop(key_index)

				# Add to embeddings list.
				embeddings.append(tf.squeeze(embedding, axis=0))
			else:
				if verbose:
					# Output that the key could not be found in
					# self.data if verbose is True.
					print(f"Entry {key} was not found in database.")

		# Remove embeddings from index. Be sure to stack the tensor
		# embeddings to create a new tensor of shape
		# (len(embeddings), 768).
		if len(embeddings) != 0:
			# Unable to use remove_ids(). Documentation around faiss is
			# hard to read so understanding why that function does not
			# work is very tough. Did see that reset() should clear out
			# the contents of an index but keep any learned parameters
			# (https://github.com/facebookresearch/faiss/issues/329).
			# If that is correct, then this does provide a solution to
			# the problem, but makes removes very expensive because
			# all items in self.data must be re-added to the index.
			#self.index.remove_ids(tf.stack(embeddings, axis=0).numpy())
			self.index.reset()
			self.add(self.data)

		# Retrain the index if specified.
		if retrain:
			self.train_index()


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

		# Conduct search on indices.
		D, I = self.index.search(embeddings.numpy(), k=self.k)

		# For each input string, return the k nearest neighbors
		# (entry, continuation) tuple string.
		nearest_neighbors = []
		for i in range(len(I)):
			indices = I[i]
			nearest_neighbors.append(
				[
					self.data[idx] for idx in indices
				]
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
		# self.data = pickle.load(open(full_path, "rb"))
		with open(os.path.join(path, "data.json"), "r") as f:
			self.data = json.load(f)
		self.data = [tuple(pair) for pair in self.data]

		# Insert another note. Currently having issues loading in the
		# index from the saved file with faiss. To get the project
		# moving along, the index will not be saved, but will be
		# re-trained with the loaded in data. This will also be
		# another costly decision that I hope to rectify soon in the
		# future.
		self.add(self.data, retrain=True)

		# Load index.
		# self.index = pickle.load(open(full_index_path, "rb"))
		# self.index = faiss.read_index(
		# 	os.path.join(path, "bert_db.index"),
		# 	faiss.IO_FLAG_ONDISK_SAME_DIR
		# )
		# self.index = faiss.deserialize_index(np.load("index.npy"))


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
		# pickle.dump(self.data, open(full_path, "wb+"))
		with open(os.path.join(path, "data.json"), "w+") as f:
			json.dump(self.data, f, indent=4)

		# Save index.
		# pickle.dump(self.index, open(full_index_path, "wb+"))
		# faiss.write_index(
		# 	self.index, os.path.join(path, "bert_db.index")
		# )
		# chunk = faiss.serialize_index(self.index)
		# np.save("index.npy", chunk)
		# print(dir(faiss))
		# print(dir(self.index))


# Divide a iist of texts into chunks of size n.
# @param: text_list, a list of texts (string) that is going to be
#	divided into even chunks.
# @param: n, the max size of the chunk.
# @return: returns (yields) a tensor of shape (n,).
def divide_chunks(text_list, n):
	# Divide up the list of tokens in the text to equal shapes of size
	# n.
	for i in range(0, len(text_list), n):
		yield text_list[i:i + n]


# Given a list of list of strings containing the chunked tokens (from
# the text).
# @param: dataset, List[List[str]], each element is a list of strings
#	from the input dataset. This is the expected format for all data
#	that is to be loaded into the BERT database.
# @param: db, the BERT database that this data is going to be loaded
#	into.
# @param: save_path, the path to save the BERT database (this is for
#	loading very large datasets).
# @return: returns the BERT database (db) instance now loaded with all
#	the data passed in from the list of string tensors (dataset).
def load_dataset_to_db(dataset, db, save_path):
	# Iterate through each item in the main list. These items can be
	# considered sections (usually they're a paragraph of text).
	print("Loading dataset to BERT database...")
	for i in tqdm(range(len(dataset))):
		section = dataset[i]
		initial_train = i == 0 # Train on first addition to the database

		if len(section) == 1:
			# In the event that there was only 1 chunk within a
			# section, the continuation is just a blank string.
			db.add((section[0], ""), retrain=initial_train)
		else:
			# Iterate through each chunk within the section. The
			# neighbor and continuation entries are the two immediate
			# chunks next to eachother in the section.
			data = [
				(section[j], section[j + 1]) 
				for j in range(len(section) - 1)
			]
			db.add(data, retrain=initial_train)

		# Retrain index every 250 or 500 sections (not necessarily
		# every 250/500 entries). 
		if i % 500 == 0:
			db.train_index()

		# Save every 1000 sections (for large datasets).
		if i % 1000 == 0:
			db.save(save_path)

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

	# Initialize and train index.
	db.data = list(zip(entries, values))
	db.train_index()

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
	neighbors = entries[4:7]
	continuations = values[4:7]
	pairs = list(zip(neighbors, continuations))
	db.add(pairs)

	print("-" * 72)

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

	print("-" * 72)

	# Attempt to remove an entry with a valid key (batch removal).
	valid_response = db.remove(entries[3:5])

	# Attempt to remove an entry with a valid key (single removal).
	valid_response = db.remove([entries[2]])

	# Attempt to remove an entry with an invalid key (batch removal).
	invalid_response = db.remove(invalid_entries[:2])

	# Attempt to remove an entry with an invalid key (single removal).
	invalid_response = db.remove([invalid_entries[0]])

	# Attempt to remove an entry with both a valid and an invalid key
	# (batch removal only).
	mixed_response = db.remove([invalid_entries[-1], entries[-1]])

	print("-" * 72)

	# Retrieve the K Nearest Neighbors from the database given the
	# input text. KNN doesnt require the input text to be a part of the
	# database.

	# Start by populating the entire database with all (key, value)
	# pairs. Be sure to reset the contents of the index before
	# populating it.
	db.index.reset()
	db.add(list(zip(entries, values)))
	
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

	print("-" * 72)

	# Test database save function.
	db.save("./BERT_DB")

	# Test database load function.
	db_copy = BERTDatabase()
	db_copy.load("./BERT_DB")

	# Exit the program.
	exit(0)