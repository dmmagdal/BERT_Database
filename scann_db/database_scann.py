# database_scann.py
# Implement BERT KNN database with SCANN.
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
import tensorflow_recommenders as tfrs
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

		# Data storage. Stores (entry, continuation) tuple, BERT
		# embeddings are computed upon indexing.
		self.data = []

		# ScaNN model for approximate nearest neighbor search. Use BERT
		# model for query model. K = 2.
		self.scann = tfrs.layers.factorized_top_k.ScaNN(
			query_model=self.bert,
			k=2
		)

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


	# Retrieve the value of a (key, value) pair from the database given
	# the keys.
	# @param: keys, string tensor containing the text entries to be
	#	retrieved from the database.
	# @param: verbose, boolean that tells whether the "key not found" 
	#	message should be printed out. Default is True. All class 
	#	functions that call get() (ie add(), remove(), etc) will pass 
	#	in their verbos value.
	# @return: returns a list of (key, value) pair tensors from the database.
	def get(self, keys):

		# Iterate trough storage data.
		pass


	# Add (key, value) entry pairs to the database.
	# @param: input_pairs, a tuple of string tensors containing the
	#	primary entry and its continuation.
	# @param: verbose, boolean that tells whether the "key not found" 
	#	message in get() should be printed out. Default is False. 
	# @return: returns nothing.
	def add(self, input_pairs):
		# Add to data storage. Override any (entry, continuation)
		# values if an entry already exists.

		# Re-index.
		pass


	# Delete (key, value) pairs from the database given the keys.
	# @param: keys, string tensor containing the text entries to be
	#	removed from the database.
	# @param: verbose, boolean that tells whether the "key not found" 
	#	message in get() should be printed out. Default is False. 
	# @return: returns nothing.
	def remove(self, keys):
		# Remove entries from data storage where key (entry) is found
		# in the data.

		# Re-index.
		pass


	# Retrieve the k nearest neighbors from the given input.
	# @param: input_tensor, tensor containing the text entries to be
	#	retrieved from the database.
	# @param: k, the number of neighbors to return. Default is 2.
	# @return: returns the neighbors containing the neighbor entry and
	#	continuation pairs.
	def get_knn(self, input_tensor):
		pass


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