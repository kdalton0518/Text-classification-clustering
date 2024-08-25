#!/usr/bin/python3


import math, random, string, itertools
from math import sqrt

# for average per row
import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures







# for plotting
from scipy import spatial
from scipy.linalg import svd
from scipy.sparse.linalg import svds

import spacy
from spacy import displacy

# for metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
from matplotlib import pyplot as plt

import logging

logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')

logger = logging


infile_path = "data/"
outfile_path = "data/out/"
my_process_objects = []


files = [
	"C1/article01.txt",
	"C1/article02.txt",
	"C1/article03.txt",
	"C1/article04.txt",
	"C1/article05.txt",
	"C1/article06.txt",
	"C1/article07.txt",
	"C1/article08.txt",
	"C4/article01.txt",
	"C4/article02.txt",
	"C4/article03.txt",
	"C4/article04.txt",
	"C4/article05.txt",
	"C4/article06.txt",
	"C4/article07.txt",
	"C4/article08.txt",
	"C7/article01.txt",
	"C7/article02.txt",
	"C7/article03.txt",
	"C7/article04.txt",
	"C7/article05.txt",
	"C7/article06.txt",
	"C7/article07.txt",
	"C7/article08.txt",
]



class Preprocessor():

	# easy way to increment an index of the class instance
	# https://stackoverflow.com/questions/1045344/how-do-you-create-an-incremental-id-in-a-python-class
	index = itertools.count()

	def __init__(self, file):
		self.file_basename = file
		self.filename = "./" + infile_path + file
		self.out_filename = "./" + outfile_path + file
		self.lines = []
		self.stop_words = set(stopwords.words('english'))
		self.lemmatizer = WordNetLemmatizer()
		self.document = []
		self.document_text = ""
		self.keywords_concepts = []
		self.ngrams = []
		self.ngrams_frequency = {}
		self.index = next(Preprocessor.index)

	def read_file(self):
		# read the file into an array of lines
		logger.info("opening file %s ...", self.filename)

		with open(self.filename) as f:
			self.lines = f.readlines()

			logger.debug("line 0: %s", self.lines[0])

	def filter_stopwords_lemmatize(self):
		logger.info("removing stopwords ...")
		new_lines = []

		for line in self.lines:
			if line is not None:

				# split and tokenize
				old_sentence = word_tokenize(line)

				for word in old_sentence:

					# remove punctuation
					# https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
					exclude = set(string.punctuation)
					word = ''.join(ch for ch in word if ch not in exclude)

					# if its not empty 
					if word is not None and len(word) > 0:

						# remove stopwords
						if word not in self.stop_words:

							# lemmatize 
							# best guess here to treat anything ending in 's'
							#	as a noun, anything else gets verb treatment
							new_word = word
							if word.endswith('s'):
								new_word = self.lemmatizer.lemmatize(word)
							else:
								new_word = self.lemmatizer.lemmatize(word, "v")

							## not entirely sure if i should be lowercasing everything
							# new_word = new_word.lower()

							# and add it to the text document
							self.document.append(new_word)
							# logger.info("%s => %s" % (word,new_word))	

		self.document_text = ' '.join(self.document)

	def apply_ner(self):
		logger.info("applying NER ...")
		NER = spacy.load('en_core_web_sm')

		mytext = NER(self.document_text)

		logger.debug("Found the following entities:")
		for ent in mytext.ents:
			# print(ent.text, ent.start_char, ent.end_char, ent.label_)
			logger.debug("\t %s : %s" % (ent.text, ent.label_))
			this_ent = ent.text

			# if there is one or more spaces in the ENT
			if " " in this_ent:
				# then convert them to underscores in the document text
				new_ent = this_ent.replace(" ","_")

				# save the ENT for later matrix
				self.keywords_concepts.append(new_ent)

				# then also replace the original text document
				self.document_text = self.document_text.replace(this_ent, new_ent)

		# also update the tokenized array
		self.document = word_tokenize(self.document_text)


	# https://www.geeksforgeeks.org/python-bigrams-frequency-in-string/
	def _find_bi_grams(self, text):

		bigrams = zip(text, text[1:])
		for gram in bigrams:

			bigram_string = ' '.join(gram)
			self.ngrams.append(bigram_string)

	def _find_tri_grams(self, text):
		# this doesnt seem to be producing as meaningful result as the bigram :/

		trigrams = zip(text, text[1:], text[2:])
		for gram in trigrams:

			trigram_string = ' '.join(gram)
			self.ngrams.append(trigram_string)

	def sliding_window_merge(self):
		logger.info("using a sliding window to merge remaining phrases ...")

		# ****************************************************
		# BI-GRAMS VS TRI-GRAMS ::
		# 
		# 	I won't use trigrams bc frequencies arent as good
		#		but logic for it is here in this block
		#
		#
		# self.ngrams = []
		#
		# self._find_tri_grams(self.document)
		#
		# for ngram in self.ngrams:
		# 	frequency = self.document_text.count(ngram)
		#
		# 	self.ngrams_frequency['ngram'] = frequency
		# 	print("%s : %s "% (ngram, frequency))
		#
		# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		logger.info("using bi-grams for this, there are more matches...")
		# ngram_dist = nltk.FreqDist(nltk.bigrams(self.document))
		# print(ngram_dist.most_common())

		# i will pick everything with freq > 1 for the merge

		self._find_bi_grams(self.document)
		# print(self.ngrams)

		# dedupe the ngrams
		self.ngrams = list(dict.fromkeys(self.ngrams))

		for ngram in self.ngrams:
			frequency = self.document_text.count(ngram)
		
			self.ngrams_frequency[ngram] = frequency
			#print("%s : %s "% (ngram, frequency))

			# if frequency > 1, merge
			if frequency > 1:
				new_ngram = ngram.replace(" ","_")

				# save the NGRAM for later matrix
				self.keywords_concepts.append(new_ngram)

				# then also replace the original text document
				self.document_text = self.document_text.replace(ngram, new_ngram)	

				print("\t\t %s : %s "% (ngram, frequency))


	def cleanup(self):

		for i in range(len(self.keywords_concepts)):
			self.keywords_concepts[i] = self.keywords_concepts[i].replace("_"," ").lower()
			# print(self.keywords_concepts[i])

		self.ngrams_frequency = {k.replace("_"," ").lower() : v for k, v in self.ngrams_frequency.items()}
		# print(self.ngrams_frequency.items())

		self.document_text = self.document_text.replace("_"," ").lower()
		# print(self.document_text)

				
	def write_output(self):

		logger.info("Writing output file "  + self.out_filename);

		# WRITE THIS FILE WITHOUT ANY UNDERSCORES

		with open(self.out_filename, "w") as outfile:
			for word in self.document_text:

				outfile.write(word.lower())


def write_keywords_concepts_file(P):

	logger.info("Appending to concepts file ...")
	concepts_file = "./" + outfile_path + "concepts.txt"
	
	with open(concepts_file, "a") as f:

		lines = P.keywords_concepts
		for line in lines:
			# print(line)
			f.write(line)
			f.write("\n")



# # collect keywords and terms across all the files
# def generate_term_document_matrix():
class DocuTermMatrix():

	def __init__(self):
		self.keywords_concepts = []
		self.matrix = []
		self.tf_idf_matrix = []
		self.total_documents = len(my_process_objects)
		self.docs_with_keyword = {}

	def consolidate_keywords_concepts(self):
		# read the file into an array of lines
		logger.info("Collecting all of the keywords concepts ...")
		for file_object in my_process_objects:

			for keyword in file_object.keywords_concepts:
				if keyword not in self.keywords_concepts:

					self.keywords_concepts.append(keyword.lower())


	def initialize_matrix(self):
		logger.info("initializing the zero matrix ...")

		# fill with 0s for the correct size matrix
		num_rows = len(my_process_objects)
		num_cols = len(self.keywords_concepts)

		# https://intellipaat.com/community/63426/how-to-create-a-zero-matrix-without-using-numpy
		self.matrix = [([0]*num_cols) for i in range(num_rows)]


	def fill_matrix(self):

		logger.info("Creating the document term matrix ...")
		i = 0
		for i in range(len(my_process_objects)):

			# print(i)
			# print(my_process_objects[i].index)
			# print(my_process_objects[i].filename)

			file_object = my_process_objects[i]

			# convert all the keys to lowercase for now
			the_files_ngrams =  {k.lower(): v for k, v in file_object.ngrams_frequency.items()}
			# print(file_object.ngrams_frequency)


			# iterate over the keywords_concepts list
			for j in range(len(self.keywords_concepts)):

				# if a keyword_concept is in the document_text of the document
				# count the number of times the substring appears
				if self.keywords_concepts[j] in file_object.document_text:

					# https://stackoverflow.com/questions/8899905/count-number-of-occurrences-of-a-substring-in-a-string
					frequency = file_object.document_text.count(self.keywords_concepts[j])
					self.matrix[i][j] = frequency

				# else:
				# 	print("%s not in document text" % self.keywords_concepts[j])
				# 	print(file_object.document_text)


	def _get_tf(self, document_object, row_index, col_index):
		# logger.debug("getting TF for a keyword in %s" % document_object.filename)

		# number of times the term occurs in the current document
		keywd_occurrence_this_document = self.matrix[row_index][col_index]

		# word count of the current document
		this_document_wordcount = len(document_object.document_text)

		# logger.info("number of words in %s : %d" % (document_object.filename, this_document_wordcount))

		# make the tf calculation 
		tf = float(keywd_occurrence_this_document / this_document_wordcount)

		return tf


	def _get_idf(self, keyword, row_index, col_index):
		# logger.info("getting IDF on keyword %s ... " % keyword)
		# math.log uses base e
		# test1 = math.log(20)
		# print(test1)

		num_documents_this_keyword = self.docs_with_keyword[keyword]
		idf = float(math.log(self.total_documents / num_documents_this_keyword))

		return idf

	def _get_num_documents_containing_keyword(self):

		for col_index in range(len(self.keywords_concepts)):

			keyword = self.keywords_concepts[col_index]
			counter = 0

			# for each row of the current column
			for row_index in range(len(my_process_objects)):

				# is the value in the matrix > 0 ?
				if self.matrix[row_index][col_index] > 0:
					counter += 1

			# logger.info("current keyword: %s num_documents %s" % (keyword, counter))

			# add it to the dictionary
			self.docs_with_keyword[keyword] = counter



	def create_tf_idf(self):
		# start with a copy of the document term matrix
		self.tf_idf_matrix = self.matrix

		self._get_num_documents_containing_keyword()

		# for column (keyword) in the matrix
		for col_index in range(len(self.keywords_concepts)):

			keyword = self.keywords_concepts[col_index]
			counter = 0

			# logger.info("current keyword: %s" % keyword)

			# for each row of the current column
			for row_index in range(len(my_process_objects)):

				document = my_process_objects[row_index]

				# logger.info("Column: %s | Row: %s" % (keyword, document.index))

				tf = self._get_tf(document, row_index, col_index)
				# logger.info("\tTF for document %s on current keyword is : %.8f" % (document.filename, tf))

				# now we make get the idf calculation
				idf = self._get_idf(keyword, row_index, col_index)
				# logger.info("\tIDF for this keyword is %.8f" % idf)

				# then combine them to get the TF-IDF weight
				tf_idf = float(tf * idf)
				# logger.info("\t\tFinal TF-IDF: %.8f" % tf_idf)

				# next, populate the new cell with the final valye
				self.tf_idf_matrix[row_index][col_index] = tf_idf



# preprocess the raw data
def do_preprocessing():

	#for file in files[0:6]:
	for file in files:

		P = Preprocessor(file)

		# and add that object to the processed objects list
		my_process_objects.append(P)

		# read the file
		P.read_file()

		# 2 - remove stopwords, lemmatize, and tokenize
		# https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
		P.filter_stopwords_lemmatize()

		# 3 - apply NER 
		# https://www.analyticsvidhya.com/blog/2021/06/nlp-application-named-entity-recognition-ner-in-python-with-spacy/#:~:text=Named%20Entity%20Recognition%20is%20the,%2C%20money%2C%20time%2C%20etc.
		P.apply_ner()

		# 4 - use sliding window approach to merge remaining phrases
		P.sliding_window_merge()

		# clean up my findings:
		# 	removes underscores from document text, lowercases
		#	removes underscores from frequency keys, lowercases
		P.cleanup()

		# 5 - at the end, write to out_file for each document for safety
		P.write_output()

		# also write the keywords concepts file
		write_keywords_concepts_file(P)

	return my_process_objects


def generate_document_term_matrix():

	M = DocuTermMatrix()

	# firt, consolidate and dedupe all keywords across the files
	M.consolidate_keywords_concepts()
	print(M.keywords_concepts)

	# second, create the matrix
	M.initialize_matrix()
	M.fill_matrix()

	print("\n~~~~ Moving on to TF-IDF section ~~~~\n")
	M.create_tf_idf()

	return M



def sort_topics(keywords_concepts, folder_aggregate_vector):
	# Overall processing 
	zipped_aggregate = list(zip(keywords_concepts, folder_aggregate_vector))

	# dedupe idk why there are duplicates
	zipped_aggregate = set(zipped_aggregate)

	# https://www.geeksforgeeks.org/python-ways-to-sort-a-zipped-list-by-values/
	# Using sorted and lambda
	sorted_topics = sorted(zipped_aggregate, key = lambda x: x[1], reverse=True)

	for x in sorted_topics:
		print(x)

	return sorted_topics


def write_topics_results(c1, c4, c7):

	logger.info("Appending to topics output file ...")
	topics_file = "./" + outfile_path + "topics.txt"
	
	with open(topics_file, "w") as f:

		# prepend string C1 to the C1 lines
		lines = c1
		for line in lines:
			line = str(line).strip("()")
			f.write("C1,")
			f.write(line)
			f.write("\n")

		# prepend string C4 to the C4 lines
		lines = c4
		for line in lines:
			line = str(line).strip("()")
			f.write("C4,")
			f.write(line)
			f.write("\n")

		# prepend string C7 to the C7 lines
		lines = c7
		for line in lines:
			line = str(line).strip("()")
			f.write("C7,")
			f.write(line)
			f.write("\n")


def generate_topics_per_folder(matrix_object):
	logger.info("generating the topics per folder ... ")

	# initialize the aggregate vectors of zeros
	c1_aggregate_vector = [float(0)] * len(matrix_object.keywords_concepts)
	c4_aggregate_vector = [float(0)] * len(matrix_object.keywords_concepts)
	c7_aggregate_vector = [float(0)] * len(matrix_object.keywords_concepts)

	# now go through the folders and combine the results per folder
	for i in range(len(my_process_objects)):

		if my_process_objects[i].file_basename.startswith("C1"):
			print("C1")
			for j in range(len(matrix_object.keywords_concepts)):

				# add the cell value to the value in the aggregate vector
				this_cell_value = matrix_object.tf_idf_matrix[i][j]
				# print(this_cell_value)

				c1_aggregate_vector[j] = float(c1_aggregate_vector[j] + this_cell_value)
				# logger.info("After adding, new value : %.8f" % c1_aggregate_vector[j])

		
		elif my_process_objects[i].file_basename.startswith("C4"):
			print("C4")
			for j in range(len(matrix_object.keywords_concepts)):

				# add the cell value to the value in the aggregate vector
				this_cell_value = matrix_object.tf_idf_matrix[i][j]
				# print(this_cell_value)

				c4_aggregate_vector[j] = float(c4_aggregate_vector[j] + this_cell_value)
				# logger.info("After adding, new value : %.8f" % c4_aggregate_vector[j])


		elif my_process_objects[i].file_basename.startswith("C7"):
			print("C7")
			for j in range(len(matrix_object.keywords_concepts)):

				# add the cell value to the value in the aggregate vector
				this_cell_value = matrix_object.tf_idf_matrix[i][j]
				# print(this_cell_value)

				c7_aggregate_vector[j] = float(c7_aggregate_vector[j] + this_cell_value)
				# logger.info("After adding, new value : %.8f" % c7_aggregate_vector[j])



	# now that we're done combining each folder's results to thier own vector
	# process the results
	c1_results = sort_topics(matrix_object.keywords_concepts, c1_aggregate_vector)
	c4_results = sort_topics(matrix_object.keywords_concepts, c4_aggregate_vector)
	c7_results = sort_topics(matrix_object.keywords_concepts, c7_aggregate_vector)

	# write the results to a file
	write_topics_results(c1_results, c4_results, c7_results)
	logger.info("Done writing the topics selection file.\n")



def exactly_equal(new_centroids, old_centroids):

	if new_centroids[0] == old_centroids[0]:
		if new_centroids[1] == old_centroids[1]:
			if new_centroids[2] == old_centroids[2]:
				return True

	return False


def average(seq):

	if len(seq) == 0:
		print("PROBLEM\n\n")
		print(seq)

	return float(sum(seq) / len(seq))

def dot_prod(vector_1, vector_2):
	return float(sum(x * y for x, y in zip(vector_1, vector_2)))

def mag(vec):
	return float(sqrt(dot_prod(vec, vec)))

def cosine_similarity(vector_1, vector_2):
	# define the cosine similarity between the two vectors
	cosine_similarity = dot_prod(vector_1,vector_2) / ( mag(vector_1) * mag(vector_2) + 0.000000001  ) 

	# print("cosine similarity: %s" % cosine_similarity)
	return cosine_similarity


def euclidean_distance(vector_1, vector_2):
	# define the euclidean distance
	# https://machinelearningmastery.com/distance-measures-for-machine-learning/
	euclidean_distance = sqrt(sum((e1-e2)**2 for e1, e2 in zip(vector_1,vector_2)))

	# print("euclidean_distance: %s" % euclidean_distance)
	return euclidean_distance



class KMeans():
	def __init__(self, matrix_object):
		self.matrix_object = matrix_object
		self.documents = matrix_object.tf_idf_matrix					# a list of vectors
		self.number_of_clusters = 3
		self.centroids = [[],[],[]]
		self.clusters = [[],[],[]]
		self.clusters_indices = [[],[],[]]

	# ensure it doesnt randomly pick the same centroid twice :(
	def initialize_centroids(self):

		new_centroids_str = []
		new_centroids = []

		while len(new_centroids) < 3:

			starting_centroid = random.choice(self.documents)

			hash_value = hash(str(starting_centroid))
			if hash_value not in new_centroids_str:

				new_centroids_str.append(hash_value)
				new_centroids.append(starting_centroid)

		self.centroids = new_centroids


	def assign_clusters(self):
		# for each of the documents 
		# measure the distance to each of the centroids
		# pick the relevant centroid index:
		#			euclidean distance -> smallest number
		#			cosine similarity -> highest number

		self.clusters = [[],[],[]]
		self.clusters_indices = [[],[],[]]

		# for each document find the distance to all three centroids
		for document_index in range(len(self.documents)):


			document_vector = self.documents[document_index]
			similarity_score = 0
			euclidean_similarity_score = 90000000000
			cluster_index = 1

			for centroid_index in range(len(self.centroids)):


				### COSINE

				# get the cosine similarity
				cosine_sim_score = cosine_similarity(document_vector, self.centroids[centroid_index])
				# logger.info("distance between document index: %s and centroid index: %s :: %s" % (document_index, centroid_index, cosine_sim_score))

				# try to maximize this number for cosine similarity
				# a cosine similarity closer to 1 means angle is small between the vectors
				# a cosine similarity closer to 0 means they are practically orthogonal vectors (very different from each other)
				if cosine_sim_score > similarity_score:
					similarity_score = cosine_sim_score
					cluster_index = centroid_index



				### EUCLIDEAN

				# euclidean distance minimization
				# euclidean_distance_score = euclidean_distance(document_vector, self.centroids[centroid_index])
				# logger.info("distance between document index: %s and centroid index: %s :: %s" % (document_index, centroid_index, euclidean_distance_score))

				# try to minimize it fr euclidean
				# update the similarity score baseline
				# save the centroid index			
				# if euclidean_distance_score < euclidean_similarity_score:
				# 	euclidean_similarity_score = euclidean_distance_score
				# 	cluster_index = centroid_index



			logger.debug("in the end, document index %s is most similar to the %s'th centroid." % (document_index, cluster_index))

			# and add the index of that document to the corresponding index in the clusters
			self.clusters_indices[cluster_index].append(document_index)
			self.clusters[cluster_index].append(self.documents[document_index])

			for cl in self.clusters_indices:
				logger.debug(cl)


	def update_centroids(self):

		# take the average of the points in each cluster group
		new_centroids = []

		for cluster in self.clusters:


			num_keywords_concepts = len(self.matrix_object.keywords_concepts)
			# initialize a new centroid for each cluster
			new_centroid = [0] * num_keywords_concepts

			if not cluster:
				logger.warning("problem! ~ this cluster is empty\n")


			for item in range(num_keywords_concepts):

				the_column_values = []
				# print("col:%s" % item)

				for row in cluster:
					value = row[item]

					#print("cell value: %s" % value)
					the_column_values.append(value)


				# once we've gone through all the row-values, 
				# take the average and append it to new new_centroid
				avg_value = average(the_column_values)
				#print("the average of column %s is %s" % (item, avg_value))

				new_centroid[item] = avg_value

			new_centroids.append(new_centroid)


		# then compare these centroids to the ones we had previously
		if exactly_equal(new_centroids, self.centroids):
			logger.debug("You have met the stopping criterion. Returning to caller ... ")
			return False

		else:
			# if they have moved, continue to iterate k-means
			self.centroids = new_centroids
			return True


def get_similarities(matrix_object):
	logger.info("Starting with K-Means and k=3 ...")
	
	# step 1, randomly pick (k=3) data points (a vector) as our initial centroids (centroid vector)
	K = KMeans(matrix_object)

	K.initialize_centroids()

	logger.debug("centroids:")
	logger.debug(K.centroids) 	# a list of 3 vectors

	counter = 1
	# main iteration for k-means
	K.assign_clusters()
	while K.update_centroids():
		K.assign_clusters()
		
		# just in case . dont go crazy
		counter += 1
		if counter > 10:
			break

	logger.debug("you finished and converged to some centroid values. wohoo!")
	logger.info("k-means took %s iterations" % counter)

	# We have the following indices in each cluster:
	for c in K.clusters:
		logger.info("a cluster has this many nodes:")
		logger.info(len(c))


	# logger.warning("checking for sanity:")
	# logger.warning(len(matrix_object.tf_idf_matrix))
	# logger.warning(len(K.documents))


	return K


class Visualize():

	def __init__(self, kmeans_object):
		self.features = kmeans_object.matrix_object.keywords_concepts
		self.tf_idf_matrix = kmeans_object.matrix_object.tf_idf_matrix
		self.predicted_clusters = kmeans_object.clusters_indices
		self.actual_clusters = [[],[],[]]


	def get_actual_clusters(self):
		self.actual_clusters[0] = [0,1,2,3,4,5,6,7]
		self.actual_clusters[1] = [8,9,10,11,12,13,14,15]
		self.actual_clusters[2] = [16,17,18,19,20,21,22,23]
		self.label_dict = {}

		logger.debug("\nactual clusters:")
		for cluster in self.actual_clusters:
			logger.debug("\t %s" % cluster)


	def label_predicted_clusters(self):

		# get the average valyes and rank them, call them lowest middle high and c1,c2,c3
		self.label_dict = {

			0: 'police force air china',
			1: 'mouth disease & healthcare',
			2: 'mortgages & banks'

		}

	def plot_actual(self):

		logger.info("plotting actual ...")
		clusters = self.actual_clusters


		A = np.array(self.tf_idf_matrix)
		svd =  TruncatedSVD(n_components = 2)
		A_transf = svd.fit_transform(A)

		logger.info("Transformed Matrix after reducing to 2 features:")
		print(A_transf)


		labels = files 
		x_vals = A_transf[:,0]
		y_vals = A_transf[:,1]

		fig = plt.figure()
		ax = fig.add_subplot()

		clusters = self.predicted_clusters

		for i in range(len(files)):

			if i in clusters[0]:
				ax.scatter(x_vals[i], y_vals[i], color="teal", label="police force & air china", alpha=0.3)

			elif i in clusters[1]:
				ax.scatter(x_vals[i], y_vals[i], color="magenta", label="mouth disease & healthcare", alpha=0.3)

			elif i in clusters[2]:
				ax.scatter(x_vals[i], y_vals[i], color="orange", label="mortgages & banks", alpha=0.3)

		plt.xlim([-0.0005, 0.005])
		plt.ylim([-0.0005, 0.005])


		# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html
		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		plt.legend(by_label.values(), by_label.keys())

		ax.grid(True)
		plt.title("Actual Clusters")
		plt.show()


	def plot_predicted(self):

		logger.info("plotting predicted cluster  ...")

		A = np.array(self.tf_idf_matrix)

		svd =  TruncatedSVD(n_components = 2)
		A_transf = svd.fit_transform(A)

		logger.debug("Transformed Matrix after reducing to 2 features:")
		logger.debug(A_transf)


		labels = files 
		x_vals = A_transf[:,0]
		y_vals = A_transf[:,1]

		fig = plt.figure()
		ax = fig.add_subplot()

		clusters = self.predicted_clusters

		for i in range(len(files)):

			if i in clusters[0]:
				ax.scatter(x_vals[i], y_vals[i], color="teal", label="police force & air china", alpha=0.3)

			elif i in clusters[1]:
				ax.scatter(x_vals[i], y_vals[i], color="magenta", label="mouth disease & healthcare", alpha=0.3)

			elif i in clusters[2]:
				ax.scatter(x_vals[i], y_vals[i], color="orange", label="mortgages & banks", alpha=0.3)

		plt.xlim([-0.0005, 0.005])
		plt.ylim([-0.0005, 0.005])


		# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html
		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		plt.legend(by_label.values(), by_label.keys())

		ax.grid(True)
		plt.title("Predicted Clusters")
		plt.show()



class ConfusionMatrix():

	def __init__(self, actual_clusters, predicted_clusters):
		self.actual_clusters = actual_clusters
		self.predicted_clusters = predicted_clusters
		self.actual_vector = []
		self.predicted_vector = []
		self.label_dict = {}
		self.precision = None
		self.recall = None
		self.f1_score = None

	def _generate_actual_vector(self):

		actual_vector = []
		for i in range(len(self.actual_clusters)):

			for j in self.actual_clusters[i]:
				actual_vector.append(i)

		logger.debug("\nactual::")
		logger.debug(actual_vector)

		self.actual_vector = actual_vector

	def _generate_predicted_vector(self):

		predicted_vector = []
		for i in range(len(self.predicted_clusters)):

			for j in self.predicted_clusters[i]:
				predicted_vector.append(i)

		logger.debug("\npredicted ::")
		logger.debug(predicted_vector)

		self.predicted_vector = predicted_vector

	

	def generate_confision_matrix(self):
		logger.info("Generating the confusion matrix now ...")
		
		self._generate_actual_vector()
		self._generate_predicted_vector()

		self.confusion_matrix = confusion_matrix(self.actual_vector, self.predicted_vector)

		logger.info("Confusion Matrix:")
		print(self.confusion_matrix)

		'''
			(predicted labels)
				0 	1 	2

				6 	2	0		2
				0	7	1		1 	(actual labels)
				0	0	8		0

		'''


	def consider_precision_recall(self):
		logger.info("precision and recall scores:")

		self.label_dict = {

			0: 'police force air china',
			1: 'mouth disease & healthcare',
			2: 'mortgages & banks'

		}
		labels = [v for v in self.label_dict.values()]

		# https://stackoverflow.com/questions/40729875/calculate-precision-and-recall-in-a-confusion-matrix
		conf = np.array(self.confusion_matrix)
		# will be same as false false for now
		true_positive = np.diag(conf)		# along the diagonal of the matrix
		false_positive = np.sum(conf, axis=0) - true_positive
		false_negative = np.sum(conf, axis=1) - true_positive

		self.precision = np.average(true_positive / (true_positive + false_positive) )
		self.recall = np.average(true_positive / (true_positive + false_negative) )

		logger.info("Precision: %s" % self.precision)
		logger.info("Recall: %s" % self.recall)


	def consider_F1_score(self):
		logger.info("calculating F1 score ...")

		# f1= 2 * [ (precision* recall ) / (precision + recall) ]
		self.f1_score = 2 * ( (self.precision * self.recall) / (self.precision + self.recall) ) 

		logger.info("F1 Score: %s" % self.f1_score)




if __name__ == '__main__':

	logger.info("starting ...");

	#
	# PART 1
	#

	# does preprocessing on the files
	# returns a list of preprocessed file objects for each file
	logger.info("First: Do preprocessing on the files")
	processed_objects = do_preprocessing()

	# then give it the matrix class here
	logger.info("Next: Generating Document Term Matrix")
	matrix_object = generate_document_term_matrix()

	# then gather the list of topics per folder
	logger.info("Then: Consolidate and identify topics of each folder")
	generate_topics_per_folder(matrix_object)

	# from looking at the generated topics file, I think the topics are:
	topics_per_folder = [
		'police force & air china',
		'mouth diseases & vaccines',
		'mortgages & banks'
	]


	#
	# PART 2
	#

	# clustering textual data
	k_means = get_similarities(matrix_object)

	# plot the original dataset
	# and plot my clusters

	V = Visualize(k_means)
	V.get_actual_clusters()
	V.label_predicted_clusters()

	# take a look at the results
	logger.warning("will render a single plot at a time !")
	logger.warning("save and close the current plot to continue the program !")
	V.plot_predicted()
	V.plot_actual()


	# generate_confusion_matrix(matrix_object, k_means)
	logger.info("Generating confusion matrix...")
	C = ConfusionMatrix(V.actual_clusters,V.predicted_clusters)
	C.generate_confision_matrix()
	C.consider_precision_recall()
	C.consider_F1_score()

















































