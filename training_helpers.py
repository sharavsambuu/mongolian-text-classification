from random import randint
import random
from clear_text_to_array import *
from gensim.models import Word2Vec
import glob, json, re
import json
import numpy as np
from wordtoken_to_id import *
from clear_text_to_array import *

class DataSetHelper():
    def __init__(self,):
        self.word2vec        = Word2Vec.load('model.bin')
        self.ids_matrix      = np.load('ids_matrix.npy')
        self.unknown_word_id = wordtoken_to_id(self.word2vec, "анноунүг")
        with open("temp_corpuses/dataset.json", "r") as f:
            self.dataset_json = json.load(f)
            self.training_set = self.dataset_json['training']
            self.testing_set  = self.dataset_json['testing' ]
        pass

    def sentence_to_ids(self, sentence, max_seq_length):
        #print("========SENTENCE=======")
        #print(sentence)
        sentence_array  = clear_text_to_array(sentence)[0]
        #print("========SENTENCEARR====")
        #print(sentence_array)
        sentence_array  = sentence_array[:max_seq_length]
        ids_of_sentence = np.zeros((max_seq_length), dtype='int32')
        for index, word in enumerate(sentence_array):
            try:
                ids_of_sentence[index] = wordtoken_to_id(self.word2vec, word)
            except KeyError:
                ids_of_sentence[index] = self.unknown_word_id # unknown word, АННОУНҮГ 
        return ids_of_sentence

    def get_training_batch(self, batch_size, max_seq_length):
        batch_labels = []
        batch_arr = np.zeros([batch_size, max_seq_length])
        for i in range(batch_size):
            random_corpus = random.choice(self.training_set)
            file_name     = random_corpus[0]
            one_hot       = random_corpus[1]
            category      = random_corpus[2]
            file_path     = "corpuses/"+category+"/"+file_name
            with open(file_path) as f:
                sentence = json.load(f)['body']
                #print("##########################")
                #print(file_path)
                #print(sentence)
                ids_of_sentence = self.sentence_to_ids(sentence, max_seq_length)
                batch_arr[i] = ids_of_sentence
            batch_labels.append(one_hot)
        return (batch_arr, batch_labels)

    def get_testing_batch(self, batch_size, max_seq_length):
        batch_labels = []
        batch_arr = np.zeros([batch_size, max_seq_length])
        for i in range(batch_size):
            random_corpus = random.choice(self.testing_set)
            file_name     = random_corpus[0]
            one_hot       = random_corpus[1]
            category      = random_corpus[2]
            file_path     = "corpuses/"+category+"/"+file_name
            with open(file_path) as f:
                sentence = json.load(f)['body']
                ids_of_sentence = self.sentence_to_ids(sentence, max_seq_length)
                batch_arr[i] = ids_of_sentence
            batch_labels.append(one_hot)
        return (batch_arr, batch_labels)

#batch_size, seq_length = 50, 500
#dataset = DataSetHelper()
#for i in range(10000):
#    inp, label = dataset.get_training_batch(batch_size, seq_length)
#import pdb; pdb.set_trace()