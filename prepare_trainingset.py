from os import listdir
from os.path import isfile, join
import shutil
import glob, os, os.path
import random
import math
import json

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from gensim.models import Word2Vec

from wordtoken_to_id import *
from clear_text_to_array import *

tfe.enable_eager_execution()

max_sequence_length = 500
wordvector_length   = 100

exp_sentence    = "Хориод жилийн өмнө чөлөөт зах зээлийг зорин хөдөлсөн монголын эдийн засаг гэх ачааны машин, агаарын бохирдолтой уралдан өтгөрч буй авлигын мананд төөрөн явсаар эдүгээ шаварт суучихаад, хөдөлгүүрээ хичнээн янгинуулан зүтгэсэн ч шавраасаа гарах байтугай, улам доош суун шигдсээр байна. "
sentence_array  = clear_text_to_array(exp_sentence)[0]
sentence_array  = sentence_array[:max_sequence_length]

word2vec        = Word2Vec.load('model.bin')
ids_matrix      = np.load('ids_matrix.npy')

#word2vec.wv.index2word([1])

ids_of_sentence = np.zeros((max_sequence_length), dtype='int32')
for index, word in enumerate(sentence_array):
    try:
        ids_of_sentence[index] = wordtoken_to_id(word2vec, word)
    except KeyError as e:
        ids_of_sentence[index] = 102153 # unknown word, АННОУНҮГ 
        print("exception at index ", index, " word at ", word)
        pass

#print(ids_of_sentence)
#print(ids_of_sentence.shape)

embeddings_tf    = tf.constant(ids_matrix)
ids_tf           = tf.constant(ids_of_sentence)
sequence_vectors = tf.nn.embedding_lookup(embeddings_tf, ids_tf)

#print(sequence_vectors)

global_corpuses = []

for filename in glob.iglob('corpuses/**/*.txt', recursive=True):
    current_file_path      = os.path.abspath(filename)
    current_directory      = os.path.abspath(os.path.join(current_file_path, os.pardir))
    current_directory_name = os.path.split(current_directory)
    category               = current_directory_name[1]
    only_file_name         = os.path.basename(filename)
    global_corpuses.append((category, only_file_name))

random.shuffle(global_corpuses)
random.shuffle(global_corpuses)
random.shuffle(global_corpuses)

split_location = math.floor(80*len(global_corpuses)/100) # 80% for training, 20% for testing
training_set   = global_corpuses[:split_location]
test_set       = global_corpuses[split_location:]
dataset_info   = {
    'training' : training_set,
    'testing'  : test_set
}

temp_corpus_dir = 'temp_corpuses'
if os.path.exists(temp_corpus_dir):
    shutil.rmtree(temp_corpus_dir)
os.makedirs(temp_corpus_dir)

with open("temp_corpuses/dataset.json", "w", encoding="utf8") as outfile:
    json.dump(dataset_info, outfile, ensure_ascii=False)

#import pdb; pdb.set_trace()