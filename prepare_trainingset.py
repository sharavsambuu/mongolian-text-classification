from os import listdir
from os.path import isfile, join
import tensorflow as tf
import numpy as np
from wordtoken_to_id import *
from clear_text_to_array import *
from gensim.models import Word2Vec
import tensorflow.contrib.eager as tfe

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

print(ids_of_sentence)
print(ids_of_sentence.shape)

embeddings_tf    = tf.constant(ids_matrix)
ids_tf           = tf.constant(ids_of_sentence)
sequence_vectors = tf.nn.embedding_lookup(embeddings_tf, ids_tf)

print(sequence_vectors)
import pdb; pdb.set_trace()

