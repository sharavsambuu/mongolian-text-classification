from clear_text_to_array import *
from wordtoken_to_id import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from gensim.models import Word2Vec

tfe.enable_eager_execution()

word2vec   = Word2Vec.load('model.bin')
ids_matrix = np.load('ids_matrix.npy')

input_sentence = "хоёр өдрийн уулзалтын үр дүнд дээд хэмжээний элчээ илгээсэн юм."

sentence_array = clear_text_to_array(input_sentence)[0]

print("---------------------")
first_word   = sentence_array[0]
second_word  = sentence_array[1]
last_word    = sentence_array[-1]
first_index  = word2vec.wv.vocab[first_word ].index
second_index = word2vec.wv.vocab[second_word].index
last_index   = word2vec.wv.vocab[last_word  ].index
print("эхний үг      : ", first_word , ", index : ", first_index )
print("хоёрдугаар үг : ", second_word, ", index : ", second_index)
print("сүүлийн үг    : ", last_word  , ", index : ", last_index  )
print("нийт үгсийн тоо : ", len(word2vec.wv.vocab))
print("---------------------")
#print(word2vec.wv[last_word])
if (np.array_equal(ids_matrix[last_index], word2vec.wv[last_word])):
    print("YES, conversion to id sequence can be implemented through gensim word2vec object.")
else:
    print("NO")

print("---------------------")
print("Өгүүлбэр ")
print(sentence_array)
print("---------------------")

# converting token sequence into sequence of ids
sentence_in_tokenids = []
for token in sentence_array:
    token_id = wordtoken_to_id(word2vec, token)
    sentence_in_tokenids.append(token_id)
print("id нуудын жагсаалт")
print(sentence_in_tokenids)

# trying to convert sequence of vectors through tensorflow embedding lookup stuff.
embeddings        = tf.constant(ids_matrix)
ids               = tf.constant(sentence_in_tokenids)
sequence_vectors = tf.nn.embedding_lookup(embeddings, ids)
print("үгэн векторуудын жагсаалт тензор хэлбэрээр:")
print(sequence_vectors)