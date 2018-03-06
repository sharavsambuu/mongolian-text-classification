from clear_text_to_array import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from gensim.models import Word2Vec

tfe.enable_eager_execution()

word2vec   = Word2Vec.load('model.bin')
ids_matrix = np.load('ids_matrix.npy')

input_sentence = "хоёр өдрийн уулзалтын үр дүнд дээд хэмжээний илчээ илгээсэн юм."

sentence_array = clear_text_to_array(input_sentence)[0]
print("Өгүүлбэр ")
print("---------------------")
print(sentence_array)
print("---------------------")
first_word  = sentence_array[0]
second_word = sentence_array[1]
last_word   = sentence_array[-1]
first_index  = word2vec.wv.vocab[first_word ].index
second_index = word2vec.wv.vocab[second_word].index
last_index   = word2vec.wv.vocab[last_word  ].index
print("эхний үг      : ", first_word , ", index : ", first_index )
print("хоёрдугаар үг : ", second_word, ", index : ", second_index)
print("сүүлийн үг    : ", last_word  , ", index : ", last_index  )
#el = word2vec[sentence_array[0]]
#print(el)

print("---------------------")



x = [[2.]]
m = tf.matmul(x, x)
print(m)