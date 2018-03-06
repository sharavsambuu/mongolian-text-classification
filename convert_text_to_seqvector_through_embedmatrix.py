import numpy as np
import tensorflow as tf
from clear_text_to_array import *

ids = np.load('ids_matrix.npy')

input_sentence = "хоёр өдрийн уулзалтын үр дүнд дээд хэмжээний илчээ илгээсэн юм."

sentence_array = clear_text_to_array(input_sentence)[0]
print(sentence_array)