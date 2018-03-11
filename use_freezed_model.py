import tensorflow as tf
import numpy as np
from training_helpers import *
from itertools import chain
from clear_text_to_array import *

def softmax(x):
    score_math_exp = np.exp(np.asarray(x))
    return score_math_exp / score_math_exp.sum(0)

frozen_graph = './models/bilstm/pretrained_bilstm-24000.pb'

with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    input_placeholder, prediction = tf.import_graph_def(
        restored_graph_def,
        input_map       = None,
        return_elements = ['input_placeholder', 'prediction_op'],
        name            = ''
    )

input_placeholder = graph.get_tensor_by_name("input_placeholder:0")
prediction_op     = graph.get_tensor_by_name("prediction_op:0")

dataset_helper = DataSetHelper()

with open('./corpuses_test/world_news_gogo_mn.txt', 'r') as content_file:
    content = content_file.read()

max_seq_length = 500
num_classes    = 6

word_ids = dataset_helper.sentence_to_ids(content, max_seq_length)

x_batch = []
for i in range(24):
    x_batch.append(word_ids)

x_batch = np.array(x_batch)

print("----- X BATCH ----")
print(x_batch)
results = []

sess = tf.Session(graph=graph)
results_tf = sess.run(prediction_op, feed_dict={input_placeholder: x_batch})
for i in results_tf:
    softmax_result = softmax(i)
    results.append(softmax_result)

print("----- RESULT -----")
print(results)
sess.close()
