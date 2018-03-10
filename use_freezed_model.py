import tensorflow as tf
from training_helpers import *
from itertools import chain
from clear_text_to_array import *

frozen_graph = './models/bilstm/pretrained_bilstm-3000.pb'

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
ids = dataset_helper.sentence_to_ids(content, max_seq_length)

print("converting news words to ids")
print("world news corpus")
print(content)
print("words ids")
print(ids)

sess = tf.Session(graph=graph)
sess.close()