import tensorflow as tf
import numpy as np
from training_helpers import *
from itertools import chain
from clear_text_to_array import *
from xmlrpc.server import SimpleXMLRPCServer # for django app
import sys # handle interrupt

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


def get_class_name(x):
    switcher = {
        0: lambda: "economy"   ,
        1: lambda: "health"    ,
        2: lambda: "politics"  ,
        3: lambda: "society"   ,
        4: lambda: "technology",
        5: lambda: "world"      
    }
    return switcher.get(x, lambda: "UNKNOWN")()

sess = tf.Session(graph=graph)

def predict_class(sess, filename):
    with open(filename, 'r') as content_file:
        content = content_file.read()

    max_seq_length = 500
    num_classes = 6

    word_ids = dataset_helper.sentence_to_ids(content, max_seq_length)
    x_batch = []
    for i in range(24):
        x_batch.append(word_ids)
    x_batch = np.array(x_batch)
    results = []

    results_tf = sess.run(prediction_op, feed_dict={input_placeholder: x_batch})
    for i in results_tf:
        softmax_result = softmax(i)
        argmax = softmax_result.argmax(axis=0)
        name = get_class_name(argmax)
        results.append([softmax_result, argmax, name])

    print("result: ", results[0][2])
    print(results[0][0])

print('----------------------------')
print("trying to predict world news")
predict_class(sess, "./corpuses_test/world_news_gogo_mn.txt")

print('----------------------------')
print("trying to predict economy news")
predict_class(sess, "./corpuses_test/economy_news_gogo_mn.txt")

print('----------------------------')
print("trying to predict technology news")
predict_class(sess, "./corpuses_test/technology_news_gogo_mn.txt")

print('----------------------------')
print("trying to predict health news")
predict_class(sess, "./corpuses_test/health_news_gogo_mn.txt")


print('----------------------------')
print("trying to predict political news")
predict_class(sess, "./corpuses_test/politics_news_ikon_mn.txt")

def predict_class_from_text(content):
    max_seq_length = 500

    word_ids = dataset_helper.sentence_to_ids(content, max_seq_length)
    x_batch = []
    for i in range(24):
        x_batch.append(word_ids)
    x_batch = np.array(x_batch)
    results = []

    results_tf = sess.run(prediction_op, feed_dict={input_placeholder: x_batch})
    for i in results_tf:
        softmax_result = softmax(i)
        argmax = softmax_result.argmax(axis=0)
        name = get_class_name(argmax)
        results.append([softmax_result, argmax, name])

    return str(results[0][2])

try:
    rpc_server = SimpleXMLRPCServer(("localhost", 50001))
    print("----------------------------")
    print("classifier RPC server is listening on port 50001...")
    rpc_server.register_function(predict_class_from_text, "predict_class_from_text")
    rpc_server.serve_forever()
except KeyboardInterrupt:
    sess.close()
    sys.exit()

sess.close()
