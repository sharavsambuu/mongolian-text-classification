import tensorflow as tf

saver = tf.train.import_meta_graph('./models/pretrained_lstm.ckpt-10000.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./models/pretrained_lstm.ckpt-10000")

# output variable name
output_node_names = "prediction:0"
output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, input_graph_def, output_node_names.split(",")
)

output_graph = "./models/pretrained_lstm.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()