import tensorflow as tf
import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="name of nn architecture")
parser.add_argument("--iteration", help="iteration value")
args = parser.parse_args()

if args.name is None or args.iteration is None:
    print("please provide parameters, more info")
    print("python freeze_tf_model.py -h")
    sys.exit()

name      = args.name
iteration = args.iteration

saver = tf.train.import_meta_graph('./models/{name}/pretrained_{name}.ckpt-{iteration}.meta'.format(name=name, iteration=iteration), clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./models/{name}/pretrained_{name}.ckpt-{iteration}".format(name=name, iteration=iteration))

# output variable name
output_node_names = "input_placeholder,prediction_op"
output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, input_graph_def, output_node_names.split(",")
)

output_graph = "./models/{name}/pretrained_{name}-{iteration}.pb".format(name=name, iteration=iteration)
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
    print("saved to ", output_graph)

sess.close()