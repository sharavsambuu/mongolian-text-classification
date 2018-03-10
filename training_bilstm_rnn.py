import tensorflow as tf
import numpy as np
from training_helpers import *

batch_size     = 24
lstm_units     = 128
num_classes    = 6
max_seq_length = 500
vector_length  = 100    # word2vec dimensions
iterations     = 100000 # 100000

dataset = DataSetHelper() 

tf.reset_default_graph()

label_placeholder = tf.placeholder(tf.float32, [batch_size, num_classes   ], name='label_placeholder')
input_placeholder = tf.placeholder(tf.int32  , [batch_size, max_seq_length], name='input_placeholder')

ids_matrix    = np.load('ids_matrix.npy')
embeddings_tf = tf.constant(ids_matrix)

batch_data = tf.Variable(tf.zeros([batch_size, max_seq_length, vector_length]), dtype=tf.float32)
batch_data = tf.nn.embedding_lookup(embeddings_tf, input_placeholder)
batch_data = tf.cast(batch_data, tf.float32) # https://github.com/tensorflow/tensorflow/issues/8281

# composing bidirectional lstm
batch_unstack = tf.unstack(batch_data, max_seq_length, 1)
fw_lstm_cell  = tf.contrib.rnn.BasicLSTMCell(lstm_units) # forward lstm cell
fw_lstm_cell  = tf.contrib.rnn.DropoutWrapper(cell=fw_lstm_cell, output_keep_prob=0.75)
bw_lstm_cell  = tf.contrib.rnn.BasicLSTMCell(lstm_units) # backward lstm cell
bw_lstm_cell  = tf.contrib.rnn.DropoutWrapper(cell=bw_lstm_cell, output_keep_prob=0.75)
outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
    fw_lstm_cell ,
    bw_lstm_cell ,
    batch_unstack,
    dtype=tf.float32
)

weight     = tf.Variable(tf.truncated_normal([2*lstm_units, num_classes]))
bias       = tf.Variable(tf.constant(0.1, shape=[num_classes]))
prediction = tf.matmul(outputs[-1], weight) + bias

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label_placeholder, 1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=label_placeholder))
optimizer = tf.train.AdamOptimizer().minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

import datetime
print("starting at ", datetime.datetime.now())
tf.summary.scalar('Loss '    , loss    )
tf.summary.scalar('Accuracy ', accuracy)
merged  = tf.summary.merge_all()
log_dir = "tensorboard/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"

init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer  = tf.summary.FileWriter(log_dir, sess.graph)
    saver   = tf.train.Saver()
    sess.run(init)

    for i in range(iterations):
        next_input_batch, next_label_batch = dataset.get_training_batch(batch_size, max_seq_length)
        sess.run(optimizer, feed_dict={input_placeholder: next_input_batch, label_placeholder: next_label_batch})
        if (i%50 == 0):
            acc = sess.run(accuracy, feed_dict={input_placeholder: next_input_batch, label_placeholder: next_label_batch})
            los = sess.run(loss    , feed_dict={input_placeholder: next_input_batch, label_placeholder: next_label_batch})
            print("___________________________________")
            print("Iteration : ", i  )
            print("Accuracy  : ", acc)
            print("Loss      : ", los)
            summary = sess.run(merged, {input_placeholder: next_input_batch, label_placeholder: next_label_batch})
            writer.add_summary(summary, i)
        if (i%1000 == 0 and i != 0):
            save_path = saver.save(sess, "models/bilstm/pretrained_bilstm.ckpt", global_step=i)
            print("model is saved to %s"%save_path)
    writer.close()

