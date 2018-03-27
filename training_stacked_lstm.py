import datetime
import tensorflow as tf
import numpy as np
from training_helpers import *

batch_size     = 24
lstm_units     = 128
num_classes    = 6
max_seq_length = 500
vector_length  = 100    # word2vec dimensions
iterations     = 100000 # 100000
stack_count    = 2

dataset = DataSetHelper() 

tf.reset_default_graph()

def shape_detective(sess, tensor, explainer=""):
    print("-------------------------")
    print(explainer, sess.run(tf.shape(tensor)))

input_placeholder = tf.placeholder(tf.int32  , [batch_size, max_seq_length], name='input_placeholder')
label_placeholder = tf.placeholder(tf.float32, [batch_size, num_classes   ])

ids_matrix    = np.load('ids_matrix.npy')
embeddings_tf = tf.constant(ids_matrix)
batch_data    = tf.Variable(tf.zeros([batch_size, max_seq_length, vector_length]), dtype=tf.float32)
batch_data    = tf.nn.embedding_lookup(embeddings_tf, input_placeholder)
batch_data    = tf.cast(batch_data, tf.float32) # https://github.com/tensorflow/tensorflow/issues/8281
batch_unstack = tf.unstack(batch_data, max_seq_length, 1)

'''
def get_lstm(lstm_units):
    lstm_cell  = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    lstm_cell  = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
    return lstm_cell
stacked_rnn = tf.contrib.rnn.MultiRNNCell([
    tf.contrib.rnn.BasicLSTMCell(lstm_units),
    tf.contrib.rnn.BasicLSTMCell(lstm_units)
])
stacked_rnn = tf.contrib.rnn.MultiRNNCell([get_lstm(lstm_units)]*stack_count)
stacked_rnn = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_units)]*stack_count)
'''

stacked_rnn = tf.contrib.rnn.MultiRNNCell([
    tf.contrib.rnn.BasicLSTMCell(lstm_units),
    tf.contrib.rnn.BasicLSTMCell(lstm_units)
])

value_before_transpose, _ = tf.nn.dynamic_rnn(stacked_rnn, batch_data, dtype=tf.float32)

weight                = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
bias                  = tf.Variable(tf.constant(0.1, shape=[num_classes]))
value_after_transpose = tf.transpose(value_before_transpose, [1, 0, 2])
last                  = tf.gather(value_after_transpose, int(value_after_transpose.get_shape()[0]) - 1)
prediction            = tf.add(tf.matmul(last, weight), bias, name='prediction_op')
#prediction = tf.add(tf.matmul(value[-1], weight), bias, name='prediction_op')

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label_placeholder, 1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=label_placeholder))
optimizer = tf.train.AdamOptimizer().minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

print("started at ", datetime.datetime.now())

loss_summary                = tf.summary.scalar('Loss'                     , loss    )
validation_accuracy_summary = tf.summary.scalar('Batch Validation Accuracy', accuracy)
testing_accuracy_summary    = tf.summary.scalar('Testing Dataset Accuracy' , accuracy)

log_dir = "tensorboard/stackedlstm/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"

init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer  = tf.summary.FileWriter(log_dir, sess.graph)
    saver   = tf.train.Saver()
    sess.run(init)
    print("detecting shape flow and changes...")

    for i in range(iterations):
        next_input_batch, next_label_batch = dataset.get_training_batch(batch_size, max_seq_length)
        test_input_batch, test_label_batch = dataset.get_testing_batch (batch_size, max_seq_length)
        sess.run(optimizer, feed_dict={input_placeholder: next_input_batch, label_placeholder: next_label_batch})
        if (i%10 == 0):
            acc = sess.run(accuracy, feed_dict={input_placeholder: next_input_batch, label_placeholder: next_label_batch})
            los = sess.run(loss    , feed_dict={input_placeholder: next_input_batch, label_placeholder: next_label_batch})
            tes = sess.run(accuracy, feed_dict={input_placeholder: test_input_batch, label_placeholder: test_label_batch})
            print("__________________________________")
            print("Iteration  : ", i  )
            print("Validation : ", acc)
            print("Loss       : ", los)
            print("Test acc   : ", tes)
            validation_accuracy_result = sess.run(validation_accuracy_summary, feed_dict={input_placeholder: next_input_batch, label_placeholder: next_label_batch})            
            testing_accuracy_result    = sess.run(testing_accuracy_summary   , feed_dict={input_placeholder: test_input_batch, label_placeholder: test_label_batch})
            loss_result                = sess.run(loss_summary               , feed_dict={input_placeholder: next_input_batch, label_placeholder: next_label_batch})
            writer.add_summary(validation_accuracy_result, i)
            writer.add_summary(testing_accuracy_result   , i)
            writer.add_summary(loss_result               , i)
        if (i%1000 == 0 and i != 0):
            save_path = saver.save(sess, "models/stackedlstm/pretrained_lstm.ckpt", global_step=i)
            print("model is saved to %s"%save_path)

        print("__________________________________")
        shape_detective(sess, input_placeholder     , explainer="input_placeholder :")
        shape_detective(sess, label_placeholder     , explainer="label_placeholder :")
        shape_detective(sess, embeddings_tf         , explainer="embeddings :")
        shape_detective(sess, batch_data            , explainer="batch_data before unstacking :")
        shape_detective(sess, batch_unstack         , explainer="batch_unstack after unstacking batch_data :")
        shape_detective(sess, weight                , explainer="weight :")
        shape_detective(sess, bias                  , explainer="bias :")
        shape_detective(sess, value_before_transpose, explainer="value shape before transpose stacked 2 lstms :")
        shape_detective(sess, value_after_transpose , explainer="value shape after_transpose :")
        shape_detective(sess, last                  , explainer="shape after gather transposed value :")
        shape_detective(sess, prediction            , explainer="dense connection, prediction shape :")




    writer.close()

print("ended at ", datetime.datetime.now())
