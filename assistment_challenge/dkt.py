import os
import random
import time
import tensorflow as tf
import numpy as np

from numpy.random import permutation as perm
from sklearn.metrics import roc_curve, auc


DATA_DIR = './'
train_file = os.path.join(DATA_DIR, 'train.csv')
test_file = os.path.join(DATA_DIR, 'test.csv')


####
# missed: mini-batch, N-fold cross validation
# save the model: the structure of neural network, the weight matrix, the bias
# run on the gpu
# other rnn structure: bi-direction rnn
####

def read_data(filename):
    # num_prob shall be the info from the dataset.
    # num_steps_max shall not be specified.
    records = []
    num_steps_max = 0
    num_probs = 0
    
    with open(filename, 'r') as f:
        
	#num_steps: num_of_responses of each stu
	num_steps, seq_probs, seq_tags = None, None, None
        for i, row in enumerate(f):
            try:
                row_0 = row
                row = map(int, row.strip().split(","))
                if i % 3 == 0:
                    num_steps = row[0]        
		elif i % 3 == 1:
                    seq_probs = row
                elif i % 3 == 2:
                    seq_tags = row
                    #only consider the records whose number of interactions >= 3
 
                    if (num_steps >= 3) and num_steps and seq_probs and seq_tags:
                        num_steps_max = max(num_steps_max, num_steps)
                        num_probs = max([num_probs] + seq_probs)
                        records += [(num_steps, seq_probs, seq_tags)]
            except:
                if i % 3 == 0:
                    num_steps = None
                elif i % 3 == 1:
                    seq_probs = None
                elif i % 3 == 2:
                    seq_tags = None
                    
                print "- broken line in {} : {}".format(i, row_0)
                
    return records, num_steps_max, num_probs+1




'''
pad each original student record to the size of num_step_max by:
adding list of -1 to skill_id so that # of skill_id == num_step_max
adding list of 0 to response so that # of response == num_step_max
'''
def padding(record, length):
    n = length - record[0]
    return (record[0], record[1]+[-1]*n, record[2]+[0]*n)




records_train, num_steps_max_train, num_probs_train = read_data(train_file)
records_train = [padding(student_tuple, num_steps_max_train) for student_tuple in records_train]
# num_steps_max_train:
# num_probs_train:

records_test, num_steps_max_test, num_probs_test = read_data(test_file)
records_test = [padding(student_tuple, num_steps_max_train) for student_tuple in records_test]
# num_steps_max_test:
# num_probs_test: 

# seq_probs, seq_tags shape = [batch_size, num_steps]

def seq_onehot(seq_probs, seq_tags, num_steps, num_probs):
    seq_probs_ = tf.one_hot(seq_probs, depth=num_probs)
    seq_probs_flat = tf.reshape(seq_probs_, [-1, num_probs])
    
    # element-wise multiplication between Matrix and Vector
    # the i-th column of Matrixelement-wisedly multiply the i-th element in the Vector    
    seq_tags_ = tf.cast(tf.reshape(seq_tags, [-1]), dtype=tf.float32)
    seq_tags_ = tf.multiply(tf.transpose(seq_probs_flat), seq_tags_)
    seq_tags_ = tf.reshape(tf.transpose(seq_tags_), shape=[-1, num_steps, num_probs])
    
    #   x       x_oh      y    input_oh
    #  >=0       1        0       -1
    #  >=0       1        1        1
    #  -1        0(None)  0(None)  0
    
    return seq_tags_ * 2 - seq_probs_, seq_tags_




# feed_in config
batch_size = 32


# network config
num_steps = num_steps_max_train
num_probs = num_probs_train

num_layers = 2
state_size = 200


# X_in (seq_probs)   : one hot 1 and -1 / one hot 1 
# Y_in (seq_tags)    : one hot 1
#                    : batch_size x num_steps x num_probs
X_ph = tf.placeholder(tf.int32, [None, num_steps])
Y_ph = tf.placeholder(tf.int32, [None, num_steps])
keep_prob_ph = tf.placeholder(tf.float32)

X_in, Y_in = seq_onehot(X_ph, Y_ph, num_steps, num_probs)

init_states_ph = tf.placeholder(tf.float32, [num_layers, 2, None, state_size])
init_states = tf.unstack(init_states_ph, axis=0)
init_states = tuple([tf.contrib.rnn.LSTMStateTuple(init_states[idx][0], init_states[idx][1]) for idx in range(num_layers)])


## build up the network
cells = [tf.contrib.rnn.LSTMCell(num_units=state_size, forget_bias=1.0, state_is_tuple=True) for _ in range(num_layers)]
cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob_ph) for cell in cells]
# cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
#
# # inputs: batch_size x num_steps x features
# # rnn_outputs: batch_size x num_steps x state_size
# rnn_outputs, final_state = tf.nn.dynamic_rnn(cells, X_in, initial_state=init_states, time_major=False)

rnn_outputs_in_list = []
rnn_inputs = X_in
for i, cell in enumerate(cells):
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_states[i], time_major=False, scope="rnn-layer-"+str(i))
    rnn_outputs_in_list += [rnn_outputs]
    rnn_inputs = rnn_outputs

print "the states series is: ", rnn_outputs
print "the final_state is: ", final_state


with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_probs])
    b = tf.get_variable('b', [num_probs], initializer = tf.constant_initializer(0.0))

Y_out = tf.matmul(tf.reshape(tf.tanh(rnn_outputs), [-1, state_size]), W) + b
Y_out = tf.sigmoid(tf.reshape(Y_out, [-1, num_steps, num_probs]))


# Y_out: batch_size x num_steps x num_probs
_, X_in_next = tf.split(X_in, num_or_size_splits = [1, num_steps-1], axis=1)
Y_out_cur, _ = tf.split(Y_out, num_or_size_splits = [num_steps-1, 1], axis=1)
_, Y_in_next = tf.split(Y_in, num_or_size_splits = [1, num_steps-1], axis=1)

print X_in_next, Y_out_cur, Y_in_next

# this code block calculate the loss using tf.gather_nd
idx_selected = tf.where(tf.not_equal(X_in_next, 0))
Y_out_selected = tf.gather_nd(Y_out_cur, idx_selected)
Y_in_selected = tf.gather_nd(Y_in_next, idx_selected)


loss = -Y_in_selected * tf.log(Y_out_selected) - (1-Y_in_selected) * tf.log(1-Y_out_selected)
total_loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(total_loss)




def shuffle(records):
    """
    mini-batch
    """

    size = len(records)
    batch = batch_size
    if batch > size:
        batch = size
    
    batch_per_epoch = int(size/batch)
    total = num_epochs * batch_per_epoch
    yield(total)
    
    for i in range(num_epochs):
        shuffle_idx = perm(np.arange(size))
        
        # each batch of i_th num_epochs
        for b in range(batch_per_epoch):
            
            # two yieldees
            x_batch = list()
            y_batch = list()
            for real_idx in shuffle_idx[(b*batch) : ((b+1)*batch)]:
                x_inp = records[real_idx][1]
                y_inp = records[real_idx][2]
                x_batch += [np.expand_dims(x_inp, 0)]
                y_batch += [np.expand_dims(y_inp, 0)]
            
            if b+1 == batch_per_epoch:
                for real_idx in shuffle_idx[(b+1)*batch:]:
                    x_inp = records[real_idx][1]
                    y_inp = records[real_idx][2]
                    x_batch += [np.expand_dims(x_inp, 0)]
                    y_batch += [np.expand_dims(y_inp, 0)]
            
            x_batch = np.concatenate(x_batch, 0)
            y_batch = np.concatenate(y_batch, 0)
            yield(x_batch, y_batch, i, b, (b+1)==batch_per_epoch)




def evaluate(sess, is_train=False):
    records = records_train if is_train else records_test
    
    y_pred = []
    y_true = []
    num_records = len(records)
    for batch_idx in range(0, num_records, batch_size):
        start_idx = batch_idx
        end_idx = min(num_records, batch_idx+batch_size)
        
        new_batch_size = end_idx - start_idx
        rnn_init_state_batch = np.zeros((num_layers, 2, new_batch_size, state_size))
        
        x_batch = np.array([record[1] for record in records[start_idx:end_idx]], dtype=np.int32)
        y_batch = np.array([record[2] for record in records[start_idx:end_idx]], dtype=np.int32)
         
        prob_pred, prob_true = sess.run( (Y_out_selected, Y_in_selected),
                                         feed_dict={ X_ph: x_batch,
                                                     Y_ph: y_batch,
                                                     init_states_ph: rnn_init_state_batch,
                                                     keep_prob_ph: 1.0,
                                                     }
                                         )
        
        y_pred += [p for p in prob_pred]
        y_true += [t for t in prob_true]
        
    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    return auc_score




def optimize(sess):
    """
    the missed: mini-batch
    """
    records = records_train
    num_records = len(records)
    
    batches = shuffle(records)
    
    for i, packet in enumerate(batches):
        if i == 0:
            total = packet
        else:
            x_batch, y_batch, idx_epoch, idx_batch, end_batch = packet
            cur_batch_size = x_batch.shape[0]
            rnn_init_state_batch = np.zeros((num_layers, 2, cur_batch_size, state_size))
            sess.run(optimizer,
                     feed_dict={ X_ph: x_batch,
                                 Y_ph: y_batch,
                                 init_states_ph: rnn_init_state_batch,
                                 keep_prob_ph: 1.0,
                                 }
                     )
            
            if idx_batch % 10 == 0:
                total_loss_eval, = sess.run((total_loss, ),
                                            feed_dict={ X_ph: x_batch,
                                                        Y_ph: y_batch,
                                                        init_states_ph: rnn_init_state_batch,
                                                        keep_prob_ph: 1.0,
                                                        }
                                            )
                print("Epoch {0:>4}, iteration {1:>4}, batch loss value: {2:.5}".format(idx_epoch, idx_batch, total_loss_eval))

            if end_batch:
                auc_train = evaluate(sess, is_train=True)
                auc_test = evaluate(sess, is_train=False)
                print("Epoch {0:>4}, Training AUC: {1:.5}, Testing AUC: {2:.5}".format(idx_epoch, auc_train, auc_test))



WITH_CONFIG = True
num_epochs = 2

start_time = time.time()
if WITH_CONFIG:
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	sess.run(tf.global_variables_initializer())
        optimize(sess)
	writer.close()
else:
    with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	sess.run(tf.global_variables_initializer())
        optimize(sess)
	writer.close()
           
end_time = time.time()

print("program run for: {0}s".format(end_time-start_time))


