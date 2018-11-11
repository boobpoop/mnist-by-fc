import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward_propagation
import os

MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "model"
MODEL_NAME = "model.ckpt"
BATCH_SIZE = 100
def train(mnist):
    #step 1.2------define placeholder of input datas
    input_data_x = tf.placeholder(tf.float32, [None, forward_propagation.INPUT_NODE], name = "input_data_x")
    input_data_y = tf.placeholder(tf.float32, [None, forward_propagation.OUTPUT_NODE], name = "input_data_y")

    
    #step 3------calculate forward propagation
    REGULARIZATION_RATE = 0.0001
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = forward_propagation.forward_propagation(input_data_x, regularizer)
   
    #step 4------define loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.argmax(input_data_y, 1), logits = y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    #step 5.1------optimization & train(moving average)
    global_steps = tf.Variable(0, trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_steps)
    moving_operation = variable_averages.apply(tf.trainable_variables())
    
    #step 5.2------optimization & train(learning_rate)
    LEARNING_RATE_BASE = 0.1
    LEARNING_RATE_DECAY = 0.99
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_steps, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    
    #step 5.3------train
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_steps)
    
    #step 6------define dependency
    with tf.control_dependencies([train_step, moving_operation]):
        train_operation = tf.no_op(name = "train")

    for variables in tf.global_variables():
        print(variables)

    #step 7------define a object to save model
    saver = tf.train.Saver(max_to_keep = 0)

    #step 8------execution
    TRAINING_STEP = 5000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for steps in range(1, TRAINING_STEP):
            input_data, output_data = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value = sess.run([train_operation, loss], feed_dict = {input_data_x: input_data, input_data_y: output_data})
            if steps % 1000 == 0:
                print("After %d steps, loss on training batch is %g" %(steps, loss_value))
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_steps)




def main(argv = None):
    #step 1.1------define a object to process input data
    mnist = input_data.read_data_sets("mnist_data", one_hot = True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()
    
