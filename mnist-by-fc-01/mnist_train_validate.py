import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100

def forward_propagation(input_data_x, variable_average, weight1, bias1, weight2, bias2):
    if variable_average == None:
        hidden_layer = tf.nn.relu(tf.matmul(input_data_x, weight1) + bias1)
        return tf.matmul(hidden_layer, weight2) + bias2
    else:
        hidden_layer = tf.nn.relu(tf.matmul(input_data_x, variable_average.average(weight1))+variable_average.average(bias1))
        return tf.matmul(hidden_layer, variable_average.average(weight2)) + variable_average.average(bias2)
        
def train(mnist):
    #step 1.2------define placeholder for input data
    INPUT_LAYER_NODE_NUMMBER = 28 * 28
    OUTPUT_LAYER_NODE_NUMMBER = 10
    input_data_x = tf.placeholder(tf.float32, [None, INPUT_LAYER_NODE_NUMMBER], name = "input_data_x")
    input_data_y = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_NODE_NUMMBER], name = "input_data_y")
    
    #step 2------define network structure
    HIDEEN_LAYER_NODE_NUMMBER = 500
    weight1 = tf.Variable(tf.truncated_normal([INPUT_LAYER_NODE_NUMMBER, HIDEEN_LAYER_NODE_NUMMBER], stddev = 0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape = [HIDEEN_LAYER_NODE_NUMMBER]))
    weight2 = tf.Variable(tf.truncated_normal([HIDEEN_LAYER_NODE_NUMMBER, OUTPUT_LAYER_NODE_NUMMBER], stddev = 0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_LAYER_NODE_NUMMBER]))
    
    #step 3.1------calculate forward progation
    y_prediction = forward_propagation(input_data_x, None, weight1, bias1, weight2, bias2)
    #step 3.2------calculate forward progation again through moving average algorithm
    MOVING_AVERAGE_DECAY = 0.99
    global_steps = tf.Variable(0, trainable = False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_steps)
    moving_average_operation = variable_average.apply(tf.trainable_variables())
    y_average = forward_propagation(input_data_x, variable_average, weight1, bias1, weight2, bias2)
    
    #step 4.1------calculate loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.argmax(input_data_y, 1), logits = y_prediction)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #step 4.2------optimization(L2-regularization,exponential-decay)
    REGULARIZATION_RATE = 0.0001
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weight1) + regularizer(weight2)

    #step 4.3------loss with optimization
    LEARNING_RATE_BASE = 0.8
    LEARNING_RATE_DECAY = 0.99
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_steps, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_steps)

    #step 5------define dependency
    with tf.control_dependencies([train_step, moving_average_operation]):
        train_op = tf.no_op(name = "train")

    #step 6------calculate accuracy
    accuracy_prediction = tf.equal(tf.argmax(y_average, 1), tf.argmax(input_data_y, 1))
    accuracy = tf.reduce_mean(tf.cast(accuracy_prediction, tf.float32))

    #step 7------execution
    TRAINING_STEPS = 30000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        validate_feed = {input_data_x: mnist.validation.images, input_data_y: mnist.validation.labels}
        test_feed = {input_data_x: mnist.test.images, input_data_y: mnist.test.labels}
        for steps in range(1, TRAINING_STEPS):
            input_data, output_data = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value = sess.run([train_op, loss], feed_dict = {input_data_x: input_data, input_data_y: output_data})
            if steps % 1000 == 0:
                validate_accuracy = sess.run(accuracy, feed_dict = validate_feed)
                #print("After %d steps, validation accuracy using average model is %g" %(steps, validate_accuracy))
                print(loss_value)
        test_accuracy = sess.run(accuracy, feed_dict = test_feed)
        print("After %d steps, test accuracy using average model is %g" %(TRAINING_STEPS, test_accuracy))

def main():
    #step 1.1------define a object to process input data
    mnist = input_data.read_data_sets("mnist_data", one_hot = True)
    train(mnist)

if __name__ == "__main__":
    main()
