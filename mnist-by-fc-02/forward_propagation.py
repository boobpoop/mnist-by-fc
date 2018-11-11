import tensorflow as tf

#step 2------define network structure
INPUT_NODE = 784
HIDDEN_NODE = 500
OUTPUT_NODE = 10

def get_weight(shape, regularizer):
    weight = tf.get_variable("weight", shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weight))
    return weight

def forward_propagation(input_data_x, regularizer):
    with tf.variable_scope("layer1"):
        weight = get_weight([INPUT_NODE, HIDDEN_NODE], regularizer)
        bias = tf.get_variable("bias", [HIDDEN_NODE], initializer = tf.constant_initializer(0.0))
        hidden_layer = tf.nn.relu(tf.matmul(input_data_x, weight) + bias)

    with tf.variable_scope("layer2"):
        weight = get_weight([HIDDEN_NODE, OUTPUT_NODE], regularizer)
        bias = tf.get_variable("bias", [OUTPUT_NODE], initializer = tf.constant_initializer(0.0))
        output_layer = tf.matmul(hidden_layer, weight) + bias
    return output_layer
