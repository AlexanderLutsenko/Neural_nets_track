import tensorflow as tf
from tensorflow.contrib import layers
import time
import cifar10_data



# Fully convolutional model created using existing layers
def conv_net_fullconv(x, is_train, n_classes):

    batchnorm_decay = 0.95

    x = layers.conv2d(x, 32, 3, activation_fn=None)
    x = layers.batch_norm(x, is_training=is_train, decay=batchnorm_decay, updates_collections=None)
    x = tf.nn.elu(x)
    x = layers.max_pool2d(x, 2)

    x = layers.conv2d(x, 64, 3, activation_fn=None)
    x = layers.batch_norm(x, is_training=is_train, decay=batchnorm_decay, updates_collections=None)
    x = tf.nn.elu(x)
    x = layers.max_pool2d(x, 2)

    x = layers.conv2d(x, 128, 3, activation_fn=None)
    x = layers.batch_norm(x, is_training=is_train, decay=batchnorm_decay, updates_collections=None)
    x = tf.nn.elu(x)

    x = layers.conv2d(x, 256, 3, activation_fn=None)
    x = layers.batch_norm(x, is_training=is_train, decay=batchnorm_decay, updates_collections=None)
    x = tf.nn.elu(x)

    x = layers.conv2d(x, n_classes, 3, activation_fn=None)
    x = layers.batch_norm(x, is_training=is_train, decay=batchnorm_decay, updates_collections=None)
    x = tf.nn.elu(x)

    x = layers.dropout(x, 0.5, is_training=is_train)

    x = tf.reduce_max(x, reduction_indices=[1,2])
    x = tf.reshape(x, shape=[-1, n_classes])
    x = tf.nn.softmax(x, dim=-1)
    return x




# Self-made model created using low level TensorFlow API
def conv_net_shallow_selfmade(x, is_train, n_classes):

    def convolution(x, kernelHeight, kernelWidth, inputChannels, outputChannels, strides=1, initializer=tf.random_normal):
        weights = tf.Variable(initializer([kernelWidth, kernelHeight, inputChannels, outputChannels]))
        biases = tf.Variable(initializer([outputChannels]))
        x = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, biases)
        return x

    def maxpool(x, k):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def reshape(x, size):
        return tf.reshape(x, [-1, size])

    def dropout(x, keep_prob, is_train):
        keep_prob_const = tf.constant(keep_prob)
        return tf.cond(is_train,
                       lambda: tf.nn.dropout(x, keep_prob_const),
                       lambda: x)

    def dense(x, numInputs, numOutputs, initializer=tf.random_normal):
        weights = tf.Variable(initializer([numInputs, numOutputs]))
        biases = tf.Variable(initializer([numOutputs]))
        x = tf.matmul(x, weights)
        x = tf.add(x, biases)
        return x

    def batch_norm(x, is_training, axes, decay=0.95, epsilon=1e-3):
        scale = tf.Variable(tf.ones([x.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([x.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([x.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([x.get_shape()[-1]]), trainable=False)

        def train():
            batch_mean, batch_var = tf.nn.moments(x, axes)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

        def predict():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

        return tf.cond(is_training, train, predict)

    def dense_batch_norm(x, is_training, decay=0.95, epsilon=1e-3):
        axes = [0]
        return batch_norm(x, is_training, axes=axes, decay=decay, epsilon=epsilon)

    def convolutional_batch_norm(x, is_training, decay=0.95, epsilon=1e-3):
        # For convolutional layers, we additionally want the normalization to obey the convolutional property –
        # so that different elements of the same feature map, at different locations, are normalized in the same way.
        axes = [0, 1, 2]
        return batch_norm(x, is_training, axes=axes, decay=decay, epsilon=epsilon)



    initializer = lambda shape: tf.random_normal(shape, stddev=0.01)

    x = convolution(x, 5, 5, 3, 32, initializer=initializer)
    x = convolutional_batch_norm(x, is_training=is_train)
    x = tf.nn.elu(x)
    x = maxpool(x, 2)

    x = convolution(x, 5, 5, 32, 64, initializer=initializer)
    x = convolutional_batch_norm(x, is_training=is_train)
    x = tf.nn.elu(x)
    x = maxpool(x, 2)

    x = convolution(x, 5, 5, 64, 128, initializer=initializer)
    x = convolutional_batch_norm(x, is_training=is_train)
    x = tf.nn.elu(x)
    x = maxpool(x, 2)

    x = tf.reshape(x, shape=[-1, 4 * 4 * 128])

    x = dense(x, 4 * 4 * 128, 1024, initializer=initializer)
    x = dense_batch_norm(x, is_training=is_train)
    x = tf.nn.elu(x)

    x = dropout(x, 0.5, is_train)

    x = dense(x, 1024, n_classes, initializer=initializer)

    x = tf.nn.softmax(x, dim=-1)
    return x





if __name__ == '__main__':
    input_h = input_w = 32
    n_classes = 10

    training_iters = 500
    batch_size = 32
    display_step = 10

    stat_folder = 'stat/{}'.format(time.strftime("%c"))

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, input_h, input_w, 3])
    target = tf.placeholder(tf.float32, [None, n_classes])
    is_train = tf.placeholder(tf.bool)

    # Construct model
    # model = conv_net_fullconv(x, is_train, n_classes)
    model = conv_net_shallow_selfmade(x, is_train, n_classes)


    # Define loss and optimizer
    def cross_entropy(y, t):
        l = -tf.reduce_sum(t * tf.log(y), reduction_indices=[1])
        return tf.reduce_mean(l)

    L = cross_entropy(model, target)

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1).minimize(L)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # We shall keep track of training loss. Training plot can be seen on TensorBoard:
    # 1. Run TensorBoard: python3 -m tensorflow.tensorboard --logdir=<project path>/stat
    # 2. Open TensorBoard web-face
    # https://www.tensorflow.org/how_tos/summaries_and_tensorboard/
    tf.scalar_summary("loss", L)
    summary = tf.merge_all_summaries()

    (images_train, labels_train), (images_test, labels_test) = cifar10_data.get_data()

    # Launch the graph
    with tf.Session() as sess:
        # Initialize the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Create summary writer
        summary_writer = tf.train.SummaryWriter(stat_folder, sess.graph)
        # Display computational graph on TensorBoard.
        summary_writer.add_graph(sess.graph)

        # Keep training until reach max iterations
        for step in range(training_iters):
            batch_x, batch_t = cifar10_data.get_batch(images_train, labels_train, batch_size)

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, target: batch_t, is_train: True})

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([L, accuracy], feed_dict={x: batch_x, target: batch_t, is_train: False})
                print("Iter {}, Minibatch Loss = {:.5f}, Training accuracy = {:.5f}".format(step, loss, acc))

                # Collect and write specified training summary.
                summary_str = sess.run(summary, feed_dict={x: batch_x, target: batch_t, is_train: False})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

        print("Optimization Finished!")

        # Calculate accuracy for test images
        testingAccuracy = sess.run(accuracy, feed_dict={x: images_test, target: labels_test, is_train: False})
        print("Testing Accuracy:", testingAccuracy)
