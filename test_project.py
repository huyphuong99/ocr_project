import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib.pyplot as plt
import numpy as np
import utils

def linear_regression():
    x_train = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    y_train = np.asarray([0.1, 0.2, 0.32, 0.43, 0.54, 0.65, 0.77, 0.88, 0.94, 1])
    n_sample = x_train.shape[0]
    x_ = tf.placeholder(tf.float32, name="x")
    y_ = tf.placeholder(tf.float32, name="y")
    w = tf.get_variable("weights", initializer=tf.constant(0.0))
    b = tf.get_variable("bias", initializer=tf.constant(0.0))
    y_predict = w * x_ + b
    loss = tf.square(y_ - y_predict, name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    writer = tf.summary.FileWriter("./graphs", tf.get_default_graph())
    writer.close()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            total_loss = 0
            for x, y in zip(x_train, y_train):
                _, _loss = sess.run([optimizer, loss], feed_dict={x_: x, y_: y})
                total_loss += _loss
            print(f"Epoch {i}: {total_loss / n_sample}")
        w_out, b_out = sess.run([w, b])
        y_predict = x_train * w_out + b_out
        for i, j in zip(y_predict, y_train):
            print(f"{i} : {j}")
        plt.plot(x_train, y_predict, "r-", label="predict")
        plt.plot(x_train, y_train, "go", label="data")
        plt.title("ABC")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
# def logistic_regression():
#     learning_rate = 1e-2
#     batch_size = 128
#     n_epochs = 30
#     x_train, x_val, x_test = utils.read_mnist('notMnist')
#     x_batch, y_batch = utils.next_batch(batch_size, x_train)
#     print(x_train)

if __name__ == "__main__":
    linear_regression()
    # logistic_regression()