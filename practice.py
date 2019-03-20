# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:39:22 2019

@author: HP
"""

import tensorflow as tf

x_data = [1.0, 2.0, 3.0]
y_data = [1.0, 2.0, 3.0]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

#x = tf.placeholder(tf.float32, shape = None)
#y = tf.placeholder(tf.float32, shape = None)
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])


h = X * W + b

loss = tf.reduce_mean(tf.square(h - Y))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
    
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([loss, W, b, train],
        feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))
            
 