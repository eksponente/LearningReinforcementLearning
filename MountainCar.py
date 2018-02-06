#Solve the OpenAI Gym MountainCar environment
#Method: TD(lambda)
#Algorithm: SARSA
#Ation-value function approximation: Neural Network
#Optimization method: gradient descent
import gym
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


#define variables
inp_size = 3 #2 state variables and 1 action variable
h_size = 5
op_size = 1
lbd = 0.9 #lambda
alpha = 0.9
gamma = 0.9

#This neural network approximates the action value function
#Its inputs are the state and an action and outputs are the value
inp = tf.placeholder(tf.float32, shape=(1, inp_size))
hidden = slim.fully_connected(inp, h_size, activation_fn=tf.nn.relu)
output = slim.fully_connected(hidden, op_size, activation_fn=tf.nn.softmax)

#Now to update the neural network, we need to calculate the gradient of the action-value function
reward = tf.placeholder(tf.float32)
q_next = tf.placeholder(tf.float32)
q_current = tf.placeholder(tf.float32)
e_t = tf.placeholder(tf.float32)

gradients = tf.gradients(output, tf.trainable_variables())

gradient = tf.placeholder(tf.float32)

err = reward + lbd*q_next - q_current
#eligibility_trace = lbd*gamma*e_t + gradient
#delta_w = alpha*err*eligibility_trace

#let's see if this is going to work...

env = gym.make('MountainCar-v0')
init = tf.global_variables_initializer()

state_action = [[0., 0., 0.]]

with tf.Session() as sess:
    sess.run(init)
    import pdb; pdb.set_trace()
    env.reset()

    gradients = sess.run(gradients, feed_dict={inp:state_action})
    for i, gradient in enumerate(gradients):
        sess.run(err, feed_dict={reward: 0.0, q_next = })
        #sess.run(tf.trainable_variables()[i].assign(new))
        
        
