import gym
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

env = gym.make('CartPole-v0')

# neural network parameters
inp_size = 4
hidden_size = 10
output_size = 2
lr = 1e-2

#define the NN
#the input is an Nx4 matrix, where each row is a state (past states)
#the output is an Nx2 matrix, where each row contains 2 probabilities for each action
tf.reset_default_graph()

inp = tf.placeholder(tf.float32, shape=(None, inp_size))
hidden = slim.fully_connected(inp, hidden_size, biases_initializer=None,activation_fn=tf.nn.relu)
output = slim.fully_connected(hidden, output_size, activation_fn=tf.nn.softmax,biases_initializer=None)
chosen_action = tf.argmax(output, axis=1)

#the loss function

#past rewards
rewards = tf.placeholder(tf.float32, shape=(None))
#the indices of the actions that have been taken (of the reshaped output array)
actions = tf.placeholder(tf.int32, shape=(None))
#the probabilities of the chosen actions
chosen_action_probs = tf.gather(tf.reshape(output, [-1]), indices=actions)

#the loss function
loss = -tf.reduce_mean(tf.log(chosen_action_probs)*rewards)

#the optimizer
tvars = tf.trainable_variables()
adam = tf.train.AdamOptimizer(learning_rate=lr)
gradients = tf.gradients(loss, tvars)


gradient_placeholders = []
for idx,var in enumerate(tvars):
    gradient_placeholders.append(tf.placeholder(tf.float32, name=str(idx)+'_holder'))

grads_and_vars = zip(gradient_placeholders, tvars)
    
optimize = adam.apply_gradients(grads_and_vars)

#the openAI gym
total_episodes = 5000
batch_size = 5
init = tf.global_variables_initializer()
#the running gradients which will be used to update the network
running_gradients = np.array([np.zeros(var.get_shape(), dtype=np.float32) for var in tvars])

#history of episodes
history = []
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []

    while i < total_episodes:
        history = []
        i += 1
        observation = env.reset()
        running_reward = 0

        for _ in range(1000):
            #reshape the observation to be the same as the input to the network
            obs = np.reshape(observation, [1,inp_size])
            
            probabilities = sess.run(output, feed_dict={inp: obs})
            #choose an action with some probability
            action = np.argmax(probabilities[0] == np.random.choice(probabilities[0], p=probabilities[0]))

            observation, reward, done, _ = env.step(action)
            history.append([list(obs[0]), observation, reward, action])
            running_reward += reward

            if done:
                total_reward.append(running_reward)
                #calculate the gradients
                hist = np.array(history)
                discounted_r = discount_rewards(hist[:, 2])

                #compute the indexes of chosen actions in the reshaped output array
                adders = range(0, len(hist)*2, 2)
                for i in range(len(hist)):
                    hist[i,3] = int(hist[i,3]+adders[i])

                running_gradients += np.array(sess.run(gradients, feed_dict={inp: np.vstack(hist[:, 0]), rewards: discounted_r, actions:hist[:,3]}))

                if i % batch_size == 0:
                    #update the network
                    feed_dict = dict(zip(gradient_holders, running_gradients))
                    _ = sess.run(optimize,feed_dict=feed_dict)
                    running_gradients *= 0
                    #decrease the learning rate after a certain number of episodes
                    if i == 10*50:
                        lr = 1e-3
                break
        
        if i % 50 == 0:
            #calculate the mean reward
            mean_rew = np.mean(total_reward[-50:])
            print(mean_rew)
            #in this case, the environment has been solved
            if mean_rew > 190:
                break
        


    #now simulate
    i = 0
    observation = env.reset()
    import time
    while i < 500:
        env.render()
        time.sleep(0.1)
        i += 1
        obs = np.reshape(observation, [1,inp_size])
        action = sess.run(chosen_action, feed_dict={inp: obs})
        observation, reward, done, _ = env.step(action[0])
        if done:
            break


"""
tvars:
[array([[ 0.20112956, -0.09313253, -0.14390953, -0.10576689, -0.23242548,
        -0.03279512, -0.09256506,  0.05344922, -0.0145091 ,  0.17781442],
       [-0.55880147, -0.22839549, -0.18893851,  0.37783822,  0.35708189,
         0.10866284,  0.00271447,  0.5347783 , -0.65613276,  0.42583269],
       [-1.29040432,  1.4923923 , -0.59283566, -0.56314385,  1.33089828,
        -0.66093689,  0.00987334,  1.44511652, -0.56227231, -0.90460503],
       [-1.14751267,  1.22765768, -0.54962319, -0.49014434,  0.76880318,
        -0.46929008, -0.30716935,  0.51233935, -0.85033798, -0.2409526 ]], dtype=float32), array([[ 0.60550141, -0.76451445],
       [-1.15472567,  0.52468735],
       [ 0.32927713, -1.0753957 ],
       [ 0.83333629,  0.19086714],
       [-0.89725739,  0.16111165],
       [ 0.31483194, -0.11407088],
       [ 0.26224145,  0.18367949],
       [-0.9768433 ,  0.6018014 ],
       [ 0.96755159, -0.11213607],
       [ 0.44475088, -0.2767992 ]], dtype=float32)]

"""
