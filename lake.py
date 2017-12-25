import gym
import numpy as np
import tensorflow as tf

env = gym.make('FrozenLake-v0')

#network
tf.reset_default_graph()
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout,1)

#loss
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

#training the network
init = tf.initialize_all_variables()

#learning parameters
y = .99
e = 0.1
num_episodes = 2000
#total rewards and steps per episode
jList = []
rList = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        while j < 99:
            j += 1
            a, allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            s1,r,d,_ = env.step(a[0])
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1

            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                e = 1./((i/50)+10)
                break
        jList.append(j)
        rList.append(rAll)
        print(i)
        print(sess.run(W))
        
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"

"""
cia eina komentaras
ismokti weigths:
[[  5.42983353e-01   5.04076540e-01   4.03621495e-01   4.40591514e-01]
 [  1.74082756e-01   2.71232516e-01   1.25533387e-01   4.49745655e-01]
 [  3.08152169e-01   2.07988426e-01   1.37857050e-01   2.27358818e-01]
 [  1.74072105e-02   1.08049735e-01   1.44456252e-02   2.60631442e-02]
 [  5.65921366e-01   3.79958570e-01   2.25518942e-01   4.57451373e-01]
 [  2.49746675e-03   2.26927758e-03   8.15584138e-03   4.27758461e-03]
 [  1.12015560e-01   1.90115292e-02   1.57285035e-01   4.51748371e-02]
 [  3.91275994e-03   4.88414778e-04   5.84173296e-03   6.94582099e-03]
 [  3.70505154e-01   2.73712099e-01   3.82843405e-01   6.29320025e-01]
 [  4.52350974e-01   6.93005323e-01   3.74327958e-01   2.47521922e-01]
 [  5.80835760e-01   2.73601800e-01   2.54539877e-01   2.80731738e-01]
 [  9.82276537e-03   9.06819105e-03   7.44858384e-03   9.31117497e-03]
 [  3.27176554e-03   4.39413171e-03   8.54636636e-03   5.81230270e-03]
 [  4.78767902e-01   6.60632908e-01   7.90444553e-01   5.72825253e-01]
 [  7.54117072e-01   9.65344548e-01   7.11807668e-01   7.12288558e-01]
 [  4.56856471e-03   8.59636255e-03   3.93604627e-03   1.62780289e-05]]
"""
