import numpy as np
import tensorflow as tf
from sacred import Experiment
from keras.models import clone_model, Model
from keras.layers import Conv2D, Dense, Input, Add, Flatten
from keras.initializers import RandomUniform
from keras import backend as K
from collections import deque
import random
import gym
import signal


def signal_handler(signal, frame):
    import pdb; pdb.set_trace()


signal.signal(signal.SIGINT, signal_handler)


ex = Experiment("ddpg")
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES


@ex.config
def cfg():
    # NN parameters
    channels = [32, 32, 32]
    kernels = [8, 4, 3]
    strides = [3, 2, 1]
    activations = ['relu'] * 3
    linear_units = 200

    # low dimensional NN parameters
    units = [400, 300]

    # image parameters
    img_size = 84
    n_images = 4

    # training parameters
    actor_lr = 1e-4
    critic_lr = 1e-3
    tau = 1e-2  # soft update tau
    gamma = 0.995  # discount factor for returns
    sigma = 0.8  # noise scale

    # general parameters
    reward_scale = 1e-2
    low_dimensional = True  # train on features?
    grayscale = True
    resize = True
    render = True
    replay_buffer_size = 10 ** 5
    replay_start_size = 1000
    batch_size = 200
    eval_n_episodes = 20
    eval_loop = 3
    target_update_interval = 1
    update_interval = 4
    sess = tf.Session()


class Actor(object):
    @ex.capture
    def __init__(self, action_shape, obs_shape, sess, tau):
        self.tau = tau
        self.sess = sess
        K.set_session(sess)
        self.t = 0
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        with tf.variable_scope("actor"):
            self.create_model()
            self.create_target_model()
            sess.run(tf.initialize_all_variables())

    @ex.capture
    def create_model(self, actor_lr, channels, kernels, strides, activations,
                     linear_units, low_dimensional, units):
        initializer = RandomUniform(-3e-4, 3e-4)
        i = 0
        self.inp = tf.placeholder(tf.float32, (None,) + self.obs_shape)
        input = Input(tensor=self.inp)

        x = input
        if low_dimensional:
            for u in units:
                x = Dense(u, activation='relu',
                          kernel_initializer=initializer,
                          bias_initializer=initializer)(x)
        else:
            for c, k, s, a in zip(channels, kernels, strides, activations):
                x = Conv2D(c, k, strides=(s, s), activation=a,
                           kernel_initializer=initializer,
                           bias_initializer=initializer)(x)
                i += 1
            x = Flatten()(x)
            x = Dense(linear_units,
                      activation='relu',
                      kernel_initializer=initializer,
                      bias_initializer=initializer)(x)
        output = Dense(self.action_shape[0], activation=K.tanh,
                       kernel_initializer=initializer,
                       bias_initializer=initializer)(x)
        self.output = output
        self.model = Model(input, output)

        self.q_grads = tf.placeholder(tf.float32, (None, self.action_shape[0]))
        self.params = tf.trainable_variables(scope="actor")
        self.pi_grads = tf.gradients(self.output, self.params, -self.q_grads)
        clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in self.pi_grads]
        self.optimize = tf.train.AdamOptimizer(actor_lr).apply_gradients(zip(clipped_grads, self.params))

    def create_target_model(self):
        obs = Input(self.obs_shape)
        self.target_model = clone_model(self.model, obs)

    def target(self, obs):
        return self.target_model.predict([obs])

    def sync_target_model(self):
        self.target_model.set_weights(
            [w * self.tau for w in self.model.get_weights()] +
            [w * (1 - self.tau) for w in self.target_model.get_weights()])

    @ex.capture
    def train(self, states, q_grads, _run):
        self.sess.run([self.optimize],
                      feed_dict={self.q_grads: np.array(q_grads)[0],
                                 self.inp: states})

    def __call__(self, state):
        return self.sess.run(self.output, feed_dict={self.inp: state})


class Critic(object):
    @ex.capture
    def __init__(self, action_shape, obs_shape, sess, tau):
        self.tau = tau
        self.t = 0
        K.set_session(sess)
        self.sess = sess

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        with tf.variable_scope("critic"):
            self.create_model()
            self.create_target_model()

    @ex.capture
    def create_model(self, critic_lr, channels, kernels, strides, activations,
                     linear_units, units, low_dimensional):
        initializer = RandomUniform(-3e-4, 3e-4)

        self.obs = tf.placeholder(tf.float32, (None,) + self.obs_shape)
        self.action = tf.placeholder(tf.float32, (None,) + self.action_shape)
        obs = Input(tensor=self.obs)
        action = Input(tensor=self.action)
        i = 0
        x = obs
        if low_dimensional:
            for u in units:
                x = Dense(u, activation='relu',
                          kernel_initializer=initializer,
                          bias_initializer=initializer)(x)
            y = Dense(units[-1], activation='relu',
                      kernel_initializer=initializer,
                      bias_initializer=initializer)(action)
            x = Add()([x, y])
        else:
            for c, k, s, a in zip(channels, kernels, strides, activations):
                x = Conv2D(c, k, strides=(s, s), activation=a,
                           kernel_initializer=initializer,
                           bias_initializer=initializer)(x)
            x = Flatten()(x)
            # this to output layer
            x = Dense(linear_units,
                      activation='relu',
                      kernel_initializer=initializer,
                      bias_initializer=initializer)(x)
            y = Dense(linear_units,
                      activation='relu',
                      kernel_initializer=initializer,
                      bias_initializer=initializer)(action)
            x = Add()([x, y])
        output = Dense(1, kernel_initializer=initializer,
                       bias_initializer=initializer)(x)
        self.output = output
        self.model = Model(inputs=[obs, action], outputs=output)

        self.gradients = tf.gradients(self.output, self.action)
        self.q_targets = tf.placeholder(tf.float32, (None, 1))
        self.loss = tf.reduce_mean(tf.squared_difference(self.output,
                                                         self.q_targets))
        self.optimizer = tf.train.AdamOptimizer(critic_lr)
        self.params = tf.trainable_variables(scope='critic')
        grads = self.optimizer.compute_gradients(self.loss)
        clipped_grads = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads]
        self.optimize = self.optimizer.apply_gradients(clipped_grads)

    def create_target_model(self):
        obs = Input(shape=self.obs_shape)
        action = Input(shape=self.action_shape)
        self.target_obs = obs
        self.target_action = action
        self.target_model = clone_model(self.model, [obs, action])

    def target(self, obs, action):
        return self.target_model.predict([obs, action])

    def sync_target_model(self):
        self.target_model.set_weights(
            [w * self.tau for w in self.model.get_weights()] +
            [w * (1 - self.tau) for w in self.target_model.get_weights()])

    def __call__(self, state, action):
        return self.model.predict([state, action])

    @ex.capture
    def train(self, targets, batch, _run):
        states = np.stack([exp['state'] for exp in batch])
        actions = np.stack([exp['action'] for exp in batch], axis=1)[0]
        self.sess.run(self.optimize, feed_dict={self.obs: states,
                                                self.action: actions,
                                                self.q_targets: targets})
        _run.log_scalar("critic_loss", self.loss)

    def compute_gradients(self, states, actions):
        return self.sess.run(self.gradients, feed_dict={self.obs: states,
                                                        self.action: actions})


class ReplayBuffer(object):
    def __init__(self, max_capacity):
        self.buffer = deque(maxlen=max_capacity)

    def append(self, x):
        self.buffer.append(x)

    def __call__(self, state, action, reward, new_state):
        self.append({'state': state, 'action': action,
                     'reward': reward, 'next_state': new_state})

    @ex.capture
    def sample(self, batch_size):
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)


class ImageProcessor(object):
    @ex.capture
    def __init__(self, image, grayscale, resize, img_size, n_images, sess):
        self.grayscale = grayscale
        self.resize = resize
        self.img_size = img_size
        self.images = deque(maxlen=n_images)
        self.sess = sess
        for _ in range(n_images):
            self(image)

    def process_image(self, image):
        if self.resize:
            image = tf.image.resize_images(image, tf.constant([self.img_size, self.img_size]))
        if self.grayscale:
            image = tf.image.rgb_to_grayscale(image)
        return image

    @ex.capture
    def get_state(self, sess):
        return tf.squeeze(tf.stack(list(self.images), axis=2)).eval(session=sess)

    def __call__(self, image):
        self.images.append(self.process_image(image))
        return self.get_state()


class AdditiveOU(object):
    """Additive Ornstein-Uhlenbeck process.
    Used in https://arxiv.org/abs/1509.02971 for exploration.
    Args:
        mu (float): Mean of the OU process
        theta (float): Friction to pull towards the mean
        sigma (float or ndarray): Scale of noise
        start_with_mu (bool): Start the process without noise
    """
    @ex.capture
    def __init__(self, mu=0.0, theta=0.15, sigma=0.3, start_with_mu=False, action_shape=None):
        self.mu = mu
        self.a_shape = action_shape
        self.theta = theta
        self.sigma = sigma
        self.start_with_mu = start_with_mu
        self.ou_state = None

    def evolve(self):
        # dx = theta (mu - x) + sigma dW
        # for a Wiener process W
        noise = np.random.normal(size=self.ou_state.shape, loc=0,
                                 scale=self.sigma)
        self.ou_state += self.theta * (self.mu - self.ou_state) + noise

    @ex.capture
    def __call__(self, _log):
        if self.ou_state is None:
            if self.start_with_mu:
                self.ou_state = np.full(self.a_shape, self.mu, dtype=np.float32)
            else:
                sigma_stable = (self.sigma /
                                np.sqrt(2 * self.theta - self.theta ** 2))
                self.ou_state = np.random.normal(
                    size=self.a_shape,
                    loc=self.mu, scale=sigma_stable).astype(np.float32)
        else:
            self.evolve()
        noise = self.ou_state
        _log.debug("noise: {}".format(noise))
        return noise


@ex.capture
def train_critic(batch, actor, critic, gamma):
    # compute the target values
    rewards = np.array([exp['reward'] for exp in batch])
    next_states = np.array([exp['next_state'] for exp in batch])
    onpolicy_actions = actor.target(next_states)
    targets = rewards + gamma * critic.target(next_states, onpolicy_actions)
    critic.train(targets, batch)


@ex.capture
def train_actor(batch, actor, critic):
    states = np.array([exp['state'] for exp in batch])
    actions = actor(states)
    q_grads = critic.compute_gradients(states, actions)
    actor.train(states, q_grads)


@ex.capture
def env_loop(env, actor, critic, replay_buffer, noise_generator, train, sess, render, low_dimensional, n_images, replay_start_size, reward_scale, update_interval, target_update_interval, _log):
    R = 0
    i = 0
    done = False
    state = env.reset()
    pixels = env.render(mode='rgb_array', close=not render)
    if not low_dimensional:
        proc = ImageProcessor(pixels)
        state = proc.get_state()
    while not done:
        action = actor(np.stack([state])) * 2.0
        _log.debug("a: {}".format(action))
        if train:
            action += noise_generator()
        # step the environment and record new state
        new_state, reward, done, _ = env.step(action)
        reward *= reward_scale
        _log.debug("r: {}".format(reward))
        pixels = env.render(mode='rgb_array', close=not render)
        if not low_dimensional:
            new_state = proc(pixels)
        else:
            new_state = new_state[:, -1]
        R += reward

        # store the experience on the replay buffer
        if train:
            replay_buffer(state, action, reward, new_state)
            state = new_state

        # train if approporiate
        if replay_buffer is None:
            continue
        if len(replay_buffer) > replay_start_size and train:
            batch = replay_buffer.sample()
            if i % update_interval == 0:
                train_actor(batch, actor, critic)
                train_critic(batch, actor, critic)
            if i % target_update_interval == 0:
                critic.sync_target_model()
                actor.sync_target_model()
        i += 1
    return R


@ex.automain
def run_experiment(img_size, n_images, replay_buffer_size, eval_n_episodes, eval_loop, _run, sess, low_dimensional, _log):
    # start the code
    # create the OpenAI Gym environment
    env = gym.make("Pendulum-v0")
    env.reset()
    if not low_dimensional:
        obs_shape = (img_size, img_size, n_images)
        action_shape = env.action_space.shape
    else:
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
    critic = Critic(action_shape, obs_shape)
    actor = Actor(action_shape, obs_shape)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    noise_generator = AdditiveOU(action_shape=action_shape)
    eps = 0
    evals = 0
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_all_variables())
    while True:
        for _ in range(eval_n_episodes):
            eps += 1
            reward = env_loop(env, actor, critic, replay_buffer, noise_generator, True)
            _log.info("episode: {}, reward: {}".format(eps, reward))
            _run.log_scalar("reward", reward, eps)
        R = 0
        for _ in range(eval_loop):
            R += env_loop(env, actor, critic, None, None, False)
        _run.log_scalar("eval_reward", R / float(eval_loop), evals)
        evals += 1
