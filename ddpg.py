import numpy as np
import tensorflow as tf
from sacred import Experiment
from keras.models import clone_model, Model, Sequential
from keras.layers import Conv2D, Dense, Input, Add
from keras.initializers import RandomUniform
from keras.losses import mean_squared_error
from collections import deque
import random
import gym

ex = Experiment("ddpg")


@ex.config
def cfg():
    # NN parameters
    channels = [32, 32, 32]
    kernels = [8, 4, 3]
    strides = [3, 2, 1]
    activations = ['relu'] * 3
    linear_units = 200

    # image parameters
    img_size = 84
    n_images = 4

    # training parameters
    actor_lr = 0.001
    critic_lr = 0.0001
    tau = 0.001  # soft update tau
    gamma = 0.99  # discount factor for returns
    sigma = 0.2  # noise scale

    # general parameters
    low_dimensional = False  # train on features?
    grayscale = True
    resize = True
    render = True
    replay_buffer_size = 10 ** 4
    replay_start_size = 20
    batch_size = 16
    eval_n_episodes = 10
    eval_loop = 3


class Actor(object):
    @ex.capture
    def __init__(self, action_shape, obs_shape, tau):
        self.tau = tau
        self.t = 0
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        self.create_model()
        self.create_target_model()

    @ex.capture
    def create_model(self, actor_lr, channels, kernels, strides, activations,
                     linear_units):
        initializer = RandomUniform(-3e-4, 3e-4)
        model = Sequential()
        i = 0
        for c, k, s, a in zip(channels, kernels, strides, activations):
            if i == 0:
                model.add(Conv2D(c, k, strides=(s, s), activation=a,
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer,
                                 input_shape=self.obs_shape))
            else:
                model.add(Conv2D(c, k, strides=(s, s), activation=a,
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer))
            i += 1
        model.add(Dense(linear_units,
                        activation='relu',
                        kernel_initializer=initializer,
                        bias_initializer=initializer))
        model.add(Dense(self.action_shape[0], activation='tanh',
                        kernel_initializer=initializer,
                        bias_initializer=initializer))

        self.model = model
        self.optimizer = tf.train.AdamOptimizer(actor_lr)
        self.model.compile(self.optimizer)

    def create_target_model(self):
        obs = Input(self.obs_shape)
        self.target_model = clone_model(self.model, obs)

    def sync_target_model(self):
        self.target_model.set_weights(
            self.model.get_weights() * self.tau +
            (1 - self.tau) * self.target_model.get_weights())

    @ex.capture
    def train(self, loss, _run):
        self.t += 1
        _run.log_scalar("actor_loss", loss, self.t)
        self.optimizer.minimize(loss)

    def __call__(self, state):
        return self.model(state)


class Critic(object):
    @ex.capture
    def __init__(self, action_shape, obs_shape, tau):
        self.tau = tau
        self.t = 0

        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.create_model()
        self.create_target_model()

    @ex.capture
    def create_model(self, critic_lr, channels, kernels, strides, activations,
                     linear_units):
        initializer = RandomUniform(-3e-4, 3e-4)
        obs = Input(shape=self.obs_shape)
        action = Input(shape=self.action_shape)

        self.obs = obs
        self.action = action

        i = 0
        x = obs
        for c, k, s, a in zip(channels, kernels, strides, activations):
            x = Conv2D(c, k, strides=(s, s), activation=a,
                       kernel_initializer=initializer,
                       bias_initializer=initializer)(x)

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
        self.model = Model(inputs=[obs, action], outputs=output)

        # self.optimizer = Adam(critic_lr)
        self.optimizer = tf.train.AdamOptimizer(critic_lr)
        self.model.compile(self.optimizer)

    def create_target_model(self):
        obs = Input(shape=self.obs_shape)
        action = Input(shape=self.action_shape)

        self.target_obs = obs
        self.target_action = action
        self.target_model = clone_model(self.model, [obs, action])

    def sync_target_model(self):
        self.target_model.set_weights(
            self.model.get_weights() * self.tau +
            (1 - self.tau) * self.target_model.get_weights())

    def __call__(self, state, action):
        return self.model([state, action])

    @ex.capture
    def train(self, targets, batch, _run):
        states = [exp['state'] for exp in batch]
        actions = [exp['action'] for exp in batch]
        vals = self.model([states, actions])
        loss = tf.reduce_mean(tf.squared_difference(vals, targets))
        self.t += 1
        _run.log_scalar("critic_loss", loss, self.t)
        self.optimizer.minimize(loss)


class ReplayBuffer(object):
    def __init__(self, max_capacity):
        self.buffer = deque(maxlen=max_capacity)

    def append(self, x):
        self.buffer.append(x)

    def __call__(self, state, action, reward, new_state):
        self.append({'state': state, 'action': action,
                     'reward': reward, 'new_state': new_state})

    @ex.capture
    def sample(self, batch_size):
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)


class ImageProcessor(object):
    @ex.capture
    def __init__(self, image, grayscale, resize, img_size, n_images):
        self.grayscale = grayscale
        self.resize = resize
        self.img_size = img_size
        self.images = deque(maxlen=n_images)
        for _ in range(n_images):
            self(image)

    def process_image(self, image):
        if self.resize:
            image = tf.image.resize_images(image, tf.constant([self.img_size, self.img_size]))
        if self.grayscale:
            image = tf.image.rgb_to_grayscale(image)
        return image

    def get_state(self):
        return tf.squeeze(tf.stack(list(self.images), axis=2))

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

    def __call__(self):
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
        return noise


@ex.capture
def compute_targets(batch, actor, critic, gamma):
    # compute the target values
    rewards = np.array([exp['reward'] for exp in batch])
    next_states = np.array([exp['next_state'] for exp in batch])

    targets = rewards + gamma * critic.target_model(next_states, actor.target_model(next_states))
    return targets


def compute_actor_loss(batch, actor, critic):
    states = np.array([exp['state'] for exp in batch])
    actions = actor(states)
    q = critic(states, actions)
    loss = -1 * tf.reduce_sum(q) / len(batch)
    return loss


@ex.capture
def env_loop(env, actor, critic, replay_buffer, noise_generator, train, render, low_dimensional, n_images, replay_start_size):
    R = 0
    done = False
    state = env.reset()
    pixels = env.render(mode='rgb_array', close=not render)
    if not low_dimensional:
        proc = ImageProcessor(pixels)
        state = proc.get_state()
    with tf.Session() as sess:
        while not done:
            import pdb; pdb.set_trace()
            action = actor(tf.stack([state]))
            if train:
                action += noise_generator()

            # step the environment and record new state
            new_state, reward, done, _ = env.step(action)
            pixels = env.render(mode='rgb_array', close=not render)
            if not low_dimensional:
                new_state = proc(pixels)
            R += reward

            # store the experience on the replay buffer
            if train:
                replay_buffer(state, action, reward, new_state)
                state = new_state

            # train if approporiate
            if len(replay_buffer) > replay_start_size and train:
                batch = replay_buffer.sample()
                actor_loss = compute_actor_loss(batch, actor, critic)
                targets = compute_targets(batch, actor, critic)
                critic.train(targets, batch)
                actor.train(actor_loss)
                critic.sync_target_model()
                actor.sync_target_model()
    return R


@ex.automain
def run_experiment(img_size, n_images, replay_buffer_size, eval_n_episodes, eval_loop, _run):
    # start the code
    # create the OpenAI Gym environment
    env = gym.make("Pendulum-v0")
    env.reset()
    obs_shape = (img_size, img_size, n_images)
    action_shape = env.action_space.shape
    critic = Critic(action_shape, obs_shape)
    actor = Actor(action_shape, obs_shape)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    noise_generator = AdditiveOU(action_shape=action_shape)
    eps = 0
    evals = 0
    while True:
        for _ in range(eval_n_episodes):
            eps += 1
            reward = env_loop(env, actor, critic, replay_buffer, noise_generator, True)
            _run.log_scalar("reward", reward, eps)
        R = 0
        for _ in range(eval_loop):
            R += env_loop(env, actor, critic, None, None, False)
        _run.log_scalar("eval_reward", R / float(eval_loop), evals)
        evals += 1
