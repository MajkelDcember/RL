import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class PolicyGradient:

   def __init__(self, n_x, n_y, learning_rate=0.01, reward_decay=0.99):
      self.n_x = n_x
      self.n_y = n_y
      self.lr = learning_rate
      self.gamma = reward_decay
      self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
      self.build_network()
      self.cost_history = []
      self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)

   def store_transition(self, s, a, r):
      self.episode_observations.append(s)
      self.episode_rewards.append(r)
      action = np.zeros(self.n_y)
      action[a] = 1
      self.episode_actions.append(action)

   def choose_action(self, observation):
      #print(observation[0])
      observation = np.expand_dims(observation, axis=0)
      #print(observation)
      prob_weights = self.model(observation).numpy()
      #print(prob_weights[0])
      action = np.random.choice(range(self.n_y), p=prob_weights[0])
      #print(action)
      return action

   def build_network(self):
      self.model = tf.keras.Sequential([
         tf.keras.layers.Input(8),
         tf.keras.layers.Dense(10, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
         tf.keras.layers.Dense(10, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
         tf.keras.layers.Dense(4, activation='softmax', kernel_initializer='glorot_uniform',
                               bias_initializer='zeros')
      ])

   def discount_and_norm_rewards(self):
      discounted_episode_rewards = np.zeros_like(self.episode_rewards)
      cumulative = 0
      for t in reversed(range(len(self.episode_rewards))):
         cumulative = cumulative * self.gamma + self.episode_rewards[t]
         discounted_episode_rewards[t] = cumulative
      #discounted_episode_rewards -= np.mean(discounted_episode_rewards)
      # discounted_episode_rewards /= np.std(discounted_episode_rewards)
      return discounted_episode_rewards

   def learn(self):
      discounted_episode_rewards_norm = self.discount_and_norm_rewards()
      observations = np.vstack(self.episode_observations)
      actions = np.vstack(self.episode_actions)
      with tf.GradientTape() as tape:
         logits = self.model(observations)
         neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=actions)
         loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_norm)
      gradients = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
      self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
      return discounted_episode_rewards_norm



env = gym.make("LunarLander-v2", render_mode='rgb_array')

env = env.unwrapped
observation, info = env.reset(seed=42)


RENDER_ENV = False
EPISODES = 1000000
rewards = []
RENDER_REWARD_MIN = 5000

PG = PolicyGradient(
    n_x = env.observation_space.shape[0],
    n_y = env.action_space.n,
    learning_rate=0.00001,
    reward_decay=0.99,
)

for episode in range(EPISODES):

   # get the state
   observation = env.reset()
   observation = observation[0]
   episode_reward = 0
   while True:

      if RENDER_ENV: env.render()

      # choose an action based on the state
      action = PG.choose_action(observation)

      # perform action in the environment and move to next state and receive reward
      observation_, reward, done, _, _ = env.step(action)
      #print(observation_)
      # store the transition information
      PG.store_transition(observation, action, reward)

      # sum the rewards obtained in each episode
      episode_rewards_sum = sum(PG.episode_rewards)

      # if the reward is less than -259 then terminate the episode
      if episode_rewards_sum < -250:
         done = True

      if done:
         episode_rewards_sum = sum(PG.episode_rewards)
         rewards.append(episode_rewards_sum)
         max_reward_so_far = np.amax(rewards)

         print("Episode: ", episode)
         print("Reward: ", episode_rewards_sum)
         print("Max reward so far: ", max_reward_so_far)

         # train the network
         discounted_episode_rewards_norm = PG.learn()

         if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = False

         break

      # update the next state as current state
      observation = observation_
# Assuming you have trained the PolicyGradient algorithm using the provided code

# Load the trained policy network
trained_policy_network = PG.model
env = gym.make("LunarLander-v2", render_mode='human')
# Reset the environment to start a new episode
EPISODES_test=10000
for episode in range(EPISODES_test):
      observation = env.reset()
      observation = observation[0]

      # Execute the environment using the learned policy
      while True:
          env.render()  # Render the environment

          # Choose an action based on the learned policy
          action = np.argmax(trained_policy_network.predict(np.expand_dims(observation, axis=0)))

          # Perform the action in the environment
          observation, reward, done, _, _ = env.step(action)

          if done:
              break  # Episode is finished