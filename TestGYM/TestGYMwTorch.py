import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


env = gym.make("CartPole-v1", render_mode="rgb_array")

# gym compatibility: unwrap TimeLimit
if hasattr(env, '_max_episode_steps'):
    env = env.env

env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape

plt.imshow(env.render())


import torch
import torch.nn as nn
import torch.nn.functional as F


model = nn.Sequential()
model.add_module('l1', nn.Linear(4, 4))
model.add_module('l2', nn.Linear(4, 2))
model.add_module('l3', nn.Softmax())
#print("Weight shapes:", [w.shape for w in model.parameters()])


def predict_probs(states):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :returns: numpy array of shape [batch, n_actions]
    """
    # convert states, compute logits, use softmax to get probability

    states = torch.tensor(states)
    prob_weights = model(states)
    return prob_weights.detach().numpy()



def generate_session(env, t_max=1000):
    """
    Play a full session with REINFORCE agent.
    Returns sequences of states, actions, and rewards.
    """
    # arrays to record session
    states, actions, rewards = [], [], []

    s = env.reset()[0]

    for t in range(t_max):
        # action probabilities array aka pi(a|s)
        action_probs = predict_probs(np.array([s]))
        #print(action_probs[0])
        # Sample action with given probabilities.
        a = np.random.choice(range(2), p=action_probs[0])
        # print(a)

        new_s, r, terminated, truncated, info = env.step(a)
        # record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if terminated or truncated:
            #print('terminated or truncated')
            break

    return states, actions, rewards

def get_cumulative_rewards(rewards,  gamma=0.99  ):
    T = len(rewards)
    G=np.zeros(T)
    G[T-1]=rewards[T-1]
    for t in range(T-1):
        G[T-2-t] = rewards[T-2-t] +gamma * G[T-1-t]
    return G


states, actions, rewards = generate_session(env)
get_cumulative_rewards(rewards)

# Your code: define optimizers
optimizer = torch.optim.Adam(model.parameters(), 1e-3,maximize=True)


def train_on_session(states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    Please use Adam optimizer with default parameters.
    """

    # cast everything into torch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    # predict logits, probas and log-probas using an agent.
    logits = model(states)
    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)

    assert all(isinstance(v, torch.Tensor) for v in [logits, probs, log_probs]), \
        "please use compute using torch tensors and don't use predict_probs function"

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    log_probs_for_actions = torch.sum(
        log_probs * F.one_hot(actions, env.action_space.n), dim=1)
    #print(log_probs_for_actions)
    #print(cumulative_returns)
    # Compute loss here. Don't forgen entropy regularization with `entropy_coef`
    #entropy = -torch.dot(torch.exp(log_probs_for_actions),log_probs_for_actions)
    loss = torch.dot(cumulative_returns,log_probs_for_actions) #+ entropy_coef*entropy
    # Gradient descent step
    optimizer.zero_grad()  # clear gradients
    loss.backward()  # add new gradients
    optimizer.step()

    # technical: return session rewards to print them later
    return np.sum(rewards)


for i in tqdm(range(100)):
    rewards = [train_on_session(*generate_session(env)) for _ in range(1000)]  # generate new sessions

    print("mean reward: %.3f" % (np.mean(rewards)))

    if np.mean(rewards) > 600:
        print("You Win!")  # but you can train even further
        break

env = gym.make("CartPole-v1", render_mode='human')
EPISODES_test=10
for episode in range(EPISODES_test):
      observation = env.reset()
      observation = observation[0]

      # Execute the environment using the learned policy
      while True:
          env.render()  # Render the environment

          # Choose an action based on the learned policy
          action_probs = predict_probs(np.array([observation]))
          #print(action_probs)
          # Sample action with given probabilities.
          a = np.argmax(action_probs[0])
          # Perform the action in the environment
          observation, reward, done, _, _ = env.step(a)

          if done:
              break  # Episode is finished