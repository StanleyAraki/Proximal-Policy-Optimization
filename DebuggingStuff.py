import os
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from utils import plot_learning_curve

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []  # states encounted
        self.probs = []  # log probs
        self.vals = []  # value critic calculates
        self.actions = []  # actions we actually took
        self.rewards = []  # rewards received
        self.dones = []  # terminal flags

        self.batch_size = batch_size

    def generate_batches(self):
        # Have a list of intergesr that correspond to indices of memories, and have batch size chuncks of those memories
        # Shuffle up those indices and take batch size chuncks of those indices
        num_states = len(self.states)
        batch_start = np.arange(0, num_states, self.batch_size)
        indices = np.arange(num_states, dtype=np.int64)
        # shuffle those indices to handle stochatic part of the minibatch of stochastic gradienst ascent
        np.random.shuffle(indices)
        # Takes all possible starting points of batches and goes from those indices to i + batch_size
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.probs.append(probs)
        self.actions.append(action)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims), # Need to manually pass in input
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1) # SoftMax takes care of the fact that we're dealing with probabilities and they sum to 1
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        # Calculating series of probabilities that we will use to get our actual actions
        # We can use that to get the log probabilities for the calculation of the ratio, for the two probabilities of our learning function

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims), # Need to manually pass in inputs
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        ) # handles batch size automatically

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # self.device = T.device('cuda:0' xif T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, num_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip = 0.2, batch_size=64, N=2048, num_epochs=10): # N is horizon, number of steps before update
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.num_epochs = num_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(num_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
    
    def choose_action(self, observation):
        # handle choosing an action that takes an observation of the current state of the environment, converts to torch tensor
        state = T.tensor([observation.flatten()], dtype=T.float) # Need to flatten the observation from (1, 80, 80) to (1, 6400)
        # print("TYPE: ", state.type())
        # state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        state = state.to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        # our learning function
        for _ in range(self.num_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            # Start calculating advantages
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                advantage_time = 0
                for k in range(t, len(reward_arr)-1):
                    # delta_t part is in the paranthesis
                    advantage_time += discount *(reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda # takes care of multiplicative factor (gamma lambda) ^ (T-t+1)delta_T-1
                advantage[t] = advantage_time
            advantage = T.tensor(advantage).to(self.actor.device) 

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch].reshape(5, 6400), dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                # we have the bottom of the numerator now(pi old)

                # we now need pi new
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad() # zero the gradient
                self.critic.optimizer.zero_grad() # zero the gradient
                total_loss.backward() # backpropagate total loss
                self.actor.optimizer.step() # upstep optimizer
                self.critic.optimizer.step() # upstep optimizer
        self.memory.clear_memory()

if __name__ == '__main__':
    env = gym.make('ALE/Pong-v5')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    print(env.observation_space.shape) # (210, 160, 3)

    env2 = gym.make('CartPole-v1')
    print(env2.observation_space.shape)

    # pre-process image 
    def prepro(image):
        image = image[35:195]  # crop
        image = image[::2, ::2, 0]  # downsample by factor of 2
        image[image == 144] = 0  # erase background (background type 1)
        image[image == 109] = 0  # erase background (background type 2)
        image[image != 0] = 1  # everything else (paddles, ball) just set to 1
        return np.reshape(image, (1,80,80))
    
    raw_image = env.reset()
    preprocessed_image = prepro(raw_image) #(1, 80, 80)

    # plt.imshow(preprocessed_image.squeeze(0))
    flattened = preprocessed_image.flatten()
    # plt.show()

    # Reason for bug: when the input size is greater than (1, ), the orderings of the inputs shift. 
    # Will fix bug later.
    agent = Agent(num_actions = env.action_space.n, batch_size = 5, alpha = 0.0003, num_epochs = 4, input_dims = flattened.shape)
    n_games = 300

    figure_file = 'plots/Pong.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        # observation = prepro(observation)
        done = False
        score = 0
        while not done:
            observation = prepro(observation) # need to preprocess each time
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0: # if true, it's time to perform learning function
                # print("Performs learning")
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score: # if best score found
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]

# if __name__ == '__main__':
#     # def prepro(image):
#     #     image = image[35:195]  # crop
#     #     image = image[::2, ::2, 0]  # downsample by factor of 2
#     #     image[image == 144] = 0  # erase background (background type 1)
#     #     image[image == 109] = 0  # erase background (background type 2)
#     #     image[image != 0] = 1  # everything else (paddles, ball) just set to 1
#     #     return np.reshape(image, (1,80,80))
    
#     env = gym.make('CartPole-v0')
#     N = 20
#     batch_size = 5
#     n_epochs = 4
#     alpha = 0.0003
#     # raw_image = env.reset()
#     # preprocessed_image = prepro(raw_image) #(1, 80, 80)
#     # flattened = preprocessed_image.flatten()
#     agent = Agent(num_actions=env.action_space.n, batch_size=batch_size, 
#                     alpha=alpha, num_epochs=n_epochs, 
#                     input_dims=env.observation_space.shape) # does work on more advanced environments but requires more fine-tuning
#     n_games = 300

#     figure_file = 'plots/Pong.png'

#     best_score = env.reward_range[0]
#     score_history = []

#     learn_iters = 0
#     avg_score = 0
#     n_steps = 0

#     for i in range(n_games):
#         observation = env.reset()
#         # observation = prepro(observation)
#         done = False
#         score = 0
#         while not done:
#             action, prob, val = agent.choose_action(observation)
#             observation_, reward, done, info = env.step(action)
#             n_steps += 1
#             score += reward
#             agent.remember(observation, action, prob, val, reward, done)
#             if n_steps % N == 0: # if true, it's time to perform learning function
#                 agent.learn()
#                 learn_iters += 1
#             observation = observation_
#         score_history.append(score)
#         avg_score = np.mean(score_history[-100:])

#         if avg_score > best_score: # if best score found
#             best_score = avg_score
#             agent.save_models()

#         print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
#                 'time_steps', n_steps, 'learning_steps', learn_iters)
#     x = [i+1 for i in range(len(score_history))]
#     # print("Final Score: ", score_history[-1])
    
#     plot_learning_curve(x, score_history, figure_file, "Training Episodes", "Average Scores")

