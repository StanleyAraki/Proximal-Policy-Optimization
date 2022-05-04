# Import necessary libraries
from email import policy
import multiprocessing
import multiprocessing.connection 
from typing import Dict, List

import gym 
import numpy as np
import torch
from labml import monit, tracker, logger, experiment 
from torch import log_, nn
from torch import optim
from torch.distributions import Categorical 
from torch.nn import functional as F
import moviepy.editor as mpy

# globally set up device environment
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

obs_for_animation = []
rawframes = []

class Game:
    '''
        Apply same action on four frames and get last frame
        Convert observation frames to gray and scale it to (80, 80)
        Stack four frames of the last four actions
        Add episode info (total reward for entire episode) for monitoring
        Restrict episode to single life so we can reset after every single life
    '''

    def __init__(self, seed): # don't need seed in our implementation
        self.env = gym.make('BreakoutDeterministic-v4')
        self.env.seed(seed)
        self.observation_4 = np.zeros((4, 80, 80))
        self.rewards = []
        self.lives = 0
    
    def step(self, action):
        # global rawframes
        # executes action and returns tuple of (observation, reward, done, episode_info) (these are what is returned in gym.step() as well)
        reward = 0
        done = False


        for i in range(4): # for the four steps available, run the model in OpenAI Gym environment
            observation, r, done, info = self.env.step(action)
            reward += r
            lives = self.env.unwrapped.ale.lives() 

            if lives < self.lives: # reset if lives is lost
                done = True 
                break
        rawframes.append(observation)
        observation = self.preprocess(observation) # transform the observation from (210, 160, 3) to (80, 80)
        self.rewards.append(reward)

        if done:
            # if finished, set episode info and reset
            episode_info = {'reward': sum(self.rewards), 'length': len(self.rewards)}
            self.reset() # call resetting function to reset environment
            # for frame in rawframes:
            #     obs_for_animation.append(frame)
            # rawframes = []
            
        else:
            # if not finished, episode info is not set
            episode_info = None

            # get the max of the last two frame observation and push it to the stack of 4 frames
            self.observation_4 = np.roll(self.observation_4, shift=-1, axis=0) # rolls array element along 0th axis by -1
            self.observation_4[-1] = observation 
        
        return self.observation_4, reward, done, episode_info
        
    def reset(self):
        observation = self.preprocess(self.env.reset()) # get preprocessed resetted observation

        for i in range(4):
            self.observation_4[i] = observation # reset all 4
        self.rewards = []

        self.lives = self.env.unwrapped.ale.lives() # get lives of current state

        return self.observation_4

    # Preprocess input to (80, 80)
    @staticmethod
    def preprocess(image): # instead of using cv2 use generic preprocessing
        image = image[35:195]  # crop
        image = image[::2, ::2, 0]  # downsample by factor of 2
        image[image == 144] = 0  # erase background (background type 1)
        image[image == 109] = 0  # erase background (background type 2)
        image[image != 0] = 1  # everything else (paddles, ball) just set to 1
        return np.reshape(image, (80, 80))

def worker_process(remote, seed): # Each worker runs this 
    '''
        This function will execute the actions that each worker takes
    '''
    game = Game(seed)

    while True: # wait for instructions from connection and execute until closed
        cmd, data = remote.recv() # get command and data
        if cmd == 'step': # if step, do step
            remote.send(game.step(data))
        elif cmd == 'reset':
            remote.send(game.reset())
        else: # if not step or reset, want to close remote
            remote.close()
            break

class Worker:
    '''
        Creates new worker and runs in separate process
        multiprocessing is a package that supports multi-processing similar to thread molecule
    '''
    
    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed)) # target is a callable object(so a function) to be invoked by run() method, args is argument tuple for target invocation
        self.process.start() 

class Model(nn.Module):
    '''
        The actual model that we will be using
        In this model we will be implementing the Proximal Policy Optimization algorithm with the help of LabML library
    '''

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size=8, stride=4) # in_channels = 1 b/c grayscale. 80 x 80 becomes 19 x 19
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size = 3, stride = 2) # 19 x 19 becomes 9 x 9
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1) # 9 x 9 becomes 7 x 7
        self.linear = nn.Linear(in_features = 7 * 7 * 64, out_features=512) # out features should be 512 features

        self.pi_logits = nn.Linear(in_features=512, out_features=4)
        self.value = nn.Linear(in_features=512, out_features=1) # layer to get value function

    def forward(self, observation):
        '''
            Forward step function called during step
        '''
        out = F.relu(self.conv1(observation))
        out = F.relu(self.conv2(out)) # might add normalizing layyer after
        out = F.relu(self.conv3(out))
        out = out.reshape((-1, 7 * 7 * 64)) # before we pass it through the linear layer, need to reshape

        out = F.relu(self.linear(out)) # pass it through the linear layer

        pi = Categorical(logits=self.pi_logits(out)) # use Categorical so we can do a .sample() to get values
        value = self.value(out).reshape(-1)

        return pi, value
    
def observation_to_torch(observation):
    '''
        Want to change observation to a tensor
    '''
    return torch.tensor(observation, dtype=torch.float32, device=device) / 255

def saveanimation(rawframes, filename):
    '''
        Saves a sequence of frames as an animation
        The filename must include an appropriate video extension
    '''
    clip = mpy.ImageSequenceClip(rawframes, fps=60)
    clip.write_videofile(filename)

class Main:

    def __init__(self):
        
        # set up variables to use in advantage calculation
        self.gamma = 0.99 
        self.lamda = 0.95 

        self.updates = 1200 # number of updates/iterations
        self.epochs = 4
        self.num_workers = 10 # number of worker processes
        self.worker_steps = 128 # number of steps to run on each process for single update
        self.num_mini_batch = 5
        self.batch_size = self.num_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.num_mini_batch
        # need to make sure batch size % number of mini batch = 0

        self.best_score = float('-inf')

        # Initialize workers
        self.workers = [Worker(47 + i) for i in range(self.num_workers)] # create list of workers

        self.observation = np.zeros((self.num_workers, 4, 80, 80), dtype=np.uint8)
        for worker in self.workers:
            # Reset environment of all workers
            worker.child.send(('reset', None))
        for i, worker in enumerate(self.workers):
            # initialize observations from all workers
            self.observation[i] = worker.child.recv()
        
        self.model = Model().to(device)
        experiment.add_pytorch_models({'base': self.model})
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)

    def sample(self):
        global obs_for_animation
        global rawframes

        # Initialize arrays to use
        rewards = np.zeros((self.num_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.num_workers, self.worker_steps), dtype=np.int32)
        done = np.zeros((self.num_workers, self.worker_steps), dtype=bool)
        observation = np.zeros((self.num_workers, self.worker_steps, 4, 80, 80), dtype=np.uint8)
        log_pis = np.zeros((self.num_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.num_workers, self.worker_steps), dtype=np.float32)

        avg_reward = 0
        count = 0
        for step in range(self.worker_steps):
            with torch.no_grad():
                observation[:, step] = self.observation # keeps track of last observation from each worker, which is input to model for next action

                old_pi, value = self.model(observation_to_torch(self.observation))
                values[:, step] = value.cpu().numpy()
                action = old_pi.sample() # sample actions from result of forward step in Categorical()
                actions[:, step] = action.cpu().numpy()
                log_pis[:, step] = old_pi.log_prob(action).cpu().numpy()
            
            # run sampled action on each worker
            for w, worker in enumerate(self.workers):
                worker.child.send(('step', actions[w, step]))

            for w, worker in enumerate(self.workers):
                self.observation[w], rewards[w, step], done[w, step], info = worker.child.recv() # receive results after actions
                # for frame in rawframes:
                #     obs_for_animation.append(frame)
                # rawframes = []
                # # obs_for_animation.append(self.observation[0]) # add to frames
                
                if info: # if info not none(so episode is finished), get episode info
                    tracker.add('reward', info['reward'])
                    avg_reward += info['reward']
                    count += 1
                    tracker.add('length', info['length'])
        advantages = self.get_advantage(done, rewards, values) # get the advantage
        samples = {
            'observation': observation,
            'actions': actions,
            'values': values,
            'log_pis':log_pis,
            'advantages': advantages
        }

        # print("AVG REWARD: ", avg_reward/count)
        avg_reward = avg_reward / count # get the average rewarad

        if avg_reward > self.best_score:
            # if average of policy rewards is better than what we've seen, update best and create checkpoint
            self.best_score = avg_reward
            print("\n... Saving Model ...")
            print("\nBEST SCORE: ", self.best_score)
            experiment.save_checkpoint()
            print("\nExperiment UUID: ", experiment.get_uuid())

        # samples are currently in [workers, time] table so need to flatten them
        samples_flat = {}
        for key, value in samples.items():
            value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            if key == 'observation':
                samples_flat[key] = observation_to_torch(value)
            else:
                samples_flat[key] = torch.tensor(value, device=device)
        
        return samples_flat

    def get_advantage(self, done, rewards, values):
        '''
            Instead of getting the usual advantage, we take a weighted average of the advantages
            A_t = delta_t + (gamma*delta_(t+1)) + (gamma^2 * delta_(t+1)) + ... is usual advantage
            Generalized Advantage Estimation(GAE) Advantage is 
                A_t = delta_t + gamma*lambda*A_(t+1)
            We can think of lambda as a smoothing parameter that helps us lower variance
            gamma is the output of the critic network
        '''        

        advantages = np.zeros((self.num_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0

        _, last_value = self.model(observation_to_torch(self.observation))
        last_value = last_value.cpu().data.numpy() # V(s_(t+1))

        for step in reversed(range(self.worker_steps)):
            mask = 1-done[:, step]
            last_value *= mask
            last_advantage *= mask

            delta = rewards[:, step] + self.gamma * last_value - values[:, step] # get the delta value
            last_advantage = delta + self.gamma * self.lamda * last_advantage # calculate advantage A_t = delta_t + (gamma * lamda) * A_(t+1)
            advantages[:, step] = last_advantage # update last advantage
            last_value = values[:, step]
        
        return advantages
    
    def train(self, samples, learning_rate, clip_range):
        '''
            Train models based on samples

        '''
        for _ in range(self.epochs):
            indices = torch.randperm(self.batch_size)

            # For each minibatch
            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indices = indices[start:end]
                mini_batch = {}

                for key, value in samples.items():
                    mini_batch[key] = value[mini_batch_indices]
                
                loss = self.get_loss(clip_range = clip_range, samples=mini_batch) # train

                # Get gradients
                for pg in self.optimizer.param_groups:
                    pg['lr'] = learning_rate
                self.optimizer.zero_grad() # zero the gradient
                loss.backward() # backpropagate
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 0.5)
                self.optimizer.step() # step gradient for optimizer

    def normalize(self, advantage):
        # normalize the advantage function
        return (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
    def get_loss(self, clip_range, samples):
        '''
            This function will calculate the PPO Loss
            Our objective is to maximize the policy reward
            max J(pi_theta) = Expectation (sum of gamma^t * r_t) t => inf
            r = reward, pi = policy, t = trajectory, gamma = discount [0, 1]
            Eventually equation becomes
            J(pi_theta) - J(pi * theta_old) = 1/(1-gamma) * Expectation[ratio of new policy to old] = L^(CPI)
        '''

        sampled_return = samples['values'] + samples['advantages']
        sampled_normalized_advantage = self.normalize(samples['advantages'])

        pi, value = self.model(samples['observation']) # sampled obs fed into model

        # POLICY
        # - log pi_theta(a_t | s_t) are actions sampled from pi_old
        log_pi = pi.log_prob(samples['actions'])
        ratio = torch.exp(log_pi - samples['log_pis']) # ratio r_t(theta) = pi_new/pi_old

        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max = 1.0 + clip_range) # clips all elements
        policy_reward = torch.min(ratio * sampled_normalized_advantage, clipped_ratio * sampled_normalized_advantage) # this is the clipping part. 
        # Take the minimum according to equation, so update isn't too large
        policy_reward = policy_reward.mean()

        # Entropy Bonus
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean() 

        # Value
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value-sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean() / 2 # Making sure that v_theta_new doesn't deviate too much from v_theta_old

        # Want to maximize clipped so take negative of it as the loss
        loss = -(policy_reward - vf_loss / 2 + entropy_bonus / 100)

        approx_kl_divergence = ((samples['log_pis'] - log_pi) ** 2).mean() / 2
        clip_fraction = (abs((ratio-1.0)) > clip_range).to(torch.float).mean()

        tracker.add({
            'policy_reward' : policy_reward,
            'vf_loss' : vf_loss,
            'entropy_bonus' : entropy_bonus,
            'kl_div' : approx_kl_divergence,
            'clip_fraction' : clip_fraction
        })

        return loss
    
    def run_training_loop(self):
        # Run the training loop

        tracker.set_queue('reward', 100, True)
        tracker.set_queue('length', 100, True)

        for update in monit.loop(self.updates):
            # global obs_for_animation
            progress = update / self.updates

            # Decrease the learning rate
            learning_rate = 0.00025 * (1 - progress)
            clip_range = (1 - progress) / 10

            # Sample with current policy
            samples = self.sample()

            # Train model
            self.train(samples, learning_rate, clip_range)
            
            # Write summary info to writer and log to screen
            tracker.save()
            
            # print("\nUUID OF EXPERIMENT: ", experiment.get_uuid()) # need to know UUID for loading checkpoints
            # if ((update+1) % 2 == 0): # save video every 50 iterations
            #     saveanimation(obs_for_animation, "{}_episode{}.mp4".format("video/Breakout", update))
            
            # obs_for_animation = [] # reset animation array
            
            if (update + 1) % 1000 == 0:
                logger.log()
    
    def destroy(self):
        # Stop workers

        for worker in self.workers:
            worker.child.send(('close', None))

if __name__ == '__main__':
    # Run experiment
    experiment.create(uuid="breakout_PPO_389_2", name='Breakout_Training')
    m = Main()
    # Load Experiment from past experiment
    print("... Loading Model ...")
    experiment.load(run_uuid="breakout_PPO_389_1", checkpoint=561) 
    experiment.start()
    m.run_training_loop()
    m.destroy()
    print("\nFinished Training Iteration")