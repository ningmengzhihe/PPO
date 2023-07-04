"""
修改自
https://zhuanlan.zhihu.com/p/538486008#

gym 0.26.2

"""

import argparse
from collections import namedtuple

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt

# Parameters
parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('Pendulum-v1').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.shape[0]
torch.manual_seed(args.seed)
env.reset(seed=args.seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 1)
        self.sigma_head = nn.Linear(100, 1) # 100

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 8)
        self.state_value = nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 1000, 32
    method = 0  # 0表示KL penalty，1表示Clip
    kl_penalty_lambda = 0.5
    kl_target = 0.01

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().float()
        self.critic_net = Critic().float()
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=1e-4)  # 原来是3e-4
        # if not os.path.exists('../param'):
        #     os.makedirs('../param/net_param')
        #     os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.actor_net(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2, 2)
        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    # def save_param(self):
    #     torch.save(self.anet.state_dict(), 'param/ppo_anet_params.pkl')
    #     torch.save(self.cnet.state_dict(), 'param/ppo_cnet_params.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_state = torch.tensor(np.array([t.next_state for t in self.buffer]), dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        with torch.no_grad():
            target_v = reward + args.gamma * self.critic_net(next_state)

        advantage = (target_v - self.critic_net(state)).detach()
        if self.method == 1:
            for _ in range(self.ppo_epoch):  # iteration ppo_epoch
                # 每次从buffer中随机选出batch_size数量的transition用来更新actor网络
                for index in BatchSampler(
                        SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                    # epoch iteration, PPO core!!!
                    mu, sigma = self.actor_net(state[index])  # !!!这里的actor_net是new policy呀
                    n = Normal(mu, sigma)
                    action_log_prob = n.log_prob(action[index])
                    ratio = torch.exp(action_log_prob - old_action_log_prob[index])

                    L1 = ratio * advantage[index]
                    L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]
                    action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
                    self.actor_optimizer.zero_grad()
                    action_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)  # clip_grad_norm梯度剪裁，只解决梯度爆炸问题，不解决梯度消失问题
                    self.actor_optimizer.step()

                    value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                    self.critic_net_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)  # clip_grad_norm梯度剪裁，只解决梯度爆炸问题，不解决梯度消失问题
                    self.critic_net_optimizer.step()

            del self.buffer[:]
        else:  # self.method == 0 'kl_penalty'
            for _ in range(self.ppo_epoch):  # iteration ppo_epoch
                # 每次从buffer中随机选出batch_size数量的transition用来更新actor网络
                for index in BatchSampler(
                        SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                    # epoch iteration, PPO core!!!
                    mu, sigma = self.actor_net(state[index])
                    n = Normal(mu, sigma)
                    action_log_prob = n.log_prob(action[index]) # !!!划重点，新策略动作值的log_prob，是新策略得到的分布上找到action对应的log_prob值

                    new_action_prob = torch.exp(action_log_prob)
                    old_action_prob = torch.exp(old_action_log_prob[index])

                    # KL散度
                    kl = nn.KLDivLoss(reduction='batchmean')(old_action_prob, new_action_prob)
                    # 计算loss
                    ratio = new_action_prob / old_action_prob
                    actor_loss = -torch.mean(ratio * advantage[index] - self.kl_penalty_lambda * kl)
                    # 梯度下降
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)  # 梯度剪裁，只解决梯度爆炸问题，不解决梯度消失问题
                    self.actor_optimizer.step()
                    if kl > 4 * self.kl_target:
                        # this in google's paper
                        break

                    value_loss = nn.MSELoss(reduction='mean')(self.critic_net(state[index]), target_v[index])

                    #value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                    self.critic_net_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(),
                                             self.max_grad_norm)  # clip_grad_norm梯度剪裁，只解决梯度爆炸问题，不解决梯度消失问题
                    self.critic_net_optimizer.step()
            if kl < self.kl_target / 1.5:
                # 散度较小，需要弱化惩罚力度
                # adaptive lambda, this is in OpenAI's paper
                self.kl_penalty_lambda /= 2
            elif kl > self.kl_target * 1.5:
                # 散度较大，需要增强惩罚力度
                self.kl_penalty_lambda *= 2
            # sometimes explode, this clipping is my solution
            self.kl_penalty_lambda = np.clip(self.kl_penalty_lambda, 1e-4, 10)
            print(self.kl_penalty_lambda)

def main():
    agent = PPO()

    training_records = []
    running_reward = -1000

    for i_epoch in range(1000): # 1000
        score = 0
        state, info = env.reset()
        if args.render:
            env.render()
        for t in range(200):
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step([action])
            trans = Transition(state, action, (reward + 8) / 8, action_log_prob, next_state)
            if args.render:
                env.render()
            if agent.store_transition(trans):
                # buffer满了触发更新
                agent.update()
            score += reward
            state = next_state

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainRecord(i_epoch, running_reward))
        if i_epoch % 10 == 0:
            print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))

        # if running_reward > -200:
        #     print("Solved! Moving average score is now {}!".format(running_reward))
        #     env.close()
        #     # agent.save_param()
        #     break
    plt.plot([item[1] for item in training_records])
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()


if __name__ == '__main__':
    main()
