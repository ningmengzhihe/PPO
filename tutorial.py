"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

pytorch
gym 0.26.2
连续动作


"""


import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy

from torch.distributions import Normal


GAMMA = 0.9  # 折扣率
EP_MAX = 1000  # episode循环次数 # 默认是1000
EP_LEN = 200  # 一个回合规定的长度 # 默认是200
A_LR = 0.0001 # actor的学习率 默认是0.0001
C_LR = 0.0002  # critic的学习率 默认是0.0002
BATCH = 32 # 缓冲池长度
A_UPDATE_STEPS = 10  # 在多少步数之后更新actor
C_UPDATE_STEPS = 10  # 在多少步数之后更新critic
S_DIM, A_DIM = 3, 1  # state维度是3， action维度是1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty # 0.5
    dict(name='clip', epsilon=0.2)  # clip
][1]  # choose the method for optimization
# METHOD[0]是Adaptive KL penalty Coefficient
# METHOD[1]是Clipped Surrogate Objective
# 结果证明，clip的这个方法更好


# class Actor(nn.Module):
#     def __init__(self,
#                  n_features,
#                  n_neuron):
#         super(Actor, self).__init__()
#         self.fc = nn.Linear(n_features, n_neuron)
#         self.mu_head = nn.Linear(n_neuron, 1)
#         self.sigma_head = nn.Linear(n_neuron, 1)
#
#     def forward(self, x):
#         x = F.tanh(self.fc(x))
#         mu = 2.0 * F.tanh(self.mu_head(x))
#         sigma = F.softplus(self.sigma_head(x))
#
#         return mu, sigma

class Actor(nn.Module):
    """
    神经网络结构
    # 全连接1
    # 全连接2
    # ReLU
    网络输出是动作的mu和sigma
    """
    def __init__(self,
                 n_features,
                 n_neuron):
        super(Actor, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features=n_features,
                      out_features=n_neuron,
                      bias=True),
            nn.ReLU()
        )
        self.mu = nn.Sequential(
            nn.Linear(in_features=n_neuron,
                      out_features=1,
                      bias=True),
            nn.Tanh()
        )
        self.sigma = nn.Sequential(
            nn.Linear(in_features=n_neuron,
                      out_features=1,
                      bias=True),
            nn.Softplus()
        )
        # self.net = nn.Sequential(
        #     nn.Linear(in_features=n_features,
        #               out_features=n_neuron,
        #               bias=True),
        #     nn.Linear(in_features=n_neuron,
        #               out_features=1,
        #               bias=True)
        # )
    def forward(self, x):
        y = self.linear(x)
        mu = 2 * self.mu(y)
        sigma = self.sigma(y)
        return mu, sigma

    # def forward(self, x):
    #     y = self.net(x)
    #     mu = 2 * torch.tanh(y)
    #     sigma = nn.Softplus()(y)
    #     return mu, sigma
    # def forward(self, x):
    #     y = self.net(x)
    #     if len(y.shape) == 1:
    #         mu = 2 * torch.tanh(y[0])
    #         sigma = nn.Softplus()(y[1])
    #     else:
    #         mu = 2 * torch.tanh(y[:, 0])
    #         sigma = nn.Softplus()(y[:, 1], dim=0)
    #         mu = mu.reshape([mu.shape[0], 1])
    #         sigma = sigma.reshape([sigma.shape[0], 1])
    #     return mu, sigma


class Critic(nn.Module):
    """
    神经网络结构
    # 全连接1
    # 全连接2
    # ReLU
    输出是状态价值
    """
    def __init__(self,
                 n_features,
                 n_neuron):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_features,
                      out_features=n_neuron,
                      bias=True),
            nn.ReLU(),
            nn.Linear(in_features=n_neuron,
                      out_features=1,
                      bias=True),
        )

    def forward(self, x):
        return self.net(x)
# class Critic(nn.Module):
#     def __init__(self,
#                  n_features,
#                  n_neuron):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(3, 100)
#         self.fc2 = nn.Linear(100, 8)
#         self.state_value = nn.Linear(8, 1)
#
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         value = self.state_value(x)
#         return value

class PPO(object):

    def __init__(self,
                 n_features,
                 n_neuron,
                 actor_learning_rate,
                 critic_learning_rate,
                 max_grad_norm=0.5  # 梯度剪裁参数
                 ):
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.actor_old = Actor(n_features, n_neuron)
        self.actor = Actor(n_features, n_neuron)
        self.critic = Critic(n_features, n_neuron)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(),
                                          lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(),
                                                 lr=self.critic_lr)
        self.max_grad_norm = max_grad_norm  # 梯度剪裁参数

    def update(self, s, a, r, log_old, br_next_state):
        """

        :param s: np.array(buffer_s)
        :param a: np.array(buffer_a)
        :param r: np.array(buffer_r)
        :param log_old: np.array(buffer_log_old)
        :param next_state: np.array(buffer_next_state)
        :return: update actor net and critic net
        """
        self.actor_old.load_state_dict(self.actor.state_dict())
        # 从buffer中取出state, action, reward, old_action_log_prob, next_state放在tensor上
        state = torch.FloatTensor(s)
        action = torch.FloatTensor(a)
        discounted_r = torch.FloatTensor(r) # discounted_r是target_v
        #old_action_log_prob = torch.FloatTensor(log_old)
        next_state = torch.FloatTensor(br_next_state)

        mu_old, sigma_old = self.actor_old(state)
        dist_old = Normal(mu_old, sigma_old)
        old_action_log_prob = dist_old.log_prob(action).detach()
        # # 优势函数advantage，也是td_error
        # with torch.no_grad():
        #     target_v = discounted_r + GAMMA * self.critic(next_state)

        target_v = discounted_r
        advantage = (target_v - self.critic(state)).detach()

        #advantage = (advantage - advantage.mean()) / (advantage.std()+1e-6)  # sometimes helpful by movan


        # update actor net，METHOD[0]是KL penalty，METHOD[1]是clip
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                # compute new_action_log_prob
                mu, sigma = self.actor(state)
                dist = Normal(mu, sigma)
                new_action_log_prob = dist.log_prob(action)  # !!!划重点，新策略动作值的log_prob，是新策略得到的分布上找到action对应的log_prob值

                new_action_prob = torch.exp(new_action_log_prob)
                old_action_prob = torch.exp(old_action_log_prob)

                # KL散度
                kl = nn.KLDivLoss()(old_action_prob, new_action_prob)
                # 计算loss
                ratio = new_action_prob / old_action_prob
                actor_loss = -torch.mean(ratio * advantage - METHOD['lam'] * kl)
                # 梯度下降
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)  # 梯度剪裁，只解决梯度爆炸问题，不解决梯度消失问题
                self.actor_optimizer.step()
                if kl > 4*METHOD['kl_target']:
                    # this in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:
                # 散度较小，需要弱化惩罚力度
                # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                # 散度较大，需要增强惩罚力度
                METHOD['lam'] *= 2
            # sometimes explode, this clipping is my solution
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)
        else:
            # clipping method, find this is better (OpenAI's paper)
            # update actor net
            for _ in range(A_UPDATE_STEPS):
                ## update step as follows:
                # compute new_action_log_prob
                mu, sigma = self.actor(state)
                n = Normal(mu, sigma)
                new_action_log_prob = n.log_prob(action)  # !!!划重点，新策略动作值的log_prob，是新策略得到的分布上找到action对应的log_prob值

                # ratio = new_action_prob / old_action_prob
                ratio = torch.exp(new_action_log_prob - old_action_log_prob)

                # L1 = ratio * td_error, td_error也叫作advatange
                L1 = ratio * advantage

                # L2 = clip(ratio, 1-epsilon, 1+epsilon) * td_error
                L2 = torch.clamp(ratio, 1-METHOD['epsilon'], 1+METHOD['epsilon']) * advantage

                # loss_actor = -min(L1, L2)
                actor_loss = -torch.min(L1, L2).mean()

                # optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                # actor_loss.backward()
                actor_loss.backward()
                # 梯度裁剪,只解决梯度爆炸问题，不解决梯度消失问题
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                # actor_optimizer.step()
                self.actor_optimizer.step()

        # update critic net
        for _ in range(C_UPDATE_STEPS):
            # critic的loss是td_error也就是advantage，可以是td_error的L1范数也可以是td_error的L2范数
            critic_loss = nn.MSELoss(reduction='mean')(self.critic(state), target_v)
            # optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            critic_loss.backward()
            # 梯度裁剪,只解决梯度爆炸问题，不解决梯度消失问题
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            # optimizer.step()
            self.critic_optimizer.step()

    def choose_action(self, s):
        """
        选择动作
        :param s:
        :return:
        """
        # 状态s放在torch.tensor上
        # actor net输出mu和sigma
        # 根据mu和sigma采样动作
        # 返回动作和动作的log概率值
        s = torch.FloatTensor(s)
        with torch.no_grad():
            mu, sigma = self.actor(s)
        # print(s, mu, sigma)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2, 2)
        return action.item(), action_log_prob.item()

    def get_v(self, s):
        """
        状态价值函数
        :param s:
        :return:
        """
        # 状态s放在torch.tensor上
        # critic net输出value
        s = torch.FloatTensor(s)
        with torch.no_grad():
            value = self.critic(s)
        return value.item()


env = gym.make('Pendulum-v1').unwrapped
env.reset(seed=0)
torch.manual_seed(0)
ppo = PPO(n_features=S_DIM, n_neuron=50,
          actor_learning_rate=A_LR, critic_learning_rate=C_LR)
all_ep_r = []  # 记录每个回合的累积reward值，当前回合的累积reward值 = 上一回合reward*0.9 + 当前回合reward*0.1

for ep in range(EP_MAX):
    s, info = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    buffer_log_old = [] # revised by lihan
    buffer_next_state = []
    ep_r = 0 # 每个回合的reward值，是回合每步的reward值的累加
    for t in range(EP_LEN):
        # in one episode
        env.render()
        # print(ep, t)
        a, a_log_prob_old = ppo.choose_action(s)
        s_, r, done, truncated, info = env.step([a])
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)  # normalize reward, find to be useful
        buffer_log_old.append(a_log_prob_old)
        buffer_next_state.append(s_)
        s = s_
        ep_r += r

        # 如果buffer收集一个batch了或者episode完了，那么update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN - 1:
            # print('update *****')
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            # discounted_r是target_v

            bs, ba = np.vstack(buffer_s), np.vstack(buffer_a)
            #br = np.vstack(buffer_r)
            br_next_state = np.vstack(buffer_next_state)
            br = np.array(discounted_r)[:, np.newaxis]
            blog_old = np.vstack(buffer_log_old)  # revised by lihan
            # 清空buffer
            buffer_s, buffer_a, buffer_r = [], [], []
            buffer_log_old = []  # revised by lihan
            buffer_next_state = []
            ppo.update(bs, ba, br, blog_old, br_next_state)  # 更新PPO
    if ep == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'EP: %i' % ep,
        "|EP_r %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name']=='kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()