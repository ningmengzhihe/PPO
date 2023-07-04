"""
ref
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.distributions import Normal
from tqdm import tqdm


class ActorCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ActorCritic, self).__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_size),
            nn.Tanh()
        )

        self.critic_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.actor_net(x), self.critic_net(x)


class PPO:
    def __init__(self, obs_size, act_size, lr, gamma, clip_ratio):
        self.actor_critic = ActorCritic(obs_size, act_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio

    def update(self, rollouts):
        obs, act, rew, logp_old, val_old = rollouts[:5]

        # Calculate returns
        returns = np.zeros_like(rew)
        for t in reversed(range(len(rew))):
            if t == len(rew) - 1:
                returns[t] = rew[t]
            else:
                returns[t] = rew[t] + self.gamma * returns[t + 1]

        # 值函数估计
        values = self.actor_critic(torch.tensor(obs).float())[1].detach().numpy()
        # 优势函数估计
        adv = returns - np.sum(values, axis=1)

        # 和前面一样，不多赘述
        act = torch.tensor(act).float()
        logp_old = torch.tensor(logp_old).float()

        mean, std = self.actor_critic(obs)
        dist = Normal(mean, std.abs() + 1e-8)
        pi_old = torch.exp(dist.log_prob(act))
        ratio = pi_old / (torch.exp(logp_old) + 1e-8)
        surr1 = ratio * torch.from_numpy(adv).float()
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * torch.from_numpy(adv).float()
        actor_loss = -torch.min(surr1, surr2).mean()

        # 计算critic loss
        val_old = torch.tensor(val_old).float()
        val = self.actor_critic(torch.tensor(obs).float())[1]
        critic_loss = nn.MSELoss()(val.squeeze(), torch.tensor(returns).float())

        # Update neural network parameters
        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train(env_name, epochs, steps_per_epoch, batch_size, lr, gamma, clip_ratio):
    env = gym.make(env_name, render_mode='human')
    obs_size = env.observation_space.shape[0]  # 2
    act_size = env.action_space.shape[0]  # 1(连续型)

    ppo = PPO(obs_size, act_size, lr, gamma, clip_ratio)
    ep_reward = deque(maxlen=10)
    print('Started!')
    for epoch in range(epochs):
        obs_buf, act_buf, rew_buf, logp_buf = [], [], [], []
        for _ in tqdm(range(steps_per_epoch)):
            obs, info = env.reset()
            ep_reward.append(0)
            for t in range(batch_size):
                '''
                我们获取的std，其实是critic的输出，而不是actor的输出。
                这是PPO算法的实现里比较常见的一种方法。
                但这里有一个问题，标准差会随着值函数的增加而增加，可能导致策略的不稳定性，和我们要的确定性是矛盾的。
                为了解决这个问题，有一些改进的算法采用了不同的方法来计算标准差。
                例如，Trust Region Policy Optimization (TRPO)算法使用了一个额外的超参数delta来限制动作分布的变化。
                PPO算法使用的剪切范围来限制新策略和旧策略之间的差异，也是为了这个目的。
                这些方法可以在一定程度上解决标准差和值函数之间的矛盾问题。
                '''
                mean, std = ppo.actor_critic(torch.tensor(obs).float())  # 动作分布，均值和方差
                dist = Normal(mean, std.abs() + 1e-8)  # 构建正态分布，和之前的Categorical是一个逻辑
                act = dist.sample()  # 抽样
                logp = dist.log_prob(act)  # 取log

                # 状态、动作、奖励、logp
                obs_buf.append(obs)
                act_buf.append(act.detach().numpy())
                rew_buf.append(0)
                logp_buf.append(logp.detach().numpy())

                # 走一步
                obs, rew, done, _, _ = env.step(act.detach().numpy())

                # 更新reward
                ep_reward[-1] += rew
                rew_buf[-1] += rew

                if done:
                    break

            # 更新actor-critic
            ppo.update((obs_buf, act_buf, rew_buf, logp_buf, np.zeros_like(rew_buf)))
        print("Epoch: {}, Avg Reward: {:.2f}".format(epoch, np.mean(ep_reward)))


if __name__ == '__main__':
    train('Pendulum-v1', epochs=2, steps_per_epoch=20, batch_size=128, lr=0.002, gamma=0.99,
          clip_ratio=0.2)

    # MountainCarContinuous-v0