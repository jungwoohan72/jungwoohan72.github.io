---
layout: post
title: Actor-Critic (A3C)
comments: true
categories:
  - RL Algorithm Replication
tags:
  - RL
  - Policy gradient
  - A2C
  - A3C
---

A3C Paper Link: [https://arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783)

# Actor-Critic 이란?

REINFORCE with Baseline의 update rule은 다음과 같다.

![image](https://user-images.githubusercontent.com/45442859/128862419-41a25faa-8079-4d46-8621-35465e3a4303.png)

REINFORCE의 근본적인 목표는 행동가치함수를 통해 현재의 policy를 학습하는 것인데, 이는 어떤 행동을 취하고 그 action의 행동가치함수 값이 높으면
그 action을 할 확률을 높이도록 policy의 parameter를 조정하는 식으로 이루어진다. 하지만 Baseline으로 사용되는 상태가치함수는 위 같은 행동을 취하기 전을 기준으로 하기 때문에
해당 action이 좋은지 나쁜지를 판단하기엔 적절하지 않다.

또한, REINFORCE는 Monte Carlo 고유의 문제인 high variance와 episode가 끝날 때까지 기다려야 한다는 단점이 있다. return을 학습한 Q-network로부터 얻어서 step마다 업데이트 하는 것을 제안한 것이
Actor-Critic Method이다.

에피소드가 끝날 때까지 기다렸다가 actual return을 사용하는 게 아닌 TD(0)와 같은 추정값을 사용하는 방식이다. 

![image](https://user-images.githubusercontent.com/45442859/128868704-ad371a00-4c05-451f-9cf1-9272b5031389.png)

여기서 R<sub>t+1</sub>은 환경으로부터 얻은 실제값이므로 취한 action에 대한 평가가 가능해진다.
앞에서 말했듯이, 위와 같 S<sub>t+1</sub>에서의 추정값을 사용하면 bias가 생기긴 하지만 variance 측면에서 장점이 있고, online update가 가능하는 장점이 있다.
bias 같은 경우는 TD(1), TD(2), ...와 같이 n-step return을 사용함으로써 줄일 수 있다. 
이렇게 action의 quality를 평가하기 위해 사용되는 상태가치함수를 critic이라고 한다.

## Pseudocode for REINFORCE

Actor-critic과 REINFORCE의 비교를 위해 REINFORCE의 Pseudo 코드를 다시 한 번 보고 가자.

![image](https://user-images.githubusercontent.com/45442859/128873153-4859a50c-94e3-4d59-9d07-c7a0df078159.png)

## Pseudocode for TD actor critic

![image](https://user-images.githubusercontent.com/45442859/128870090-a57ee7ad-9a46-41b2-94a0-92b7cee43380.png)

action을 sampling하는 현재의 policy를 actor라 칭하고, 이를 평가하는 상태가치함수를 critic이라 칭한다. 
둘다 학습을 시켜줘야된다. TD Actor-critic은 critic으로 TD error를 사용하는 경우이다.
Actor의 경우 경사 하강을 통해 loss function을 최소화 해줘야 하고, Critic 같은 경우는 앞에서 본 것과 같이 최대화 시켜줘야 하기 때문에 경사 상승을 이용한다.

```python

import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical

class A2C(nn.Module):
    def __init__(self, policy_net, value_net, gamma, lr, device):
        super(A2C, self).__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        self.gamma = gamma
        self.lr = lr

        params = list(policy_net.parameters()) + list(value_net.parameters())
        self.optimizer = torch.optim.Adam(params = params, lr = lr)

        self._eps = 1e-25
        self._mse = torch.nn.MSELoss()
        self.device = device

    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy_net(state).to(self.device)
            dist = Categorical(logits = logits)
            a = dist.sample() # torch.Size([1])
        return a

    def update(self, state, action, reward, next_state, done):

        # action size: torch.Size([1,1])

        next_state = torch.from_numpy(next_state).float().to(self.device)
        next_state = next_state.view((1,4)) # value_net input은 size [1,4]여야 함.
        reward = torch.tensor(reward).to(self.device)

        with torch.no_grad():
            td_target = reward + self.gamma * self.value_net(next_state) * (1-done)
            td_error = td_target - self.value_net(state)

        dist = Categorical(logits = self.policy_net(state)) # torch.Size([1,2])
        prob = dist.probs.gather(1,action)

        v = self.value_net(state)

        loss = -torch.log(prob + self._eps)*td_error + self._mse(v, td_target*td_error  # policy loss + value loss / shape: torch.Size([1,1])
        loss = loss.mean() # shape: torch.Size([])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

```

TD Error의 경우 Advantage Function의 unbiased estimate이므로 Critic으로 Advantage Function을 사용하기도 한다.
하지만 TD Error을 사용하면 Advantage Function을 사용할 때와 달리, 상태가치함수만 학습하면 된다.

## Pseudocode for Q actor critic

Critic으로 학습한 Q function 사용

![image](https://user-images.githubusercontent.com/45442859/128870844-0e8ce83f-1a74-4ddf-9a2d-964d5c8eea80.png)

## Actor-Critic 종류

![image](https://user-images.githubusercontent.com/45442859/129993657-1aa7d106-6773-461a-ae99-e4a99db60894.png)

# Asynchronous Advantage Actor-Critic

논문에서 다루는 A3C 알고리즘은 TD Actor-Critic을 Asynchronous하게 업데이트한다. 즉, Global하게 공유하는 Actor-Critic pair를 여러개의 Actor-Critic thread를 통해
업데이트하는 과정이다. Training과 정은 TD Actor-Critic과 동일하며, 여러 개의 thread를 사용해서 비동기적으로 업데이트한다는 특징이 있다.

## Pseudocode for A3C

![image](https://user-images.githubusercontent.com/45442859/130006230-ef9e6924-2ee9-4439-8e6f-77c656d75c83.png)

* t는 local actor-critic thread 업데이트를 위해 사용됨.
* T는 local actor-critic update의 총합. 즉, global actor-critic이 몇 번 업데이트 되었는지를 체크.
* local actor-critic은 global actor-critic으로부터 parameter를 t<sub>max</sub>마다 복사해서 학습에 사용.
* Loss function을 보면 TD error가 사용된 것을 볼 수 있다.

## Implementation of A3C

Miltiprocessing을 진행해야 하기 때문에 구현을 어떻게 해야할지 감이 안 왔다. 그래서 그냥 느낌만 잡고 가기로 결정!

[https://github.com/seungeunrho/minimalRL](https://github.com/seungeunrho/minimalRL)

강화학습 유튜버 팡요랩 님이 운영하시는 Github인데 논문 읽기 전에 관련 영상을 보고 읽으면 이해가 더 잘 된다. 추천!

그리고 위 A3C 코드가 내가 구글링해서 본 모든 코드 중에서 가장 간결하고 논문 flow 그대로 구현한 것 같다. 


위의 Vanilla Actor-Critic과 비교했을 때 진짜 빠르고, 성능이 좋다... 신기...

```python

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time

# Hyperparameters
n_train_processes = mp.cpu_count()
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_ep = 500
max_test_ep = 520


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def train(global_model, rank):
    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

    env = gym.make('CartPole-v1')

    for n_epi in range(max_train_ep):
        done = False
        s = env.reset() # s.shape -> (4,)
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                prob = local_model.pi(torch.from_numpy(s).float()) # torch.size([2]) | torch.from_numpy(s).shape: torch.size([4])
                m = Categorical(prob)
                a = m.sample().item() # int
                s_prime, r, done, info = env.step(a)

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r/100.0)

                s = s_prime
                if done:
                    break

            s_final = torch.tensor(s_prime, dtype=torch.float) # torch.size([4])
            R = 0.0 if done else local_model.v(s_final).item()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward # n-step TD target
                td_target_lst.append([R])
            td_target_lst.reverse()

            s_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                torch.tensor(td_target_lst) # torch.size([update_interval,4]), torch.size([update_interval,1]), torch.size([update_interval,1])
            advantage = td_target - local_model.v(s_batch)

            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)
            loss = -torch.log(pi_a) * advantage.detach() + \
                F.smooth_l1_loss(local_model.v(s_batch), td_target.detach()) # torch.size([5,1])

            optimizer.zero_grad()
            loss.mean().backward()
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict()) 

    env.close()
    print("Training process {} reached maximum episode.".format(rank))


def test(global_model):
    env = gym.make('CartPole-v1')
    score = 0.0
    print_interval = 20

    for n_epi in range(max_test_ep):
        done = False
        s = env.reset()
        while not done:
            if n_epi > 390:
                env.render()
            prob = global_model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().item()
            s_prime, r, done, info = env.step(a)
            s = s_prime
            score += r

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, score/print_interval))
            score = 0.0
            time.sleep(1)
    env.close()


if __name__ == '__main__':
    global_model = ActorCritic()
    global_model.share_memory()

    processes = []
    print("Available CPU Count:", n_train_processes)
    for rank in range(n_train_processes + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(global_model,))
        else:
            p = mp.Process(target=train, args=(global_model, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

```

* mp.cpu_count()를 하면 돌릴 수 있는 cpu 개수가 나오는데 내껀 12개였다.
* 각각 actor-critic thread가 max_train_ep만큼 데이터를 수집하고, update_interval마다 global actor-critic을 업데이트하므로 
각 local thread에서 max_train_ep/update_interval (여기선 500/5 = 100)만큼 global actor-critic을 업데이트 한다.
* Local thread가 12개이므로 총 1200번의 업데이트가 이루어지는데, 그냥 vanilla actor-critic을 썼을 때보다 10배 정도 에피소드 효율이 좋은 것 같다.