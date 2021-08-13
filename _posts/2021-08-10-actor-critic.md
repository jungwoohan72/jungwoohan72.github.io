---
layout: post
title: Actor-Critic (A2C and A3C)
categories:
  - RL Algorithm Replication
tags:
  - RL
  - Policy gradient
  - A2C
  - A3C
---

A3C Paper Link: [https://arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783)

# Synchronous Actor-Critic (A2C)이란?

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

# Pseudocode for REINFORCE

Actor-critic과 REINFORCE의 비교를 위해 REINFORCE의 Pseudo 코드를 다시 한 번 보고 가자.

![image](https://user-images.githubusercontent.com/45442859/128873153-4859a50c-94e3-4d59-9d07-c7a0df078159.png)

# Pseudocode for TD actor critic

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

# Pseudocode for Q actor critic

Critic으로 학습한 Q function 사용

![image](https://user-images.githubusercontent.com/45442859/128870844-0e8ce83f-1a74-4ddf-9a2d-964d5c8eea80.png)
