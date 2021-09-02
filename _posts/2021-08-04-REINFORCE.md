---
layout: post
title: REINFORCE Algorithm
comments: true
categories:
  - RL Algorithm Replication
tags:
  - RL
  - Policy gradient
---

Paper Link: [https://arxiv.org/abs/1604.06778](https://arxiv.org/abs/1604.06778)

Continuous Control에 대한 벤치마크를 제공한 논문인 Benchmarking Deep Reinforcement Learning for Continuous Control에서 
policy gradient 알고리즘 중 하나로 사용됨. 

# 위 논문이 왜 중요한가?
* 많이들 아는 Atari 게임을 강화학습을 통해 플레이한 것은 raw pixel data를 받아서 discrete한 action space 중 적절한 action을 취하도록 한 것임.
* 이 논문이 나올 2016년 당시에만 해도 continuous action space에서 challenging한 문제를 강화학습으로 푼 벤치마크가 존재하지 않아 알고리즘 별로 성능 비교하기가 어려웠음.
* Systematic한 성능 평가와 알고리즘 별 성능 비교는 이 분야의 발전을 위해서 필수적임.
* 위와 같은 이유로 31가지의 continuous control 문제에 대한 벤치마크를 제공함.
  * Basic Task: cart-pole balancing 같은 비교적 간단한 문제
  * Locomotion Task: Swimmer 같이 dynamics가 학습이 안된 임의의 물체를 앞으로 나아가게 하는 비교적 높은 DoF를 가진 문제
![image](https://user-images.githubusercontent.com/45442859/128172670-f0c322ce-a5e3-49f6-972a-76a025a02e41.png)

  * Partially Observable Task: sensor noise나 delayed action을 고려하여 full state가 주어지지 않는 문제
  * Hierarchical Task: Low-level motor control과 high-level planning 등을 고려한 문제. 예를 들면 Locomotion task에서의 dynamics 학습 task와
  학습한 dynamics를 가지고 미로를 빠져나가는 경로를 planning하는 것을 합친 것.
![image](https://user-images.githubusercontent.com/45442859/128172769-bdef2559-8149-4214-88d0-1de2ba51e0d5.png)
  
# REINFORCE Algorithm

이번 포스트에서는 위 논문에서 사용된 여러가지 policy gradient 알고리즘 중 가장 대표적인 REINFORCE 알고리즘에 대한 공부를 해보고 구현까지 해보도록 한다.

강화학습에서 가장 중요한 것 중 하나는 action을 결정하는 policy일 것이다. 이러한 policy는 &pi;(a|s,**&theta;**)로 표기가 많이 되는데 이는 단순히 s라는 
state에서 a라는 action을 취할 확률을 &theta;라는 parameter로 표현했다는 뜻이다.

Policy gradient 알고리즘은 **&theta;** 로 표현될 수 있는 cost function J(**&theta;**)을 최소화 혹은 최대화 시키는 방향으로 **&theta;** 값을 
업데이트 하겠다는 의미이다. 좀더 강화학습스럽게 풀어보면, reward를 최대화할 수 있는 방향으로 **&theta;** 를 학습시키는 과정으로 생각하면 될 것 같다.

<p align="center">
    <img src = "https://user-images.githubusercontent.com/45442859/128175275-97264c06-daa8-4d1d-9040-4ece3632ddd6.png" alt = "env" width = "50%" height = "50%"/>
</p>

많은 경우 cost function으로 value function을 사용한다. Value fuction에 gradient를 취해주게 되면 아래와 같이 최종 form을 유도할 수 있다. (사실 아직 수학적으로 완벽하게 이해는 못한 상태다...)

<p align="center">
    <img src = "https://user-images.githubusercontent.com/45442859/128176528-bbb5f19c-14cf-4c03-a831-755882640cd3.png" alt = "env" width = "75%" height = "75%"/>
</p>

여기서 **&mu;** 는 on-policy distribution under **&pi;** 로 policy **&pi;** 를 따라서 episode를 진행했을 때 s라는 state가 몇번 나타나는지에 대한 distribution이다. 
따라서 앞부분 sigma는 아래와 같이 각 state에 대한 expectation으로 표시되어질 수 있다.

<p align="center">
    <img src = "https://user-images.githubusercontent.com/45442859/128177691-8b8cf131-c02c-4b73-8d1e-4c1f5b113194.png" alt = "env" width = "75%" height = "75%"/>
</p>

그리고 우리가 아는 REINFORCE 알고리즘으로 가기 위해서는 한가지 trick이 존재하는데 이는 아래와 같다.

<p align="center">
    <img src = "https://user-images.githubusercontent.com/45442859/128178207-f45475da-36e7-4ef7-8e29-e774f6d4c296.png" alt = "env" width = "75%" height = "75%"/>
</p>

이제 우리가 아는 REINFORCE 알고리즘 형태이다!

<p align="center">
    <img src = "https://user-images.githubusercontent.com/45442859/128178483-ce5a5d6f-8e5a-449a-9c3a-630309fa1643.png" alt = "env" width = "75%" height = "75%"/>
</p>

처음에 수식적으로 왜 +인지 살짝 헷갈렸는데, 보통의 gradient descent가 아니라, 여기선 theta를 objective function을 maximize하는 방향으로 학습시켜줘야 되기 때문에 경사 방향을 그대로 유지해줘야 한다.
예를 들면, gradient가 음수라고 하면, theta가 감소할 때 objective function이 증가한다는 뜻이다. 그래서 기존 theta 값에서 빼줘야 된다. 

# Pseudocode for vanilla REINFORCE

<p align="center">
    <img src = "https://user-images.githubusercontent.com/45442859/128178615-b78f22b8-cb15-467a-9660-befa4b6d4f51.png" alt = "env" width = "100%" height = "100%"/>
</p>

마지막에 ln 함수가 붙는 이유는 아래에 있다.

![Image](https://user-images.githubusercontent.com/45442859/128178929-fe68a42b-4576-417d-9fcf-224628493c44.png)

# REINFORCE with Baseline

REINFORCE는 Monte Carlo 방식이므로 unbiased한 return을 사용할 수 있다는 장점이 있지만, episode 별로 분산이 크다. 이런 단점을 해결하기 위해 Baseline이라는 
개념을 도입한다. 

![image](https://user-images.githubusercontent.com/45442859/128862179-01a1a2dc-1d72-4457-bbeb-f2bae85b6c98.png)

b(s)를 빼줘도 전체 수식에 영향이 없는 이유는 b(s)는 a에 independent하기 때문이다.

![image](https://user-images.githubusercontent.com/45442859/128862303-78750d8d-28f4-4791-b51f-858dae9a2385.png)

그래서 update rule은 아래와 같이 유도할 수 있다.

![image](https://user-images.githubusercontent.com/45442859/128862419-41a25faa-8079-4d46-8621-35465e3a4303.png)

보통 이 baseline을 value function으로 잡는데, 그 이유는 직관적으로 생각해보면 G<sub>t</sub>-b(S<sub>t</sub>) 식으로부터 현재의 행동으로 인한 return (G or Q)이 평균적으로 얻을 수 있는 return (V)보다 
얼마나 좋은지를 측정 가능하기 때문이다. 

# Pseudocode for REINFORCE with Baseline

![image](https://user-images.githubusercontent.com/45442859/128862971-38f98d82-786e-438f-885e-370364ede6dc.png)

추가적인 데이터를 얻어야 되는 필요성 없이 baseline이 되는 value function을 학습 가능하다. 에피소드로부터 time step마다의 return을 계산 가능하고, 
학습하는 value function이 이같은 return을 따라가도록 학습시키는 것이다. 생각보다 이같은 baseline 기법을 사용하는 게 성능 차이가 많이 난다고 한다.

# REINFORCE Implementation

```python
import torch
import torch.nn as nn
import numpy as np

from torch.distributions.categorical import Categorical

class REINFORCE(nn.Module):
    def __init__(self, policy, gamma, lr, device):
        super(REINFORCE, self).__init__()
        self.policy = policy
        self.gamma = gamma
        self.opt = torch.optim.Adam(params = self.policy.parameters(), lr = lr)
        self._eps = 1e-25
        self.device = device

    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy(state).to(self.device)
            dist = Categorical(logits=logits)
            a = dist.sample()
        return a
    
    ###위에서 설명한 것처럼 매 time step마다 update하는 버전
    def update_step_by_step(self, episode):
        states = episode[0].flip(dims = [0]).to(self.device)
        actions = episode[1].flip(dims = [0]).to(self.device)
        rewards = episode[2].flip(dims = [0]).to(self.device)

        g = 0
        for s,a,r in zip(states, actions, rewards):
            g = r + g*self.gamma
            dist = Categorical(logits = self.policy(s)) # sampling
            prob = dist.probs[a]

            loss = -torch.log(prob + self._eps)*g

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
    
    ### 매 step 업데이트하는 건 비효율적이기 때문에 보통 episode마다 업데이트한다고 함.
    def update_episode(self, episode):
        states = episode[0].flip(dims = [0]).to(self.device)
        actions = episode[1].flip(dims = [0]).to(self.device)
        rewards = episode[2].flip(dims = [0]).to(self.device)

        g = 0
        returns = []
        for s,a,r in zip(states, actions, rewards):
            g = r + self.gamma*g
            returns.append(g)

        returns = torch.tensor(returns).to(self.device)
        
        # baseline trick!!! Return들의 평균값을 baseline으로 사용.
        returns = (returns - returns.mean()) / (returns.std() + self._eps)
        returns.to(self.device)

        dist = Categorical(logits = self.policy(states)) # probability for each action -> sampling
        prob = dist.probs[range(states.shape[0]), actions] # (states.shape[0], 1) tensor

        self.opt.zero_grad()
        loss = -torch.log(prob+self._eps)*returns
        loss = loss.mean()
        loss.backward()
        self.opt.step()
```

* 그냥 log가 아니라 -log인 이유는 pytorch의 opt.step()은 loss function을 minimizing하는 방향으로 학습을 시키기 때문인데, 여기서는 loss function이 value function이고,
이를 최대화 시키는 방향으로 policy를 학습시키고 싶기 때문이다. -log를 minimize하는 건 log를 maximize하는 것과 같음.

## 학습 결과
* 간단한 환경이지만 학습이 잘 된듯하다. max episode length인 500까지 모두 도달.

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/45442859/128811906-10e920df-047a-4e19-b449-9e353b9da543.gif)
