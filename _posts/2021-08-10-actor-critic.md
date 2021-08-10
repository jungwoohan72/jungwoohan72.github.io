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

Paper Link: [https://arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783)

# Actor-Critic이란?

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

# Pseudocode for Q actor critic

Critic으로 학습한 Q function 사용

![image](https://user-images.githubusercontent.com/45442859/128870844-0e8ce83f-1a74-4ddf-9a2d-964d5c8eea80.png)
