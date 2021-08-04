---
layout: post
title: REINFORCE Algorithm
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

This is $\theta$



