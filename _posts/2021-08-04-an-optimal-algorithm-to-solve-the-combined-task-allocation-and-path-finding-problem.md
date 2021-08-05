---
layout: post
title: An Optimal Algorithm to Solve the Combined Task Allocation and Path Finding Problem
categories:
  - Papers Explained
tags:
  - TAPF
  - Multi agents
  - Task allocation
---
Paper Link: [https://arxiv.org/abs/1907.10360](https://arxiv.org/abs/1907.10360)

# Abstract
1. Delivering items from a given start to a goal pose in a factory setting while the delivering robots need to avoid 
collisions with each other.
2. **Task Conflict-Based Search (TCBS)** algorithm
3. **NP-hard** so the optimal solver cannot scale

# Introduction
1. Multi-Agent Path Finding (MAPF) considers **collision-free** path when assigning a fixed start and goal state to each 
agent.
2. Multi-Agent Task Allocation (MATA) or Multi-Agent Pick-up and Delivery (MAPD) focus on solving the allocation of 
delivery jobs to the agents while not considering such collisions between the agent paths.
3. 이 논문의 저자는 MAPF와 MATA의 특성을 모두 고려하여, Task Allocation을 진행하되 collision-free Path Finding을 제시하는 Combined
Task Allocation and Path Finding (CTAPF)에 contribution을 잡은 듯 하다. 내가 알기론 Task Allocation and Path Finding (TAPF)라는 
연구 분야가 존재하는 걸로 아는데 아무튼 그렇다고 한다.

# Problem Formulation
1. 아래는 이 논문이 사용한 환경 세팅이다. 각 agent가 부품을 특정 goal position까지 collision 없이 옮기는 미션이다. 

<p align="center">
    <img src = "https://user-images.githubusercontent.com/45442859/128128655-2f07aad7-11fe-44b7-862a-673080966d37.png" alt = "env" width = "50%" height = "50%"/>
</p>

2. **Single-Task/Single-Robot/Time-Extended**로 설계 되었으며, 이 의미는 하나의 로봇은 한번에 하나의 task만 수행할 수 있고, 모든 task는 하나의 로봇으로 수행 가능하다는 뜻이다.
3. 각 robot의 capacity는 1로 설정 (i.e. 한번에 하나의 부품만 운반 가능)
4. Collision-free path를 찾기 위해 optimal solver로 분류되는 **Conflict-Based Search (CBS)** 사용.
5. MILP를 사용해 MATA와 MAPF를 함께 풀고자 하는 시도가 있었지만 agent 간의 conflict가 고려가 안 됐기 때문에 본 논문에서는 **Mixed Integer Non-Linear Programming (MINLP)** 로 풀고자 함.

