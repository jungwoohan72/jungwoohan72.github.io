---
layout: post
title: An Optimal Algorithm to Solve the Combined Task Allocation and Path Finding Problem
categories:
  - Papers Explained
tags:
  - MATF
  - Multi agents
---

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
4. 아래는 이 논문이 사용한 환경 세팅이다. 

![env](./_screenshots/2021_08_04_1.png)
