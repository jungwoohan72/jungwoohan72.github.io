---
layout: post
title: "ScheduleNet: Learn to Solve MinMax Multiple Travelling Salesman Problem"
comments: true
categories:
  - Papers Explained
tags:
  - Multi agents
  - Task allocation
  - Reinforcement Learning
  - MARL
---

# Abstract

* Kool의 Attention! Learn to Solve Routing Problems 같은 경우 single-agent라는 한계 조재
* 기존 방법론은 대부분 MinSum이지만 본 논문에서는 MinMax에 집중
* MinMax란 minimizing the length of the longest subroute
* mTSP는 distributed scheduling policy를 통해 cooperative strategic routing을 만들어 내야 되서 어려움
* 또한, delayed and sparse single reward라는 점도 학습을 어렵게 만듬.
* ScheduleNet은 arbitrary number of workers and tasks에 대해 적용 가능 -> Scalability가 좋음.
* Training 방법으로는 Clipped REINFORCE 사용.

# Introduction

* MinMax의 장점은 MinSum과 달리 agent 별로 balanced된 task 분배가 가능하다는 점이다.
* 이 논문의 주요 부분은 아래와 같음.

    1. Decentralized policy를 통해 arbitrary number of agents and tasks에 대해 적용 가능.
    2. Attention-based node embeddings
    3. Training with a single delayed shared reward signal

# Problem Formulation

## State Definition
1. Salesman: V<sub>T</sub> = {1,2,...,m}
2. Cities: V<sub>C</sub> = {m+1,m+2,...,m+N}
3. p<sup>i</sup>: 2D-coordinates of entities (salesman, cities, and the depot)
    1. time-dependent for workers
    2. static for tasks and the depot 
4. &tau;: event where any worker reaches its assigned city
5. t(&tau;): time of event &tau;
6. State s: s<sup>i</sup><sub>&tau;</sub> = (p<sup>i</sup><sub>&tau;</sub>, 1<sup>active</sup><sub>&tau;</sub>, 1<sup>assigned</sup><sub>&tau;</sub>)
7. 1<sup>active</sup><sub>&tau;</sub>: 
    1. already visited task is inactive
    2. worker at the depot is inactive
8. 1<sup>assigned</sup><sub>&tau;</sub>:
    1. whether worker is assigned to a task or not
9. s<sup>env</sup><sub>&tau;</sub>: current time, sequence of tasks visited by each worker
10. s<sub>&tau;</sub> state at the &tau;-th event: ({s<sup>i</sup><sub>&tau;</sub>}<sup>m+N</sup><sub>i=1</sub>, s<sup>env</sup><sub>&tau;</sub>)

## Action
1. worker-to-tak assignment

## Reward 
1. For nonterminal events, r(s<sub>&tau;</sub>) = 0
2. r(s<sub>T</sub>) = t(T)

# ScheduleNet

<img width="550" alt="스크린샷 2021-11-13 오후 7 38 55" src="https://user-images.githubusercontent.com/45442859/141615446-3af13180-8f06-4a26-bdb9-6eca5f29834d.png">

* Whenever event occurs, contruct graph as shown above.
* Edge feature is the Euclidean distance between two nodes
* h<sub>i</sub>: node embedding (maybe node coordinate?)
* h<sub>ij</sub>: edge embedding (high-dim conversion of Euclidean distance)
* Single iteration of TGA embedding consists of three phases: (1) edge update (2) message aggregation (3) node update

    <img width="176" alt="스크린샷 2021-11-13 오후 8 15 03" src="https://user-images.githubusercontent.com/45442859/141634414-a082e975-0d4b-4a7f-af30-9b8e39a0b247.png">

* h'<sub>ij</sub>: embeddings after TGA update (message from source node j to destination node i)
* z<sub>ij</sub>: how valuable is node i is to node j (just like compatibility in self-attention)
* k<sub>j</sub>: type of entity j (i.e. active worker if 1<sup>active</sup><sub>&tau;</sub> = 1)

# How Type-aware Graph Attention works?
1. "Type-aware" edge update  
&nbsp;&nbsp;&nbsp;&nbsp;i. Context embedding c<sub>ij</sub> of edge e<sub>ij</sub>  
&nbsp;&nbsp;&nbsp;&nbsp;![Screenshot from 2021-11-15 14-27-11](https://user-images.githubusercontent.com/45442859/141727321-5bee78aa-c81c-4b34-963b-a3f443539540.png)  
&nbsp;&nbsp;&nbsp;&nbsp;* Source node j의 종류에 따라 context node Embedding 달라짐.

&nbsp;&nbsp;&nbsp;&nbsp;ii. Type-aware edge encoding  
&nbsp;&nbsp;&nbsp;&nbsp;![Screenshot from 2021-11-15 14-43-35](https://user-images.githubusercontent.com/45442859/141728693-0311320f-1b34-4c0a-af40-58b8c8feab14.png)  
&nbsp;&nbsp;&nbsp;&nbsp;* MI layer dynamically generates its parameter depending on the context c<sub>ij</sub>  
&nbsp;&nbsp;&nbsp;&nbsp;* Dynamic edge feature which varies depending on the source node type.


   Compute "type-aware" edge encoding u<sub>ij</sub>
    2. 
   
Using the edge encoding, compute the updated edge features and attention logit using a MI layer.
    3. 