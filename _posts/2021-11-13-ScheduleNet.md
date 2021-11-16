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

   i. Context embedding c<sub>ij</sub> of edge e<sub>ij</sub>
   ![Screenshot from 2021-11-15 14-27-11](https://user-images.githubusercontent.com/45442859/141727321-5bee78aa-c81c-4b34-963b-a3f443539540.png)
   * Source node j의 종류에 따라 context node Embedding 달라짐.  

   ii. Type-aware edge encoding

   ![Screenshot from 2021-11-15 14-43-35](https://user-images.githubusercontent.com/45442859/141728693-0311320f-1b34-4c0a-af40-58b8c8feab14.png)
   * MI layer dynamically generates its parameter depending on the context c<sub>ij</sub>  
   * Dynamic edge feature which varies depending on the source node type.

   iii. Type-aware edge encoding

   ![Screenshot from 2021-11-15 14-53-30](https://user-images.githubusercontent.com/45442859/141729548-efdc586c-8286-43dc-b71e-8426019848a5.png)
   * Upper one is for TGA<sub>E</sub>
   * Lower one is for TGA<sub>A</sub>

2. Message Aggregation
   i. k type 별로 따로따로 attention score 구함.

   ![Screenshot from 2021-11-15 15-01-54](https://user-images.githubusercontent.com/45442859/141730343-9a79d2be-12cf-416a-853a-94e5a9043efb.png)
   * v<sub>i</sub>를 기준으로 주변에 type-k neighbor에 대하여 attention score 계산.
   * 현재 세팅에선 agent, task, depot 세가지 종류의 노드가 존재하므로 두가지 attention score가 생길 것으로 예상.

   ii. Attention score에 앞에서 구한 node embedding을 node type별로 곱해서 concat 진행.

   ![Screenshot from 2021-11-15 18-51-34](https://user-images.githubusercontent.com/45442859/141760399-0607712d-826d-418a-bf69-e6939d863c62.png)  
   ![Screenshot from 2021-11-15 18-53-05](https://user-images.githubusercontent.com/45442859/141760593-4eaae7b0-ffad-4b89-a839-15f3f646ba5c.png)

3. Node Update
   i. Context embedding c<sub>i</sub> of node v<sub>i</sub>

   ![Screenshot from 2021-11-15 18-57-56](https://user-images.githubusercontent.com/45442859/141761370-535d7bff-ff19-4251-92b4-8a098da6578c.png)

   ![Screenshot from 2021-11-15 18-58-59](https://user-images.githubusercontent.com/45442859/141761490-7dce797d-a9fd-43a5-92e1-edce69331d56.png)

   ![Screenshot from 2021-11-15 18-59-30](https://user-images.githubusercontent.com/45442859/141761579-6260ed3f-2d68-4a88-afb8-3c7f64fe876d.png)

4. Type-aware aggregation이 중요한 이유
* Node distribution은 task의 갯수가 월등히 많을 가능성이 큼.
* Type-aware aggregation은 위 같은 불균형으로 인해 생기는 학습의 어려움을 어느정도 완화. 

# Overall Procedure

![Screenshot from 2021-11-15 19-07-16](https://user-images.githubusercontent.com/45442859/141762834-379992f6-0e49-4535-b85c-8d0242cbc07e.png)

* raw-2-hid: encode initial node and edge features to obtain initial h<sup>(0)</sup><sub>i</sub>, h<sup>(0)</sup><sub>ij</sub>
* hid-2-hid: encode the target subgraph G<sup>s</sup><sub>&tau;</sub>
* The subgraph is composed of a target-worker(unassigned-worker) node and all unassigned-city nodes.
* hid-2-hid layer is repeated H times to obtain final hidden embeddings h<sup>(H)</sup><sub>i</sub>, h<sup>(H)</sup><sub>ij</sub>

# Probability Generation Process

![Screenshot from 2021-11-15 19-13-20](https://user-images.githubusercontent.com/45442859/141763664-bcbb3008-66cf-42cf-b128-9a92135d3afe.png)
![Screenshot from 2021-11-15 19-13-49](https://user-images.githubusercontent.com/45442859/141763760-40a83659-e2db-47d2-bc79-1fa5229b6539.png)

# Training

1. Makespan Normalization

   i. Only reward signal is makespan of mTSP, which is M(&pi;)
   ii. Such reward varies depending on the problem size, topology of the map, and ...
   iii. So, it is normalized using the equation below.
   ![Screenshot from 2021-11-15 19-32-38](https://user-images.githubusercontent.com/45442859/141766563-472a6ab0-8a41-48be-84b5-60277b9d2a73.png)

