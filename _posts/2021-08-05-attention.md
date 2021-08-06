---
layout: post
title: Attention, Learn to Solve Routing Problems!
categories:
  - Papers Explained
tags:
  - RL
  - Routing
---
Paper Link: [https://arxiv.org/abs/1803.08475](https://arxiv.org/abs/1803.08475)

# Abstract

1. 조합최적화 문제를 풀 때 기존의 optimal solution을 구하는 과정이 아닌, heuristic을 사용하여 sub-optimal 하지만 computational cost
측면에서 이점을 가지는 solution을 찾고자 하는 아이디어가 제시됨.
2. 본 논문에서는 Pointer Network에서 사용된 attention 개념을 토대로 조합 최적화 문제를 풀기 위한 모델을 제시하고 REINFORCE 알고리즘을 사용해 학습 시키는
방법을 제시. 
3. Travelling Salesman Problem (TSP), Vehicle Routing Problem (VRP), Orienteering Problem (OP), Prize Collecting TSP (PCTSP)
등 여러 routing 문제를 하나의 hyperparameter로 풀어서 제시한 모델의 generalizability를 강조.

# What is Combinatorial Optimization (조합최적화)?

Definition: Process of searching for maxima (or minima) of an objective function F whose domain is discrete but
large configuration space (as opposed to an N-dimensional continuous space)

쉽게 말하면 objective function을 최대화 하거나 loss function을 최소화하는 조합을 찾아내는 게 목적인 연구분야.

대표적으로 TSP가 있는데, traveling distance나 traveling time을 최소화 하면서 주어진 node들을 한번씩만 방문하고 출발지점으로 돌아오는 문제이다.

![image](https://user-images.githubusercontent.com/45442859/128311518-2d3cff43-ec1e-4ca9-9eae-903d25762afb.png)

조합최적화에 접근하는 방법은 크게 두가지가 있는데, exact solution, 즉 optimal한 solution을 찾기 위한 방법이 있고, heuristic을 사용하여 sub-optimal하지만
computational cost 측면에서 이점을 가지는 solution을 찾기 위한 방법이 있다. 이 논문에서 제시한 방법론은 후자에 속하는 접근 방법으로, RL을 사용하였다. 

![image](https://user-images.githubusercontent.com/45442859/128312683-4dabaa4f-13e6-48cc-9801-a1b0ac86ff77.png)

저자는 조합최적화 문제로 분류되는 여러 routing 문제들을 풀기 위해 이 모델을 제시했는데, 이 모델을 설명하기 위해 TSP problem setting을 이용한다고 한다.
모델이 TSP 문제만을 풀 수 있는건 아니며, 문제 세팅마다 약간의 모델 수정이나 환경 세팅을 해줌으로써 다양한 routing 문제를 풀 수 있다고 한다.

# Problem Setting

* Problem instance **s** as a graph with **n** nodes, which are fully connected (including self-connections)
* Each node is represented by feature **x<sub>i</sub>** which is coordinate of node **i**
* Solution is defined as a permutation of nodes **&pi;** = (&pi;<sub>1</sub>,...,&pi;<sub>n</sub>) where 
&pi;<sub>t</sub> &ne; &pi;<sub>t'</sub>
* Stochastic policy for choosing next node p<sub>&theta;</sub>(**&pi;**|s) = **&prod;<sub>t=1</sub>** p<sub>&theta;</sub>(&pi;<sub>t</sub> | s,**&pi;**<sub>1:t-1</sub>)

# Attention Model

* Encoder produces embeddings of all input nodes.
* Decoder produces the sequence **&pi;** of input nodes, one node at a time. Also, the decoder observes a mask to know which nodes have been visited.

