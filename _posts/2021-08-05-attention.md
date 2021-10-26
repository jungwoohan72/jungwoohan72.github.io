---
layout: post
title: Attention, Learn to Solve Routing Problems!
comments: true
categories:
  - Papers Explained
tags:
  - RL
  - Routing
---
Paper Link: [https://arxiv.org/abs/1803.08475](https://arxiv.org/abs/1803.08475)  
Source Code (Official): [https://github.com/wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)

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
* Stochastic policy for choosing next node p<sub>&theta;</sub>(**&pi;**ㅣs) = **&prod;<sub>t=1</sub>** p<sub>&theta;</sub>(&pi;<sub>t</sub>ㅣs,**&pi;**<sub>1:t-1</sub>)

# Attention Model

* Encoder produces embeddings of all input nodes.
* Decoder produces the sequence **&pi;** of input nodes, one node at a time. Also, the decoder observes a mask to know which nodes have been visited.

<p align="center">
    <img src = "https://user-images.githubusercontent.com/45442859/128447036-ce112ed4-3a59-472d-ba62-e5ffe31c7025.png" alt = "attention" width = "75%" height = "75%"/>
</p>

Attention은 seq-to-seq 모델에 많이 쓰이는데, 한 문장을 다른 언어로 번역하는 예가 대표적이다. 즉 특정 단어를 output으로 내기 위해 어떤 input 단어들에 "집중" 할 것인지 결정하는 게 attention mechanism이라고 
생각하면 될 것 같다.

## Encoder

<p align="center">
    <img src = "https://user-images.githubusercontent.com/45442859/128447359-d68e4783-8ddc-4522-95e3-d6a31f3d6863.png" alt = "encoder" width = "75%" height = "75%"/>
</p>

* Input은 각 노드의 좌표 (2-dimensional)
* Output은 여러 Multi-Head-Attention layer를 거친 embedding vector (128-dimensional)
* 각 노드의 embedding과 더불어 노드들의 평균을 낸 aggregated embedding도 output으로 내줌.

### 각 Attention layer는 아래와 같이 구성

1. 일단 Raw Input이 MLP를 거치고 나면 128-dimensional Embedding이 만들어짐. (첫번째 초록색 화살표)
2. Embedding에 Weight matrix를 곱해서 (query, key, value) set을 만듬. Multi-Head Attention이라고 불리우는 이유는 좀 더 다양한 feature들을 고려하기 위해 (query, key, value) set을 생성할 때 
dimension을 쪼개기 때문이다. 예를 들면 Single Head Attention으로 128x128 weight matrix를 사용해 128-dimensional vector로 project 해주는 대신에 8개의 16x128 weight matrix를 사용해서 16-dimensional vector 8개를 만들어
나중에 합친다.

<p align="center">
    <img src = "https://user-images.githubusercontent.com/45442859/128448124-29776d0f-6f63-42c8-a1b1-8383469d0063.png" alt = "query" width = "50%" height = "50%"/>
</p>

3. 기준이 되는 node의 query와 나머지 주변 node들의 key끼리 dot-product를 해줘서 compatibility를 계산. 예를 들면, 1번 노드에게 나머지 노드들이 얼마나 의미를 가지는가 하는
점수를 계산해주는 과정. 너무 멀리 떨어져 있는 node의 경우 아래와 같이 처리.

<p align = "center">
    <img src = "https://user-images.githubusercontent.com/45442859/128448677-58382d71-5595-4249-a494-8106ec025a9b.png" alt = "MHA" width = "75%" height = "75%"/>
</p>

<p align = "center">
    <img src = "https://user-images.githubusercontent.com/45442859/128448736-5aa89b09-1dc6-4d0e-b037-bacb7d209352.png" alt = "u" width = "50%" height = "50%"/>
</p>

4. 계산된 compatibility에 softmax function을 씌워서 normalize 시켜준 값을 attention score로 씀.

<p align = "center">
    <img src = "https://user-images.githubusercontent.com/45442859/128448860-8dc3d6a9-d875-4640-8118-067328e00cb2.png" alt = "a" width = "25%" height = "25%"/>
</p>

5. 각 attention score는 각 노드의 value vector와 곱해져서 전부 더해짐.

<p align = "center">
    <img src = "https://user-images.githubusercontent.com/45442859/128448936-b8c8e0f6-c512-435a-b052-bbbc978bc3db.png" alt = "h" width = "25%" height = "25%"/>
</p>

6. Multi-Head Attention인 경우 위의 h'<sub>i</sub> vector는 16x1의 크기를 가진다. 앞에서 말했듯이 이 같은 8개의 vector에 128x16 weight matrix를 곱해주어 모두 더해서 최종적으로
128x1 Embedding vector를 만들어 낸다.

<p align = "center">
    <img src = "https://user-images.githubusercontent.com/45442859/128452206-936ed6eb-f3d5-413f-a2cc-4d7f1e98a835.png" alt = "MHA_sig" width = "75%" height = "75%"/>
</p>

위 과정이 하나의 Attention layer에서 일어나는 일이다. 

### Attention Layer를 통과하고 난 다음의 Feed-Forward Layer는 단순하게 ReLU와 Batch Normalization으로 이루어짐.

<p align = "center">
    <img src = "https://user-images.githubusercontent.com/45442859/128452744-f8bc1fb8-be4e-40f4-9e4d-d10641048b59.png" alt = "BN" width = "75%" height = "75%"/>
</p>

<p align = "center">
    <img src = "https://user-images.githubusercontent.com/45442859/128452912-436ec81e-11eb-4c53-b978-3ec542a2e70a.png" alt = "FF" width = "75%" height = "75%"/>
</p>

## Decoder