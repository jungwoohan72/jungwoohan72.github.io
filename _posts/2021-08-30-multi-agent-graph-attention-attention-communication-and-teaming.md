---
layout: post
title: Multi-Agent Graph-Attention Communication and Teaming
categories:
  - Papers Explained
tags:
  - Multi agents
  - Task allocation
  - Reinforcement Learning
  - MARL
---
Paper Link: [https://arxiv.org/abs/1911.10715](https://arxiv.org/abs/1911.10715)

# Abstract

* 언제 다른 agent와 협력할 것인가?
* 어떤 agent와 협력할 것인가?
* 어떻게 받아온 message를 처리할 것인가?
* Graph-attention을 기반으로 함.
* Scheduler에서 어떤 agent와 언제 협력할 것인지를 결정. Encoder 단에서 differentiable attention을 사용하여 Message Processor에 differentiable graph를 제공하여 end-to-end 학습을 가능하게 함.
* Message Processor에서 Graph Attention Network를 사용하여 message를 처리

# Introduction

* A novel graph communication protocol that determines "when" and "whom" to communicate via an end-to-end framework.
* Modeling the topology of connections between agents as a dynamic directed graph that accommodates time-varying communication needs and accurately captures the relations between agents.
* 언제 누구와 communicate할지에 대한 정보는 scheduler의 encoder 단에서 directed graph 형태로 가공된다.
* Message processor는 graph attention network를 사용하여 받은 message와 directed graph를 가공한다.
* 가공된 메시지들은 각 agent의 policy 학습에 쓰인다.
* 언제 communicate해야 좋을지에 대한 논문은 있었지만, 누구와 communicate하는지 결정하지 않고 모든 agent와 communicate 했기 때문에 computation 측면에서 낭비가 있었다. -> Learning when to communicate at scale in multiagent cooperative and competitive tasks
* 특정 agent와 협력하는 논문도 있었지만 언제해야되는지에 대한 부분은 다루지 않았었다.

## Contribution

1. Scheduler와 Message Processor로 이루어진 novel graph-attention communication protocol 제시
2. Differentiable graph를 다루기 위해 Message Processor에 GAT 적용. 이 덕분에 end-to-end 학습 가능.
3. 성능 측면에서 Prior method 능가
4. 3대2 축구 게임 세팅에서 실제 실험 진행

# Related Work

* Each agent observes other agents as part of the environment.
* Difficult for each agent to deduce its own contribution to the team's success
* Many MARL algorithms have pursued centralized training and decentralized execution
  * Cooperative scheme without explicit communication channels through centralized critics
  * MADDPG, COMA
* MARL with communication
  * Agents communicate and exchange messages during execution.
  * DIAL builds up limited-bandwidth differentiable discrete communication channels
  * CommNet extends to a continuous communication protocol designed for fully cooperative tasks by receiving averaged encoded hidden states from other agents. In this case, there will be some information loss during averaging process.
  * IC3Net uses a gating mechanism to decide when to communicate, so it can be applied to competitive scenarios.
  * 하지만 CommNet과 IC3Net 같이 hidden feature를 단순 평균을 내는 것은 성능 측면에서 좋지 못함.
  * TarMAC은 when to communicate와 whom to send message를 다루고 있지 않음.
  * ATOC와 SchedNet은 communication group을 manually configure한다는 단점이 있음.
* MARL using GNN
  * DGN employs multi-head dot-production attention
  * MAGNet
  * HAMA
  * G2ANet

# Method

* Partially observable setting of N agents

![image](https://user-images.githubusercontent.com/45442859/131297826-23c3391a-9825-4489-9f44-bb176634e219.png)

1. LSTM to provide hidden state in the beginning

   ![image](https://user-images.githubusercontent.com/45442859/131298069-25b41b89-3cab-4c2c-a277-67479b0b99b8.png)
    
    * e()는 fully connected layer for dimension elevation
    * h는 hidden state from previous time step
    * c는 cell state from previous time step

2. m<sup>t(0)</sup>은 message processing 전의 round 0 message를 의미.
3. Scheduler는 agent가 각 time step에 어떤 agent에게 message를 보내는지 결정
4. Message processor는 받은 message 처리를 맡음
5. Encoded message m<sup>t(l)</sup>은 sub-processor l+1과 sub-scheduler l+1에 각각 forward pass 된다.
6. Sub-scheduler l은 adjacency matrix G<sup>t(l)</sup>을 생성한다. 이 adjacency matrix는 directed graph로 각 agent가 각 time step t 마다 어떤 agent에게 message를 보낼지 결정한다.
7. Sub-processor는 sub-scheduler가 생성한 adjacency matrix 정보를 받아 integrated message set을 생성한다. 각 agent 별로 m<sup>t(l)</sup>으로 표시된 것이 message를 뜻함.
8. m<sup>t(L)</sup>일 경우, 즉 마지막 round of communication일 경우 FC layer를 통과하여 m<sup>t</sup>을 생성한 뒤 agent policy 결정에 쓰인다.
9. l<L 일 경우 m<sup>t(l)</sup>은 Sub-scheduler l+1과 Sub-processor l+1의 input으로 쓰인다. (뒤에 나오는 sub-scheduler에 대한 설명을 보면 sub-scheduler l+1에는 쓰이지 않는 것 같기도...)
10. m<sup>t</sup>은 맨 처음 얻었던 hidden state h<sup>t</sup>와 concatenate 되어 policy head와 value head의 input feature로 사용한다.
11. Policy head는 FC를 씌운 뒤 softmax를 적용한 것이고, time step t에서의 action은 softmax 함수를 통과한 뒤의 distribution에서 sampling을 통해 결정된다.
12. Value head는 single FC layer로 이 알고리즘의 baseline으로 사용된다. (Advantage의 value function)

## Scheduler

* Decides when each agent should send messages and whom each agent should address messages to.

![image](https://user-images.githubusercontent.com/45442859/131336840-cb3f9f1b-9039-4288-a613-51ea85970a47.png)

* 각각의 sub-scheduler는 m<sup>t(0)</sup>을 input으로 받아 directed graph G<sup>t(l)</sup>을 output으로 내놓는다.
* 첫번째 sub-scheduler의 경우만 GAT network를 포함하고 있다. GAT는 Graph Attention Network에 나온 구조 그대로를 사용하였다.
* 각각의 agent에 대한 attention score는 아래와 같이 계산한다.

![image](https://user-images.githubusercontent.com/45442859/131338335-4ef3dee8-982c-47ba-8bda-fd2a5147eeb4.png)

1. Wm = (D'xD)x(Dx1) = D'x1
2. Concatenation -> 2D'x1 <- ||는 columnwise concatenation
3. a<sup>T</sup>Wm = 1x1 <- a의 size: 2D'x1

* E<sup>t</sup><sub>i,j</sub> = (e<sup>t</sup><sub>i</sub> || e<sup>t</sup><sub>j</sub>) -> size: 2D x 1
* 내 생각엔 E<sup>t</sup> matrix는 Nx2DxN이 되야 될 것 같은데... 이건 코드 보면서 자세히 봐보도록 하자.
* 이런 E<sup>t</sup>을 MLP와 Gumbel Softmax에 차례로 통과시키면 G<sup>t(l)</sup>을 얻을 수 있다. 이는 binary value로 이루어진 그래프로 ij번째 값이 1이면 j번째 agent가 i번째 agent에게 t time step의 l번째 라운드에서 메시지를 보낸다는 소리이다.

## Message Processor

* Input: 각 라운드의 directed graph G + 각 라운드의 encoded message m
* Output: processed message m
* 구조: GAT which takes input of {m<sup>t(l-1)</sup><sub>i</sub>}<sup>N</sup><sub>1</sub>

![image](https://user-images.githubusercontent.com/45442859/131343400-b61b1080-7bdc-4bb4-98ce-b101f917a80b.png)

