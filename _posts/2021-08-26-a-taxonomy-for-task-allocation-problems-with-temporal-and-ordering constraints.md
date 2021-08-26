---
layout: post
title: A taxonomy for task allocation problems with temporal and ordering constraints
categories:
  - Papers Explained
tags:
  - Multi agents
  - Task allocation
---
Paper link: [https://www.sciencedirect.com/science/article/pii/S0921889016306157](https://www.sciencedirect.com/science/article/pii/S0921889016306157)

초심으로 돌아가자. 요즘 너무 RL에 지배당한 느낌이다.

일단 Multi-robot을 운용하려면 task allocation이 필수적인데 최근 떠오른 연구 아이디어가 여러가지 constraint를 고려한 task allocation이라 비 RL 관련 논문들을 좀 찾아 보고자 한다.

주로 찾아봐야 할 내용은 아래와 같다.
* Resource constraint (로봇의 resource가 정해져 있을 때의 효율적인 task allocation)
* Priority considering (작업의 중요도에 따른 순차적인 task allocation)
* Synchronization constraint (여러 대의 로봇이 같이 수행해야 되는 작업)
* Precedence constraint (특정 task를 수행하기 전에 수행되어야 하는 task 목록)

이 논문에서 얻어갈 수 있는 아이디어는 simultaneity constraint와 precedence constraint 정도인 것 같다.

## Multi-robot Task Allocation이란?

* 아마존에서는 1초에 426개의 물건이 판매된다고 한다. 물건을 배송하기 위해서는 물류창고의 로봇이 주문을 받고, 물건을 꺼내오고, 포장하고, 배송 시스템에 전달을 해야한다.
* 과연 로봇 한대만 가지고서 많은 주문을 위 순서대로 빠르게 처리가 가능할까? 대부분 아니라고 답할 것이다.
* 그렇다면 여러대의 로봇을 사용해야 된다는 뜻인데, 어떤 로봇이 어떤 물건을 꺼내올 것인지 정하고, 로봇끼리 충돌 없이 움직일 수 있는 경로를 계획하는 등등 고려해야할 것이 상당히 많아진다.
* 이러한 문제를 해결하는 과정이 Task Allocation 과정이다.

## 논문의 Focus

* Temporal constraint와 ordering constraint가 존재하는 multi-robot task allocation 문제 -> MRTA/TOC(Multi-Robot Task Allocation/Temporal and Ordering Constraint)
  * Temporal constraint: 특정 시점에 수행이 되어야 하는 task 고려
  * Ordering constraint: 특정 순서대로 수행이 되어야 하는 task 고려
* 위 constraint 등을 고려한 objective function을 최적화하는 것이 task allocation의 목적
  * Cost는 총 소요시간인 makespan이 될 수도 있다.
  * 또 다른 옵션으로는 로봇이 움직인 거리 (traveled distance)로 설정할 수 있다.

# Task Allocation vs VRP

* Task allocation을 보면서 제일 먼저 든 생각은 VRP랑 다른 게 뭐지?라는 점이다. Cost를 만약 traveled distance로 하면 VRP랑 똑같아지지 않을까라는 생각이 있었는데, 이 논문에서 이런 점을 다루고 있다.
* VRP와 다른점?
  * VRP는 정해진 vehicle 수가 보통 없는 반면, robotics 도메인에서의 task allocation은 가용 가능한 로봇의 수가 정해져 있다. 그리고 미션을 수행함에 따라 그 숫자가 줄어들 수 있다.
  * VRP의 경우 모든 vehicle이 정해진 depot에서 출발해야하고, 돌아와야 한다. 하지만 robotics 도메인에서 이러한 설정은 일반적이지 않다.
  * 대부분의 VRP 문제는 homogeneous vehicle을 가정한다.
  * 로봇을 사용한 task allocation 문제에서는 로봇 간의 communication이 중요하게 작용한다.
    * S.S. Ponda - Distributed Chance-Constrained Task Allocation for Autonomous Multi-Agent Teams
    * J. Jackson - Distributed Constrained Minimum-Time Schedules in Networks of Arbitrary Topology
    * T. Mercker - An Extension of Consensus-Based Auction Algorithms for Decentralized, Time-Constrained Task Assignment

# Constraints

* Time Window Constraint
  * [earliest start time, latest start time, earliest finish time, latest finish time]으로 표현되기도 한다.
  * 위 같을 경우 time window의 lower boundary는 earliest start time이 되고, upper boundary는 latest finish time이 된다.
  * Deadline constraint 같은 경우는 로봇이 task가 expire되기 전에 task에 도달해야만 하는 constraint를 부여한다.
  * Task allocation with time window constraint는 NP-hard 문제임.

* Precedence and Synchronization Constraints
  * Time window 없이 partial order 혹은 total order에만 제약 조건을 걸어줌. 'A task 전에 B task가 반드시 수행되어야 함'과 같은 constraint
  * Multi-robot을 사용하게 되면 precedence/synchronization constraint에 걸쳐 있는 task들이 서로 다른 로봇에게 할당되는 경우도 생기는데, 이런 경우는 한 로봇이 다른 로봇의 미션 수행에 큰 영향을 끼칠 수 있으므로 undesirable하다.

* Hard vs Soft Temporal Constraint
  * Hard: time window를 못 맞추면 utility function이 0이 됨.
    * 예시: SAR (Search And Rescue) 상황에서 사람이 재난 상황에 처하고, 특정 time window 안에 구조를 하지 못하면 utility function이 0으로 drop. 
  * Soft: time window를 못 맞추면 utility function이 exponentially 감소함. (패널티 부여)