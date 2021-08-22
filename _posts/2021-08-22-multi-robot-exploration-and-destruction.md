---
layout: post
title: Multi-Robot Dynamic Task Allocation for Exploration and Destruction
categories:
  - Papers Explained
tags:
  - Multi agents
  - Task allocation
---
  
논문 링크: [https://link.springer.com/article/10.1007/s10846-019-01081-3](https://link.springer.com/article/10.1007/s10846-019-01081-3)

## Abstract

* Multi-Robot Tasks Allocation (MRTA) in exploration and destruction domain where robots cooperatively search for targets hidden in the environment and attempt to destroy them.
* Robots have prior knowledge about the suspicious locations, not the exact locations of the targets.
* Destruction task is dynamically generated along with the execution of exploration task.
* Each robot has different strike ability and each target has uncertain anti-strike ability. Either the robot or target is likely to be damaged in the destruction task.
* Approach via auction-based approach, vacancy chain approach, and deep Q-learning.

## 논문에서 강조하는 점

* Multi-agent Dynamic Task Allocation -> 바뀌는 환경에 따라 Task Allocation이 이뤄져야 함. 예를 들어, hidden target이 발견되면 destruction task를 진행해야 하는 것처럼 기존엔 없던 Task가 생겨서 이를 로봇에게 할당해야 함.
* 대부분의 MATA 문제들은 Search And Rescue (SAR)이나 Delivery 문제 같이 비교적 안전한 문제 세팅에서 이루어짐. 이 논문에서는 confrontational environment에서의 MATA 문제를 다룸.
* Pure exploration이나 routing 문제 같은 것들은 MATA로 포장은 하지만 Task의 위치를 미리 알고 있는 경우에는 mTSP 문제로 generalize 시킬 수 있다.

## 방법론 설명

* Auction-based: 어떤 Task에 대해 각 robot은 'bidding'을 진행함. 그리고 가장 높은 'bidding' 값을 제시한 robot이 해당 task를 가져감. (개인적으로는 이게 heuristic을 사용한 approach랑 다른 게 뭔지 모르겠음...)
* Vacancy Chain: Multi-robot 시스템에서 수행할 task가 없는 로봇이 생기거나 unallocated task가 발견 되면 reallocation을 진행해서 비는 시간을 줄이는 방법.
* Learning-based: Deep Q-Learning 사용

## 시나리오 세팅

* k개의 homogeneous한 로봇이 존재
* 각 로봇마다 P<sub>k</sub>의 strike ability를 가지고 있음. 
* n개의 Hidden target이 존재.
* 각 target은 P<sub>k</sub> tilde로 표기되는 anti-strike ability를 가지고 있음.
* Distributed (decentralized) approach를 사용해서 각각의 robot은 local observation을 토대로 독립적인 decision making을 내림.

## 시나리오 진행 Flow

* 로봇들은 suspicious한 target location l<sub>m</sub> where n < m 을 알고 있음.
* Exploration task가 먼저 수행 되는데, 모든 l<sub>i</sub>을 방문해서 hidden target을 찾는 과정임.
* 모든 suspicious target location을 방문하고 나면 exploration이 종료됨.
* Exploration이 끝나면 hidden target들에 대한 destruction task가 진행됨.
* Target을 발견하고 나면 target의 위치는 알 수 있지만 target의 anti-strike ability는 알 수 없음.
* Destruction 수행 결과는 둘중 하나이다:
  * Robot의 strike ability가 target의 anti-strike ability보다 높아서 target을 없앨 수 있는 경우
  * Robot의 strike ability가 target의 anti-strik abilitiy보다 낮아서 target을 없앨 수 없는 경우. 이 경우 target은 다른 robot에게 reallocate된다. 
* Destruction task는 모든 target을 없앨 때까지 진행된다.

## Auction-based

![image](https://user-images.githubusercontent.com/45442859/130351780-1aa3ea43-0d5f-41d0-9e37-cb0d105a2f43.png)

* Team objective는 로봇들의 total travelling cost를 최소화하는 MINISUM을 objective로 설정하고 진행됨.
* Allocation을 위한 'bidding'은 robot의 현재 위치부터 target까지의 Euclidean distance로 estimate 된다.
* Exploration task에서는 현재 로봇 위치로부터 suspicious target location까지의 거리를 사용. 가장 작은 bidding을 한 robot이 task를 가져감.
* 몇 번의 bidding round를 거쳐서 suspiciouss location 전부를 방문할 수 있도록 initial task allocation을 진행.
* Exploration task를 진행하는 중에 target을 발견하게 되면 destruction task를 바로 진행하게 되는데, 자신의 strike ability보다 강한 target을 만나서 로봇이 손상을 입으면 그 로봇에게 할당되었던 task들은 더 이상 수행 불가하므로 다른 로봇들에게 재할당 됨.
* 위 같은 이유 때문에 dynamic task allocation 문제로 생각해야 함.

![image](https://user-images.githubusercontent.com/45442859/130351803-896c102f-5941-40d1-a1df-44b2bff85397.png)

* exploration 중간에 새로운 allocation이 발생하면 travel cost가 늘어날 수 밖에 없는데, 기존의 allocated된 task들의 순서는 바꾸지 않고, 새로운 task를 기존의 task들 사이에 끼워넣는 형태로 reallocation을 진행한다.
* 옵션들 중에서 가장 travelling cost가 작은 execution sequence를 채택한다.

## Learning-based

* State은 아래와 같이 정의

  ![image](https://user-images.githubusercontent.com/45442859/130351271-95b10c01-599b-4959-84cd-6dadcb74ac3d.png)

  * 왼쪽 Matrix 같은 경우 1은 자기 자신을 뜻하고, 2는 다른 agent를 뜻한다. 그리고 음수의 값들은 task들을 의미한다.
  * 오른쪽 Matrix 같은 경우 1은 자기 자신을 뜻하고, 파란색 점선으로 표시된 0 (allocated) 혹은 -1 (unallocated)은 task의 allocation 상태를 뜻한다. 그리고 빨간색 점선으로 표시된 양수는 해당 로봇까지의 travelling cost를 뜻한다.
  * 이런 state representation을 사용하게 되면 map size가 변하지만 않으면 state matrix 크기를 고정시킬 수 있다는 장점이 있다. 
  * 다만 여기서 조금 애매한 점은 task나 로봇이 아닌 경우 모두 0으로 표기하는데, 이렇게 되면 allocated task나 가까이 있는 다른 agent의 표기와 겹치게 되어 학습에 문제를 줄 수도 있을 것 같다.

* Action은 next allocated task로 정의된다. 따라서 action space는 task의 갯수가 될 것 같다. 
* Action은 매 step 진행되는 것이 아니라 unallocated task가 발생하면 그때마다 action을 수행한다.
* Reward는 아래와 같다.

  ![image](https://user-images.githubusercontent.com/45442859/130351964-9906fe35-0af3-4202-89fa-3168215e7a51.png)
  * c<sub>i</sub>: travel cost of robot i
  * 첫번째 항은 total travelling cost에 대한 penalty term
  * 두번째 항은 makespan에 대한 penalty term
  * 세번째 항은 unbalanced된 traveling cost에 대한 penalty term
  * 네번째 항은 성공적인 allocation에 대한 reward term이다.
  * Reasonable 해 보이지만 조금 더 잘 정의할 수 있을 것 같은데...

## Experimental Result

![image](https://user-images.githubusercontent.com/45442859/130352341-65eda282-de36-4301-9b19-a41c0a17f3e6.png)

* 10 robot / 40 suspicious locations
* 보면 알 수 있듯이 learning-based 방법의 성능이 아주 좋지 않다. Attention! Learn to solve routing problems 논문이 나온 해에 쓰여진 논문이라 뭔가 더 비교되는 거 같은데, 
attention 방법론을 쓰면 auction-based보다 좀 나은 성능을 기대해볼 수 있지 않을까...