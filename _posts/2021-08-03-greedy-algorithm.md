---
layout: post
title: Benchmarking Deep Reinforcement Learning for Continuous Control
categories:
  - Algorithm Explained
tags:
  - algorithm
  - RL
last_modified_at: 2021-08-03T09:42:52-05:00
---

## 그리디 (Greedy) 알고리즘이란?

1. 현재 상황에서 지금 당장 좋은 것만 고르는 방법
   
2. 현재의 선택이 나중에 미칠 영향에 대해서는 고려하지 않음.

3. 간혹 문제에서 '가장 큰 순서대로', '가장 작은 순서대로'와 같은 기준을 암시하는 경우도 있음.

4. 대부분의 문제는 그리디 알고리즘을 사용했을 때 '최적의 해'를 찾을 수 없을 가능성이 다분.
  * 예를 들면, 가지고 있는 동전 중에서 가장 큰 단위가 작은 단위의 배수가 아닌 경우.
  * 위와 같은 경우는 **다이나믹 프로그래밍 (Dynamic Programming)**을 통해 해결할 수 있을 가능성이 있음.

5. 바로 문제 유형을 파악하기 어렵다면 그리디 알고리즘부터 적용해서 정당한지 체크!

## 예제 코드

