---
title: Dynamic Programming
layout: post
categories:
  - Algorithm Explained
tags:
  - Algorithm
  - Dynamic Programming
---

## 다이나믹 프로그래밍 (DP)

* 메모리 공간을 약간 더 사용하면 연산 속도를 비약적으로 증가시킬 수 있는 방법.
* Top-Down과 Bottom-Up 방식이 있다.
* 대표적인 예로는 피보나치 수열이 있다.

```python

def fibo(x):
    if x == 1 or x == 2:
        return 1
    return fibo(x-1) + fibo(x-2)
```

* n이 커질수록 수행 시간이 기하급수적으로 늘어남. 시간보잡도는 약 O(2<sup>N</sup>).
* 위 연산을 수행하다 보면 동일한 함수가 반복적으로 호출되는데, 연산 횟수를 메모이제이션 기법을 통해 줄일 수 있다.
* 메모이제이션이란 한 번 구한 결과를 메모리 공간에 메모해두고 같은 식을 다시 호출하면 메모한 결과를 그대로 가져오는 기법이다.

```python

d = [0]*100

def fibo(x):
    if x == 1 or x == 2:
        return 1
    if d[x] != 0:
        return d[x]
    d[x] = fibo(x-1)+fibo(x-2)
    return d[x]

```

* 다이나믹 프로그래밍을 이용했을 때 피보나치 수열 알고리즘의 시간 복잡도는 O(N)

## Top-Down

* 큰 문제를 해결하기 위해 작은 문제를 호출

## Bottom-Up

* 작은 문제부터 차근차근 답을 도출

```python

d = [0]*100

d[1] = 1
d[2] = 1
n = 99

for i in range(3,n+1):
    d[i] = d[i-1] + d[i-2]

```

* 특정 문제를 완전 탐색 알고리즘으로 접근했을 때 시간이 매우 오래 걸리면 다이나믹 프로그래밍을 적용할 수 있는지 확인.
* 가능하다면 탑다운 보다는 보텀업 방식을 추천.

## 예제

* 정수 X가 주어질 때 정수 X에 사용할 수 있는 연산은 다음과 같이 4가지이다.

1. X가 5로 나누어떨어지면, 5로 나눈다.
2. X가 3으로 나누어떨어지면, 3으로 나눈다.
3. X가 2로 나누어떨어지면, 2로 나눈다.
4. X에서 1을 뺀다.

연산 4개를 적절히 사용해서 1을 만드려고 한다. 연산을 사용하는 횟수의 최솟값을 출력하시오.

```python

import sys

x = int(input())

d = [0]*30001

for i in range(2,x+1):
    d[i] = d[i-1]+1
    
    if i % 2 == 0:
        d[i] = min(d[i], d[i//2]+1)
    if i % 3 == 0:
        d[i] = min(d[i], d[i//3]+1)
    if i % 5 == 0:
        d[i] = min(d[i], d[i//5]+1)

print(d[x])

```
