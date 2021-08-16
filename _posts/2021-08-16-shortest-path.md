---
title: Shortest Path
layout: post
categories:
  - Algorithm Explained
tags:
  - Algorithm
  - Shortest path
  - Djikstra
---

## 최단거리 알고리즘

* 그리디 알고리즘과 다이나믹 프로그래밍 알고리즘이 최단 경로 알고리즘에 그대로 적용됨.

1. 다이스트라
2. 플로이드 워셜
3. 벨만 포드

## 다익스트라 최다 경로 알고리즘

* 음의 간선이 없을 때 정상적으로 작동.
* 매번 가장 비용이 적은 노드를 선택해서 임의의 과정을 반복하기 때문에 그리디 알고리즘으로 분류.
* '각 노드에 대한 현재까지의 최단 거리 정보'를 항상 1차원 리스트에 저장하며 리스트를 계속 갱신.
* 과정
    1. 출발 노드 설정
    2. 최단 거리 테이블 초기화
    3. '방문하지 않은 노드 중'에서 최단 거리가 가장 짧은 노드 선택
    4. 해당 노드를 거쳐 다른 노드로 가는 비용을 계산하여 최단 거리 테이블을 갱신
    5. 3번과 4번 과정을 반복

### 간단한 버전

```python

import sys
input = sys.stdin.readline
INF = int(1e9)

n,m = map(int, input().split())
start = int(input())

graph = [[] fo i in range(n+1)]
visited = [False]*(n+1)
distance = [INF]*(n+1)

for _ in range(m):
    a,b,c = map(int, input().split()) # a에서 b로 가는 비용 c
    graph[a].append(b,c)
    
def get_smallest_node(): # 순차탐색
    min_value = INF
    index = 0
    for i in range(1, n+1):
        if distance[i] < min_value and not visited[i]:
            min_value = distance[i]
            index = i 
    return index

def dijkstra(start):
    distance[start] = 0
    visited[start] = True
    
    for j in graph[start]: # 시작 노드와 연결된 노드로 가는 cost 갱신
        distance[j[0]] = j[1] 
        
    for i in range(n-1): # 나머지 노드에 대해서 가장 cost가 작은 노드를 찾고 최단거리 갱신
        now = get_smallest_node()
        visited[now] = True
        for j in graph[now]:
            cost = distance[now] + j[1]
            if cost < distance[j[0]]
                distance[j[0]] = cost
                
dijkstra(start)

for i in range(1, n+1):
    if distance[i] == INF:
        print("INFINITY")
    else:
        print(distance[i])

```

* 시간 복잡도는 O(V<sup>2</sup>). 왜냐하면 O(V)에 걸쳐서 최단 거리가 가장 짧은 노드를 선형 탐색해야 하고, 현재 노드와 연결된 노드를 매번 일일이 확인해야하기 때문.
* 전체 노드의 개수가 5000개 이하라면 괜찮지만, 100000개가 넘어가면 사용 불가.

### 구현은 어렵지만 더 빠른 버전

* 최악의 경우에도 시간 복잡도 O(ElogV) 보장
* 최단 거리가 가장 짧은 노드를 찾기 위해서 O(V)의 시간을 소요했던 것을 개선.

#### 힙 자료구조

* 우선순위 큐를 구현하기 위해 사용하는 자료구조 중 하나.
* 우선순위가 가장 높은 데이터를 가장 먼저 삭제. (Queue는 가장 먼저 삽입된 데이터를 먼저 삭제했던 것과 비슷)
* heapq 사용
* 우선순위 값을 표현할 때는 일반적으로 정수형 자료형의 변수가 사용됨.
* 우선순위 큐 라이브러리에 데이터 묶음을 넣으면, 첫 번째 원소를 기준으로 우선순위를 설정.
* 힙 자료구조에서는 N개의 자료를 삽입(삭제)할 때 O(NlogN)의 연산이 필요. 반면 리스트를 사용하면 삭제할 때 하나의 원소를 삭제할 때 O(N)만큼의 시간이 걸리므로 N개 모두 삭제하려면 O(N<sup>2</sup>) 
만큼의 시간이 걸린다.

```python

import sys
import heapq

input = sys.stdin.readline
INF = int(1e9)

n,m = map(int, input().split())
start = int(input())

graph = [[] fo i in range(n+1)]
visited = [False]*(n+1)
distance = [INF]*(n+1)

for _ in range(m):
    a,b,c = map(int, input().split()) # a에서 b로 가는 비용 c
    graph[a].append(b,c)

def dijkstra(start):
    q = []
    heapq.heappush(q, (0,start))
    distance[start] = 0
    
    while q:
        dist, now = heapq.heappop(q)
        if distance[now] < dist:
            continue
        for i in graph[now]:
            cost = dist + i[1]
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(q,(cost,i[0]))
                
dijkstra(start)

for i in range(1, n+1):
    if distance[i] == INF:
        print("INFINITY")
    else:
        print(distance[i])
```

* E (특정 노드에 연결된 노드의 개수)개의 원소를 우선순위 큐에 넣었다가 모두 빼내는 연산과 매우 유사. 앞에서 말했듯이 힙에 N개의 데이터를 모두 넣고, 이후에 모두 빼는 과정은 O(NlogN)이다.
따라서 간단하게 생가하면 전체 시간 복잡도는 O(ElogE)이다.
* 이때 중복 간선을 포함하지 않는다면, E는 V<sup>2</sup>보다 항상 작다. 따라서 O(logE) < O(logV)이다.
* 그러므로 시간 복잡도는 O(ElogV)이다. 

