---
layout: post
title: 구현 (Implementation)
categories:
  - Algorithm Explained
tags:
  - Algorithm
  - Implementation
---

# 구현이란?

1. 머리 속에 있는 알고리즘을 소스코드로 바꾸는 과정.
2. 코딩에서의 피지컬이 여기에 해당.
3. 완전탐색 (모든 경우의 수를 고려), 시뮬레이션 (문제에서 제시한 알고리즘을 한단계씩 수행) 모두 구현으로 분류

# 예제 문제

문제는 책 p118 참조.

```python
def turn_left():
    global curr
    curr = curr - 1
    if curr == -1:
        curr = 3

n, m = map(int, input().split())
i,j,k = map(int, input().split())

graph = []
for _ in range(m):
    graph.append(list(map(int, input().split())))

curr = k # currently facing direction

dx = [-1,0,1,0]
dy = [0,-1,0,1]

total_turn = 0
ans = 1

graph[j][i] = 1

while True:
    if j+dy[curr] < 0 or j+dy[curr] > n or i+dx[curr] < 0 or i+dx[curr] > m:
        turn_left()
        total_turn += 1
        continue

    try:
        if graph[j + dy[curr]][i + dx[curr]] == 0:
            ans += 1
            graph[j + dy[curr]][i + dx[curr]] = 1
            i = i + dx[curr]
            j = j + dy[curr]
            turn_left()
            total_turn = 0
        else:
            turn_left()
            total_turn += 1

    except IndexError:
        turn_left()
        total_turn += 1

    if total_turn == 4:
        if graph[j - dx[curr]][i + dy[curr]] == 1:
            break
        else:
            i = i + dy[curr]
            j = j - dx[curr]
            total_turn = 0

print(ans)
```

머리 속에서 시뮬레이션을 잘 굴려야 풀 수 있는 문제. 중간에 한번 꼬여서 계속 헤맸다... 문제에선 외곽이 다 바다라는 조건이 주어져 있었지만 조금 더 general하게 외곽이 바다가 아닐 경우
IndexError가 나면 예외처리 해주는 부분도 추가를 해서 코드를 작성해봤다. 예를 들어, 외곽에서 맵 밖으로 나가려고 시도하면 안 되기 때문에 그때는 IndexError를 raise하고 코드를 진행한다.