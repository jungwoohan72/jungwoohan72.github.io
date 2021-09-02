---
layout: post
title: Tasks with Cost Growing over Time and Agent Reallocation Delays
comments: true
categories:
  - Papers Explained
tags:
  - Multi agents
  - Task allocation
---
Multi-robot task allocation with growing completion cost and simultaneity constraints

## Abstract

* Completion costs change predictably over time.
* Important to allocate agents to prevent tasks from growing so much that they become unsolvable.

## Introduction

* Identical homogeneous agents
* Tasks with growing completion costs can become difficult or impossible to complete later.
* Two famous methods for task allocation:
  * Threshold based method: agents individually assess the constraints and their ability to complete each task.
  * Auction based method: market inspired auction methods typically require more communication and are more centralized.
  Agent with largest bidding takes the task.
* Proposed method strikes a balance between distribution and centralization.
  * Each agent is directed to an area by central authority, but upon reaching the destination, agents act on their own logic.
  
## Problem Description

* Agent must be on a task's location in order to apply work.
* More agents than tasks since multiple agents must be assigned to a task.
* Task는 다음과 같이 정의되는 cost를 가지고 있다.
![image](https://user-images.githubusercontent.com/45442859/131081908-fd828cdf-23c0-41dc-98aa-6b59f2cb1720.png)

    * w: work per time unit per agent
    * h: monotonically increasing function
    * n: number of agents working on task i at time t
* If h(f) > w x n, it means that the task is growing faster than the assigned agents can reduce it. Then, the task can never be completed.
* 