---
layout: post
title: 그리디 (Greedy) 알고리즘  
categories:
  - Algorithm Explained
tags:
  - algorithm
  - coding test
last_modified_at: 2021-08-03T09:42:52-05:00
---

Nested and mixed lists are an interesting beast. It's a corner case to make sure that lists within lists do not break the ordered list numbering order and list styles go deep enough.

## Ordered -- Unordered -- Ordered

1. 그리디 (Greedy) 알고리즘이란?
   * **unordered**
    1. 현재 상황에서 지금 당장 좋은 것만 고르는 방법
    2. 
    
2. ordered item
  * **unordered**
  * **unordered**
    1. ordered item
    2. ordered item
3. ordered item
4. ordered item

## Ordered -- Unordered -- Unordered

1. ordered item
2. ordered item
  * **unordered**
  * **unordered**
    * unordered item
    * unordered item
3. ordered item
4. ordered item

## Unordered -- Ordered -- Unordered

* unordered item
* unordered item
  1. ordered
  2. ordered
    * unordered item
    * unordered item
* unordered item
* unordered item

## Unordered -- Unordered -- Ordered

* unordered item
* unordered item
  * unordered
  * unordered
    1. **ordered item**
    2. **ordered item**
* unordered item
* unordered item