---
title:    "Coding Test[Python] - 힙"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-01-06 12:00:00 +0800
categories: [Classlog,coding test review]
tags: [coding-test, heapq]
toc: True
comments: True
math: true
mermaid: true
#image:
#  src: /commons/devices-mockup.png
#  width: 800
#  height: 500
---

프로그래머스에 올라와 있는 코딩테스트를 진행하면서 공부한 내용 및 막혔던 부분들에 대해 리뷰하고자 합니다. 

이 글은 python을 통한 힙에 대해 리뷰하고 다른 것들을 참고하시려면 아래 링크를 참고해주세요

* [스택/큐](https://dkssud8150.github.io/classlog/codingteststack.html)
* [해시](https://dkssud8150.github.io/classlog/codingtesthash.html)
* [정렬](https://dkssud8150.github.io/classlog/codingtestsort.html)
* [완전탐색](https://dkssud8150.github.io/classlog/codingtestsearch.html)
* [연습문제]
* [카카오]

<br>

## overview

### 힙(heap)

힙 모듈은 이진 트리 기반의 최소 힙 자료구조를 제공한다. heapq을 사용하면 원소들이 항상 정렬된 상태로 추가되고 삭제되며, 가장 작은 값이 항상 0에 위치한다. 또한 내부적으로 모든 원소k는 항상 자식 원소들(2k+1, 2k+2)보다 크기가 작거나 같도록 원소가 추가되고 삭제된다.

heapq 라이브러리를 사용하여 heapq.heappush/heapq.heappop 등을 사용하여 추가 및 삭제한다.

### Index

1. 더 맵게
2. 이중우선순위큐
3. 디스크 컨트롤러

<br>

<br>

# 더 맵게

## 나의 전체 코드

```python
import heapq as hq

def solution(scoville, K):
    answer = 0
    hq.heapify(scoville)
    
    while scoville[0] < K:
        if len(scoville) <= 1: 
            break
        a = hq.heappop(scoville)
        b = hq.heappop(scoville)
    
        hq.heappush(scoville, a+2*b)
    
        #print(scoville)
        
        answer += 1
    
    if scoville[0] < K: answer = -1
        
    return answer
```

## heapq

* 리스트를 힙으로 변환

```python
전 scoville list        [9, 8, 7, 6, 5]
heapq.heapify(scoville)
변환 후 scoville list 	[5, 6, 7, 9, 8]
```

이처럼 기존의 리스트를 heap 배열에 맞게 변경할 수 있다.

<br>

* 힙에 원소 추가

```python
전 scoville list        [5, 6, 7, 9, 8]
heapq.heappush(scoville, 4)
후 scoville list 	    [4, 6, 5, 9, 8, 7]
```

원소를 추가할 수 있는데, heap 배열에서는 최솟값이 항상 0에 위치하므로 맨 앞에 추가된 것을 볼 수 있다. 또한, 그에 맞게 heap 배열 형태로 재배치되었다.

<br>

* 힙에 원소 삭제

```python
전 scoville list        [4, 6, 5, 9, 8, 7]
heapq.heappop(scoville)
후 scoville list 	    [5, 6, 7, 9, 8]
```

<br>

* [응용] 최대 힙

블로그에는 다르게 나와 있지만, 나의 아이디어로는 heappop은 최솟값을 삭제하는 것이니 list의 길이-1 만큼 pop을 시키면 최댓값만 남지 않을까하는 생각이다.

```python
heapq.heapify(scoville)
    
while len(scoville) != 1:
    heapq.heappop(scoville)
    
print(scoville)
```

```shell
scoville:   [1, 2, 3, 9, 10, 12]
max value:  [12]
```

<br>

* [응용] K번째 최솟값/최댓값

위의 방법을 사용하여 k번째 최댓값을 찾으면 된다.

```python
k = 4
maxs = 0

while len(scoville) >= k:
    k_maxs = heapq.heappop(scoville)
print('k번째 최댓값: ', k_maxs)
print(scoville)
```

```shell
scoviile: [1, 2, 3, 9, 10, 12] 
k(4)번째 최댓값:  3
scoville: [9, 12, 10]
```

scoville의 길이가 k이상일 때까지만 실행, 즉 k보다 낮아지면 멈춘다는 것이다. 그렇다면 scoville의 길이가 3이 되면 멈추게 되고, 3번 반복했기 때문에 3번째 최솟값이자, 4번째 최댓값이 되는 것이다.

<br>

* [응용] 힙 정렬

```python
k = 0
sort_list = []
for i in range(len(scoviile)):
    k = heapq.heappop(scoville)
    sort_list.append(k)
```

```shell
scoville heap list [5, 6, 7, 9, 8]
scoville sort list [5, 6, 7, 8, 9]
```

* [참고 블로그](https://www.daleseo.com/python-heapq/)

<br>

# 이중우선순위큐

## 나의 전체 코드 

```python
import heapq

def solution(operations):
    answer = []
    s = []
    
    for i in operations:
        num = int(i.split(' ')[1])
        #print("i: {} \tanswer: {} ".format(i, s))
        if i.startswith("I"): heapq.heappush(s,num)
        else:
            if len(s) == 0: pass
            elif num == -1: heapq.heappop(s)
            else:
                if len(s) == 1: heapq.heappop(s)
                else: s.remove(max(s))
    
    if not s: answer=[0,0]
    else: 
        answer.append(max(s))
        answer.append(min(s))
    
    return answer
```

```shell
입력값 〉	["I 16", "I -5643", "D -1", "D 1", "D 1", "I 123", "D -1"]
출력 〉	
i: I 16 	answer: [] 
i: I -5643 	answer: [16] 
i: D -1 	answer: [-5643, 16] 
i: D 1 	answer: [16] 
i: D 1 	answer: [] 
i: I 123 	answer: [] 
i: D -1 	answer: [123] 
```

일단 입력된 operation을 원소마다 판별을 하는데, "I 16"의 경우 str 형태이고, 각각을 비교해야 하므로  ' '를 기준으로 split하고 i가 "I"로 시작할 때와 D로 시작할 때로 경우를 나누고, D에서 1과 -1을 또 분리한다. 그래서 s가 빈 리스트라면 [0,0]을 그렇지 않다면 [최댓값, 최솟값] 을 리턴한다.

<br>

# 디스크 컨트롤러




