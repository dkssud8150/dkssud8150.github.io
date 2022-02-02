---
title:    "Coding Test[Python] - 정렬"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-01-09 12:00:00 +0800
categories: [Classlog,coding test review]
tags: [coding test, sorted]
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

이 글은 python을 통한 힙에 대해 리뷰하고 다른 것들을 참고하시려면 아래 링크를 참고해주세요.

* [스택/큐](https://dkssud8150.github.io/classlog/codingteststack.html)
* [힙](https://dkssud8150.github.io/classlog/codingtestheapq.html)
* [해시](https://dkssud8150.github.io/classlog/codingtesthash.html)
* [완전탐색](https://dkssud8150.github.io/classlog/codingtestsearch.html)
* [연습문제](https://dkssud8150.github.io/classlog/codingtestpra.html)
* [카카오](https://dkssud8150.github.io/classlog/codingtestkakao.html)

<br>

## overview

### 정렬

`list.sort()`는 내림차순으로 정렬하고, `list.sort(reverse=True)`는 오름차순으로 정렬하는 것이다. `list.sort(key=len)`으로 하면, 길이를 기준으로 정렬할 수 있다. 

`list.reverse()`를 사용하게 되면 전체 list를 뒤집는다.

위의 코드들은 출력만 한다. 결과를 반환하고자 한다면, sorted와 reversed를 사용해야 한다.

```python
>>> x = [1 ,11, 2, 3]
>>> s = sorted(x)
>>> s
[1, 2, 3, 11]

>>> r = reversed(x)
>>> y
<list_reverseiterator object at 0x1060c9fd0>
>>> list(y)
[3, 2, 11, 1]
```

reversed를 하면 확인을 위해서 list로 한번 더 변형을 해야 한다.

### Index

1. k번째수
2. 가장 큰 수
3. H-Index

<br>

<br>

# K번째 수

## 나의 전체 코드

```python
def solution(array, commands):
    answer = []
    
    for i in commands:
        start = i[0]; end = i[1]
        arr = array[start-1:end]
        arr = sorted(arr)
        answer.append(arr[i[2]-1])
        #print("start: {}\t end: {}\t arr: {}\t answer: {}".format(start,end,arr,answer))
        
    return answer
```

```python
입력값 〉	[1, 5, 2, 6, 3, 7, 4], [[2, 5, 3], [4, 4, 1], [1, 7, 3]]
출력 〉	
start: 2	 end: 5	 arr: [2, 3, 5, 6]	 answer: [5]
start: 4	 end: 4	 arr: [6]	 answer: [5, 6]
start: 1	 end: 7	 arr: [1, 2, 3, 4, 5, 6, 7]	 answer: [5, 6, 3]
```

<br>

# 가장 큰 수

## 참조한 코드

```python
def solution(numbers):
    answer = ''
    
    numbers = list(map(str,numbers))
    numbers.sort(key= lambda x:x*4, reverse=True)
    
    return str(int(''.join(numbers)))
```

나의 경우 for문을 중복해서 사용했다. 이는 시간 초과를 야기했고, 26점을 받았다. 그래서 다른 사람의 풀이를 보았는데, map이라는 함수와 key를 적절히 사용한 것을 보았다. 생각치도 못했다. 계속 x*4를 만들어주고, 그를 통해 정렬한 후 원래 숫자에 해당하는 길이만큼 잘라주었는데, 반례가 너무 많았다.

### map

map은 리스트의 요소를 지정된 함수로 처리해주는 함수다. 원본 리스트를 변경하지 않고 새 리스트를 생성하게 된다.
- list(map(함수,리스트))
- tuple(map(함수,튜플))

```python
>>> a = [1.2, 2.5, 3.7, 4.6]
>>> a = list(map(int, a))
>>> a
[1, 2, 3, 4]
```

매번 for문을 반복하면서 요소를 변경하지 않아도 되는 편한 함수다.

* [참고 블로그](https://dojang.io/mod/page/view.php?id=2286)

<br>

# H-index

## 나의 전체 코드

```python
def solution(citations):
    answer = 0
    maxs = 0
    citations.sort()
    
    for h in range(max(citations)+1):
        up = [h <= i for i in citations]
        
        count = up.count(True)
        
        if h <= count: maxs = h
        else: return h
        print(h,up,count,maxs)
        
    return maxs
```

```python
입력값 〉	[10, 8, 5, 4, 3]
출력 〉	
0 [True, True, True, True, True] 5 0
1 [True, True, True, True, True] 5 1
2 [True, True, True, True, True] 5 2
3 [True, True, True, True, True] 5 3
4 [False, True, True, True, True] 4 4
5 [False, False, True, True, True] 3 4
6 [False, False, False, True, True] 2 4
7 [False, False, False, True, True] 2 4
8 [False, False, False, True, True] 2 4
9 [False, False, False, False, True] 1 4
10 [False, False, False, False, True] 1 4
```

문제를 이해하기만 하면 풀기 쉬운 문제다.

