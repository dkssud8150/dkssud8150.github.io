---
title:    "Coding Test[Python] - 스택 / 큐"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-01-05 12:00:00 +0800
categories: [Classlog,coding test review]
tags: [coding-test, push&pop]
toc: True
comments: True
math: true
mermaid: true
#image:
#  src: /commons/devices-mockup.png
#  width: 800
#  height: 500
---

프로그래머스에 올라와 있는 코딩테스트를 진행하면서 공부한 내용 및 막혔던 부분들에 대해 리뷰하고자 합니다. 저는 거의 대부분을 python을 사용하였습니다. 

이 글은 python을 통한 스택/큐에 대해 리뷰하고 다른 것들을 참고하시려면 아래 링크를 참고해주세요

* [힙](https://dkssud8150.github.io/classlog/codingtestheapq.html)
* [해시](https://dkssud8150.github.io/classlog/codingtesthash.html)
* [정렬](https://dkssud8150.github.io/classlog/codingtestsort.html)
* [완전탐색](https://dkssud8150.github.io/classlog/codingtestsearch.html)
* [연습문제]
* [카카오]

<br>

## Overview

#### 스택
스택(stack)은 데이터 구조 중 하나로 데이터 삽입은 push, 데이터 추출은 pop이라 한다. LIFO(Last In First Out)는 후입선출 구조, 즉 후에 들어온 것이 먼저 나가는 방식이다. 스택은 방문기록 뒤로가기, 역순 문자열, 실행 취소 등에 활용된다.

파이썬에서는 push와 pop을 각각 append, pop 으로 되어 있다.


#### 큐
반대로 큐(queue)는 데이터 구조 중 하나로 데이터 삽입은 인큐(enqueue), 제거는 디큐(deque)라고 한다. FIFO(First In First Out) 선입선출 구조, 즉 먼저 들어온 것이 먼저 나가는 방식이다. 큐는 은행 업무, 프린터 인쇄 대기열, 캐시 등에 활용된다. 

큐는 너비 우선 탐색(BFS)에 주로 사용된다. 

큐 연산을 구현하는 방법으로 3가지 정도가 있다.
1. list 자료형 사용
2. Collections 모듈의 deque 사용
3. queue 모듈의 Queue 클래스 사용

나의 경우 list를 많이 사용하기 때문에 list를 위주로 리뷰할 것이다. list 자료형 사용의 경우 append, pop, insert를 많이 사용한다. 이 방법의 단점은 성능이 비교적 좋지 않다.

* [참고 블로그](https://blog.naver.com/dsz08082/222559458327)



### INDEX
1. 기능개발
2. 프린터
3. 다리를 건너는 트럭
4. 주식가격

<br>

# 기능 개발

<br>

## 나의 전체 코드

```python
def solution(pro, spe):
    answer = []
    for e in range(100):
        c = []
        for i,(p, s) in enumerate(zip(pro, spe)):
            if pro[i] < 100: pro[i] = p + s
        
        while len(pro) != 0 and pro[0] >= 100:
            c.append([pro.pop(0),spe.pop(0)])

        #print(e,'\tpro:',pro,spe,'\tc:',c)
    
        if len(c) != 0: 
            answer.append(len(c))
            c = []
        
        #print(k,'\n')
    
        if len(pro) == 0: return answer
```

```shell
입력값 〉	[95, 90, 99, 99, 80, 99], [1, 1, 1, 1, 1, 1]
출력 〉	
0 	pro: [96, 91, 100, 100, 81, 100] 	spe:  [1, 1, 1, 1, 1, 1] 	c: []
answer:  [] 

1 	pro: [97, 92, 100, 100, 82, 100] 	spe:  [1, 1, 1, 1, 1, 1] 	c: []
answer:  [] 

2 	pro: [98, 93, 100, 100, 83, 100] 	spe:  [1, 1, 1, 1, 1, 1] 	c: []
answer:  [] 

3 	pro: [99, 94, 100, 100, 84, 100] 	spe:  [1, 1, 1, 1, 1, 1] 	c: []
answer:  [] 

4 	pro: [95, 100, 100, 85, 100] 	spe:  [1, 1, 1, 1, 1] 	c: [100]
answer:  [1] 

5 	pro: [96, 100, 100, 86, 100] 	spe:  [1, 1, 1, 1, 1] 	c: []
answer:  [1] 

6 	pro: [97, 100, 100, 87, 100] 	spe:  [1, 1, 1, 1, 1] 	c: []
answer:  [1] 

7 	pro: [98, 100, 100, 88, 100] 	spe:  [1, 1, 1, 1, 1] 	c: []
answer:  [1] 

8 	pro: [99, 100, 100, 89, 100] 	spe:  [1, 1, 1, 1, 1] 	c: []
answer:  [1] 

9 	pro: [90, 100] 	spe:  [1, 1] 	c: [100, 100, 100]
answer:  [1, 3] 

        .
        .
        .

18 	pro: [99, 100] 	spe:  [1, 1] 	c: []
answer:  [1, 3] 

19 	pro: [] 	spe:  [] 	c: [100, 100]
answer:  [1, 3, 2] 
```

전에는 다소 비효율적인 코드를 작성했다. 하지만, zip+enumerate 와 pop을 사용하게 되면서 좀 더 쉽게 풀었다.

<br>

<br>

### pop

pop함수란 리스트 요소를 꺼내는 함수로, `pop()`으로만 사용하게 되면 맨 마지막 요소를 꺼내고 리스트에서 그 요소를 삭제한다. 

```python
>>> a = [1,2,3]
>>> a.pop()
3
>>> a
[1, 2]
```

pop(x) 를 사용하게 되면 x번째에 있는 요소를 꺼내겠다는 것이다.
```python
>>> a = [1,2,3]
>>> a.pop(1)
2
>>> a
[1, 3]
```

* [응용]

```python
>>> while len(pro) != 0 and pro[0] >= 100:
    c.append([pro.pop(0),spe.pop(0)])
    print(c)
[100]
[100,120]
...
```

while과 함께 사용하여 조건에 만족할 동안 맨 앞에 있는 요소를 계속 추출하고, 다른 list에 입력시킨다.



pop과 반대로 push가 있는데 나의 경우 push는 잘 사용하지 않고, append를 많이 사용한다.


<br>

<br>

# 프린터

priorities 각각의 숫자와 위치를 구분하기 위해 문자를 추가하려고 했는데, 사실은 그냥 숫자를 부여해주면 된다. dict나 tuple 다 상관없지만, 이를 구분해주는 방법이 for문이 아니라 enumerate를 사용했으면 더 편했다. 그러나 위치를 나타내는 방법이 문자이면 dict가 편하지만, 숫자면 tuple이 더 간편하다.

예를 들어, 내가 찾고자 하는 것이 apple이라 하면 dict['apple'] 을 통해 맞는 위치를 찾으면 되지만, 숫자라면 서로 구분이 되지 않기 때문에, tuple(1)을 통해 묶어준 튜플에서 알맞는 값을 불러온다. 하지만 나는 풀 당시 dict만을 생각하고 풀었기 떄문에 어려움을 겪었다.

또한, 하나의 원소로 리스트 각각의 값과 비교하는 방법을 몰라서 어렵게 했지만, any()라는 함수를 사용하면 된다.

<br>
## 나의 전체 코드

```python
def solution(priorities, location):
    answer = 0
    process = [(i,p) for i,p in enumerate(priorities)]
    
    while True:
        testp = process.pop(0)
        if any(testp[1] < p[1] for p in process):
            process.append(testp)
        else:
            answer += 1
            if testp[0] == location: return answer
```

```shell
입력값 〉	[2, 1, 3, 2], 2
출력 〉	
process:  [(0, 2), (1, 1), (2, 3), (3, 2)]

testprocess:  (0, 2)
testprocess:  (1, 1)
testprocess:  (2, 3)

```


도저히 너무 길게만 짜져서 다른 사람 풀이를 참고하여 만들었다.

<br>

### enumerate

`for i,k in enumerate(list):` 의 형태로 사용하는데, 순서와 리스트의 값을 전달해주는 기능을 한다. 리스트 이외에 튜플, 문자열 등도 가능하다. 

```python
>>> process = [(i,p) for i,p in enumerate(priorities)]
>>> print(process)
process:  [(0, 2), (1, 1), (2, 3), (3, 2)]
```

이와 같은 형태로 출력이 된다. 즉 i는 0,1,2,3... p는 순서대로 list의 각 요소이다.

* 참고 블로그: https://wayhome25.github.io/python/2017/02/24/py-07-for-loop/

<br>

### any()

`any(iterablevalue)`는 전달받은 자료형의 element 중 하나라도 True일 경우 True를 돌려주고, 그렇지 않은 경우 False를 돌려준다. 빈 자료일 경우도 False를 반환한다.


```python
>>> a = [True,False,True] 
>>> any(a) 
True
```

* [참고 블로그](https://technote.kr/241)

<br>

### startwith()

any()와 비슷하게 True or False 를 반환해주는 함수가 있다. 대소문자를 구분하고, 인자값에 있는 문자열이 string에 있으면 True, 없으면 False를 반환한다. 인자값으로는 tuple만 사용가능하다. 따라서 tuple안의 요소 중 하나라도 string에서 시작하는 문자열과 동일하다면 True를 반환한다.

```python
string = "hello startswith"
print(string.startswith("hello"))
>>> True
```

* [참고 블로그](https://security-nanglam.tistory.com/429)

<br>

# 다리를 지나는 트럭

이 문제의 경우 선입선출이므로 큐 문제에 해당한다. list 자료형을 사용할 것이다.

```python
def solution(bridge_length, weight, truck_weights):    
    arrive = [] 
    ing = []
    truck = [[t,0] for t in truck_weights]
    w = 0
    sec = 0
    
    while len(arrive) != len(truck_weights):        
        for i in range(len(ing)): ing[i][1] += 1
        
        if len(ing) != 0 and ing[0][1] >= bridge_length: 
            w -= ing[0][0]
            arrive.append(ing.pop(0)[0])
        
        if len(truck) != 0 and w + truck[0][0] <= weight and len(ing) <= bridge_length-1:
            w = w + truck[0][0]
            ing.append(truck.pop(0))
            
        sec += 1
        #print("truck: {}\nw: {}\ning: {}\t arrive: {}\nsec: {}\n".format(truck,w,ing,arrive, sec))
    return sec
```

```shell
입력값 〉	2, 10, [7, 4, 5, 6]
출력 〉	
truck: [[4, 1], [5, 1], [6, 1]]
w: 7
ing: [[7, 1]]	 arrive: []
sec: 1

truck: [[4, 1], [5, 1], [6, 1]]
w: 7
ing: [[7, 2]]	 arrive: []
sec: 2

truck: [[5, 1], [6, 1]]
w: 4
ing: [[4, 1]]	 arrive: [7]
sec: 3

truck: [[6, 1]]
w: 9
ing: [[4, 2], [5, 1]]	 arrive: [7]
sec: 4

truck: [[6, 1]]
w: 5
ing: [[5, 2]]	 arrive: [7, 4]
sec: 5

truck: []
w: 6
ing: [[6, 1]]	 arrive: [7, 4, 5]
sec: 6

truck: []
w: 6
ing: [[6, 2]]	 arrive: [7, 4, 5]
sec: 7

truck: []
w: 0
ing: []	 arrive: [7, 4, 5, 6]
sec: 8
```

일단 지나가는 중인 리스트, 도착 리스트를 따로 구성하여 조건에 만족할 때 해당 리스트로 저장한다. w는 다리 위의 트럭들의 무게, sec는 초에 해당한다. 각 조건에 맞게 if문을 구성하였고, 해당 truck이 모두 지나갈 때까지 이므로 while을 사용했다.


<br>

<br>

# 주식가격

이 문제는 지문이 조금 이해하기 어려웠다. 내가 이해하기로는 초 단위로 주어진 prices일 때 구매를 한다고 가정할 때, 산 가격보다 더 떨어지지 않는 기간이 얼마인가를 출력해야 하는 것 같다. 

```python
def solution(prices):
    answer = [0]*len(prices)
    
    for i in range(len(prices)):
        k = prices[i]
        for j in prices[i+1:]:
            answer[i] += 1
            print(k,j)
            if k > j: break
    return answer
```

```shell
입력값 〉	[2, 2, 5, 8, 13, 1]
출력 〉	
k: 1 - answer: [4]
k: 2 - answer: [4, 3]
k: 3 - answer: [4, 3, 1]
k: 2 - answer: [4, 3, 1, 1]
k: 3 - answer: [4, 3, 1, 1, 0]
```

정확성은 통과했지만 효율성은 통과하지 못했다. 아마 for문이 중첩되어 그런 듯하다.

<br>


