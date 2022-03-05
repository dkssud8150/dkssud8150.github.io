---
title:    "[데브코스] 1주차 - 알고리즘 문제 풀기 (1)"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-02-15 12:00:00 +0800
categories: [Classlog, devcourse]
tags: [algorithm, devcourse]
toc: True
comments: True
---

> 코드에 대한 주석들은 추후에 달도록 하겠습니다. 현재는 블로그 정리만 하려고 올리는 것입니다.

<br>

# 자료구조와 알고리즘

## 자료구조
- 문자열 (str) "this is a string"
- 리스트 (list) [4,9,2,7]
- 사전 (dict) {'a':2, 'bc',:6}
- 순서쌍 (tuple), 집합 (set), ...

해결할 문제에 대한 적합한 구조를 사용하는 것이 중요하다.

## 알고리즘

- 사전적 정의: 어떤 문제를 해결하기 위한 절차, 방법, 명령어들의 집합
- 프로그래밍적 의미: 주어진 문제의 해결을 위한 자료구조와 연산 방법에 대한 선택

해결하고자 하는 문제에 따라 최적의 해법이 서로 다르기 때문에 최적의 방법을 찾기 위해 자료구조를 이해해야 한다.

<br>

<br>

# 선형 배열

배열이란 원소들을 순서대로 늘어놓은 것이다. 이는 인덱스가 부여되는데, 0부터 시작한다. 파이썬에서는 리스트를 사용한다. 다른 언어들은 같은 타입의 원소만으로 구성되야 하지만, 파이썬은 다른 타입의 원소들로도 구성이 가능하다.

리스트는 대괄호`[]`, 튜플은 소괄호`()`, 딕셔너리는 중괄호`{}` 이다.

리스트 연산
- 원소 덧붙이기: x.append()
- 끝에서 꺼내기: x.pop()

이들은 원소의 길이와 상관없이 맨 앞 또는 맨뒤에 것들에 대한 작업을 진행한다. 이처럼 빠르게 할 수 있는 작업을 리스트의 길이와 무관한 것을 O(1)이라 한다.

<br>

- 원소 삽입하기: x.insert(3,65) -> 3 위치에 원소 65를 삽입
- 원소 삭제하기: del(x[2]) -> 2 위치에 원소를 삭제

이들은 원소의 길이가 길수록 시간이 오래걸린다. 이처럼 리스트의 길이가 길면 오래 걸리는 작업을 리스트의 길이에 비례(선형 시간) 이를 O(n)이라 한다.

<br>

pop과 del의 차이는 추출한 원소를 저장하느냐 하지 않느냐의 차이다. a = x.pop(2) 처럼 pop을 통해 x의 2인덱스의 원소를 a에 저장할 수 있다.

- 원소 탐색하기: x.index(65)

<br>

<br>

# 배열 - 정렬과 탐색

## 정렬

- sorted(x,reverse=True) : 정렬된 새로운 리스트를 얻어내어 다른 곳에 저장 가능
- x.sort(reverse=True) : 해당 리스트를 정렬

default값이 오름차순인데, 내림차순하고 싶은 경우 `reverse = True`, 문자열로 이루어진 리스트의 경우 알파벳 순서를 따르고, 길이 순서로 정렬하고 싶은 경우 key를 사용
```python
sorted(x, key=lambda x: len(x))
```

<br>

## 탐색

- 선형 탐색 : 탐색이 오래걸리는 O(n)에 해당

- 이진 탐색 : 리스트를 절반씩 줄여가며 탐색

이진탐색은 리스트가 크기순으로 되어 있다는 것을 가정하기에 탐색하려는 리스트를 정렬하고나서 사용가능하다. 이진탐색은 O(log n)에 비례하는 복잡도를 가진다. 리스트를 반으로 자르고, 계속 반으로 자르는 재귀함수를 사용해야 할 것이다.

<br>

<br>

# 재귀 알고리즘(recursive algorithm) 기초

재귀함수(recursive algorithm)란 하나의 함수에서 자신을 다시 호출하여 작업을 수행하는 것

> e.g. 이진 트리(binary trees)
> 왼쪽 서브트리의 원소들은 모두 작거나 같은 것, 오른쪽 서브트리의 원소들은 모두 큰 것으로 이루어져 있다.

재귀 함수는 종결 조건을 설정하는 것이 매우 중요하다.

- recursive(재귀) version Vs iterative(반복) version
복잡도 측면에서 보면 두 방법이 O(n)으로 같다. 하지만 효율성에 대해서는 재귀가 더 좋다.

<br>

<br>

# 재귀 알고리즘 - 응용

## 조합의수

조합의 수 : n개의 서로 다른 원소에서 m개를 택하는 경우의 수

이는 특정한 하나의 원소의 입장에서 볼 때, 이 원소를 포함하는 경우와 그렇지 않은 경우를 따로 계산해서 더하는 것이다.

이것은 재귀로서

```python
def combi(n,m):
    return combi(n-1,m) + combi(n-1,m-1)
```

로 표현이 가능하다. 하지만 여기서 종결 조건이 없기 때문에 이를 설정하여 다시 표현하면

```python
def combi(n,m):
    if n == m: return 1
    elif m == 0: return 1
    else: return combi(n-1,m) + combi(n-1,m-1)
```

<Br>

## 하노이탑

시간이 날 때, 재귀함수로 짜보고자 한다.

<br>

<br>

# 알고리즘의 복잡도

- 시간 복잡도 (Time Complexity)

문제의 크기와 이를 해결하는 데 걸리는 시간 사이의 관계

- 공간 복잡도 (Space Complexity)

문제의 크기와 이를 해결하는 데 필요한 메모리 공간 사이의 관계

이번 강의에서는 시간 복잡도에 대해서만 강의한다.

<br>

시간 복잡도에도 종류가 나뉜다.

- 평균 시간 복잡도 (Average Time Complexity)

임의의 입력 패턴을 가정했을 때 소요되는 시간의 평균

- 최악 시간 복잡도 (Worst-case Time Complexity)

가장 긴 시간을 소요하게 만드는 입력에 따라 소요되는 시간

<br>

## Big-O Notation

점근 표기법(asymptotic notation)의 하나로 어떤 함수의 증가 양상을 다른 함수와의 비교로 표현하는 것이다. 알고리즘의 복잡도를 표현할 때 흔히 쓰인다.

O(logn), O(n), O(n^2), O(2^n) 등으로 표기한다. 이는 입력의 크기가 n일 때,

- O(logn): 입력의 크기의 로그에 비례하는 시간 소요
- O(n): 입력의 크기에 비례하는 시간 소요

<br>

- O(n) - 선형 시간 알고리즘

n개의 무작위로 나열된 수에서 최댓값을 찾기 위해 선형 탐색 알고리즘을 적용한다. 여기서 최댓값을 찾기 위해서는 끝까지 다 살펴보기 전까지는 알 수 없게 된다. 따라서 O(n)에 해당한다.

- O(logn) - 로그 시간 알고리즘

n개의 크기 순으로 정렬된 수에서 특정 값을 찾기 위해 이진 탐색 알고리즘을 적용한다.

- O(n^2) - 이차 시간 알고리즘

삽입 정렬이 있다. 알고리즘이 복잡하기에 간단하게만 정리한다. 알고리즘의 실행 시간이 n의 제곱에 비례하기 때문에 10개를 푸는 시간이 s 라면, 100개를 푸는 시간은 100 * s 가 된다.

- O(nlogn) - 병렬 정렬

정렬할 데이터를 반씩 나누어 각각을 정렬시킨다. 이는 O(logn), 정렬된 데이터를 두 묶음씩 합친다. 이는 O(n). 따라서 O(nlogn)이 된다.

<br>

<br>

# 연결 리스트 (Linked Lists) (1)

## 추상적 자료구조 (Abstract Data Structures)

- Data: 정수, 문자열, 레코드 ...
- A set of perations: 삽입, 삭제, 순회, 정렬, 탐색

1개의 노드에는 값인 `data`와 다음을 가르키는 `link(next)` 값이 포함되어 있다. 노드는 숫자뿐만 아니라 문자, 레코드 등 다양하게 들어올 수 있다. 또한, 연결 리스트에는 기본적으로 처음과 끝을 가르키는 head와 tail, 총 길이를 가르키는 # of nodes가 있어야 한다.

초기 node로서 코드는

```python
class Node:
    def __init__(self, item):
        self.data = item
        self.next = None
```

그 후 연결 리스트를 구성하기 위해

```python
class LinkedList:
    def __init__(self):
        self.nodeCount = 0
        self.head = None
        self.tail = None
```

이 연결 리스트를 이용한 연산들에는

1. 특정 원소 위치 참조
2. 리스트 순회
3. 길이 얻어내기
4. 원소 삽입
5. 원소 삭제
6. 두 리스트 합치기

가 있다. 

대부분의 연결 리스트에서 인덱스는 0이 아닌 1부터 시작한다. 그 이유는 0번을 head로 두기 때문이다.

특정 원소 참조하는 코드(특정 위치의 노드)는

```python
def getAt(self, pos):
    if pos <= 0 or pos > self.nodeCount: return None
    i = 1
    current = self.head
    while i < pos:
        current = current.next
    i += 1
    return current
```

<br>

일반 배열과 비교를 해보면

저장 공간:

- 배열: 연속된 위치
- 연결리스트: 임의의 위치

특정 원소 지칭:

- 배열: index를 통한 매우 간편함
- 연결리스트: 선형탐색과 유사하게 처음부터 다 탐색해야 함

따라서 배열은 O(1), 연결리스트는 O(n)의 복잡도를 가지지만, 다른 장점이 있기 때문에 사용한다.

<br>

<br>

# 연결 리스트 (Linked Lists) (2)

이번에는 위의 1~6중에 4~6에 대해 알아보자. 

## 연결 리스트 연산 - 원소의 삽입

```python
def insertAt(self, pos, newNode):
    prev = self.getAt(pos - 1)
    newNode.next = prev.next
    prev.next = newNode
    self.nodeCount += 1
```

이 때, pos는 1 <= pos <= nodeCount + 1 의 범위 안에 있어야 하므로 pos에 대한 조건을 정의해야 한다.

두 가지 주의 사항이 존재하는데

1. 삽입하려는 위치가 리스트 맨 앞일 때
- prev 없음
- Head 조정 필요
2. 삽입하려는 위치가 리스트 맨 끝일 때
- Tail 조정 필요

따라서 코드는

```python
def insertAt(self, pos, newnode):
    if pos < 1 or pos > self.nodeCount + 1: 
        return False
    
    if pos == 1:
        newnode.next = self.head
        self.head = newnode
    
    elif pos == self.nodeCount + 1: 
        self.tail = newnode
    
    else:
        if pos == self.nodeCount + 1:
            prev = self.tail # 맨 뒤 일 경우 처음부터 읽을 필요없이 tail만 사용
        else:
            prev = self.getAt(pos - 1)
        newnode.next= prev.next
        prev.next= newnode

    if pos == self.nodeCOunt + 1:
        self.tali = newNode

    self.nodeCount += 1
    return True
```

이 때, 맨 앞에 삽입하는 경우는 O(1), 맨 끝에 삽입하는 경우도 O(1), 중간에 삽입하는 경우가 O(n)이 된다.

<br>

## 연결 리스트 연산 - 원소의 삭제

def popAt(self, pos):

인데, 이때도 주의 사항은

1. 삭제하려는 node가 맨 앞의 것일 때
- prev 없음
- head 조정 필요
2. 리스트 맨 끝의 node를 삭제할 때
- tail 조정 필요

맨 끝의 node를 삭제하는 경우 그 전의 pos -1에 대한 값을 가져올 방법이 없으므로 다 순회해야 한다. 따라서 맨 앞에 삽입하는 경우는 O(1), 중간 또는 맨 끝에 삽입하는 경우는 O(n)에 해당한다.

<br>

## 연결 리스트 연산 - 두 리스트의 연결

```python
def concat(self,L):
    self.tail.next = L.head
    if L.tail:
        self.tail = L.tail
    
    self.nodeCount += L.nodeCount
```

이때는 L이 None이 아닌지 판단해야 하기 때문에, if를 삽입했다.

<br>

<br>

# 연결리스트 (Linked Lists) (3)

매번 할 때마다 getAt를 통해 리스트를 순회하는 것은 너무 불필요하다. 따라서 insertAt,popAt가 아닌 insertAfter(prev, newnode),popAfter를(prev) 통해 이전의 값을 넣으면 그 다음에 노드를 지칭하도록 만들고자 한다. 그러기 위해 0 index에 None인 dummy node를 만들어준다.

```python
class LinkedList:
    def __init__(self):
        self.nodeCount = 0
        self.head = Node(None)
        self.tail = None
        self.head.next = self.tail
```

이를 통해 dummy node를 생성하고, 그 노드가 원래의 1 index를 가르키도록 만든다.

## 리스트 순회

```python
def traverse(self):
    answer = []
    curr = self.head
    if curr.data == None: return answer
    
    answer.append(curr.data)
    while curr.next != None:
        curr = curr.next
        answer.append(curr.data)
    return answer

''' ---- dummy 추가 ---- '''

def traverse(self):
    answer = []
    curr = self.head
    while curr.next:
        curr = curr.next
        answer.append(curr.data)
    return answer
```

<br>

## k번째 원소 얻기

```python
def getAt(self, pos):
    if pos <= 0 or pos > self.nodeCount: return None
    i = 1
    current = self.head
    while i < pos:
        current = current.next
        i += 1
    return current

''' ---- dummy 추가 ---- '''

def getAt(self, pos):
    if pos < 0 or pos > self.nodeCount: 
        return None
    
    i = 0
    current = self.head
    while i < pos:
        current = current.next
        i += 1
    return current
```

<br>

## 원소 삽입

```python
def insertAfter(self, prev, newnode):
    newnode.next= prev.next
    if prev.next is None:
        self.tail = newnode
    prev.next= newnode
    self.nodeCount += 1
    return True

```

insertAfter를 활용하여 insertAt을 구현하고자 한다. 이때 고려할 사항들은

1. pos 범위 조건 확인
2. pos == 1인 경우 head 뒤에 새 node삽입
3. pos == nodeCount + 1 인 경우 prev가 tail
4. 그렇지 않은 경우 prev = getAt()

```python
def insertAt(self, pos, newnode):
    if pos < 1 or pos > self.nodeCount + 1: 
        return False

    if pos != 1 and pos == self.nodeCount + 1: 
        prev = self.tail # pos가 1이라면 nodeCount가 0, 즉 빈 리스트이기 때문에 prev가 존재하지 않을 것이다. 따라서 예외 처리

    else: 
        prev = self.getAt(pos - 1)

    return self.insertAfter(prev,newnode)
```

prev가 상황마다 달라지기 때문에 그에 따른 prev를 각각 정의해준 후 insertAfter에 넣은 것이다.

<br>

## 원소 삭제

고려 사항들에는

1. prev가 마지막 노드일 때(prev.next == None)
- 삭제할 노드가 없다.
- return None
2. 리스트의 맨 끝의 노드를 삭제할 때(curr.next == None)
- tail 조정 필요
- 두 리스트 연결

```python
def popAfter(self, prev):
    curr = prev.next
        
    if curr.next is None: 
        self.tail = prev
        
    prev.next = curr.next
    self.nodeCount -= 1
    return curr.data

def popAt(self, pos):
    if pos < 1 or pos > self.nodeCount:
        raise IndexError
        
    if pos == self.nodeCount + 1: 
        return None
        
    if pos == 1:
        prev = self.head
    else:
        prev = self.getAt(pos - 1)
    return self.popAfter(prev)
```

<br>

## 리스트 연결

여기서는 dummy를 뺀 첫번째 값들부터 연결시켜야 한다.

```python
self.tail.next = L2.head.next
self.tail = L2.tail
```

로 연결시켜준다.

```python
def concat(self,L):
    self.tail.next = L.head.next
    if L.tail:
        self.tail = L.tail
    self.nodeCount += L.nodeCount
```

<br>

<br>

# 양방향 연결 리스트

한 쪽 방향으로만 하면 뒤로 돌아오는 것이 불가능하기에 양방향으로 연결하여 이진 탐색 구조로 탐색이 가능하도록 만든다.

```python
class Node:
    def __init__(self,item):
        self.data = item
        self.prev = None
        self.next = None
```

여기서는 처음과 끝에 dummy node를 만들어야 한다.

```python
class DoublyLinkedList:
def __init__(self):
        self.nodeCount = 0
        self.head = Node(None)
        self.tail = Node(None)
        self.head.prev = None
        self.head.next = self.tail
        self.tail.prev = self.head
        self.tail.next = None
```

<br>

## 리스트 역순환

```python
def reverse(self):
    ans = []
    reverse_answer = []
    curr = self.tail
        
    while curr.prev.prev:
        curr = curr.prev
        ans.append(curr.data)
        reverse_answer.insert(0,curr.data) # 리스트를 역순으로 배치하는 것도 생각해보았습니다.
    return ans, reverse_answer
```

<br>

## 원소 참조

양방향 연결 리스트를 활용하여 특정 원소 참조 함수를 정의하면 다음과 같다.

```python
def getAt(self, pos):
    if pos < 0 or pos > self.nodeCount:
        return None

    if pos > self.nodeCount // 2:
        i = 0
        curr = self.tail
        while i < self.nodeCount - pos + 1:
            curr = curr.prev
            i += 1
    else:
        i = 0
        curr = self.head
        while i < pos:
            curr = curr.next
            i += 1

    return curr
```

<br>

## 원소 삽입

```python
def insertAfter(self, prev, newNode):
    next = prev.next
    newNode.prev = prev
    newNode.next = next
    prev.next = newNode
    next.prev = newNode
    self.nodeCount += 1
    return True


def insertBefore(self, next, newNode):
    curr = next.prev
        
    if curr.prev is None:
        self.head.next = newNode
        newNode.prev = self.head
    elif next.next is None:
        self.tail.prev = newNode
        newNode.next = self.tail
        
    curr.next = newNode
    next.prev = newNode    
        
    newNode.prev = curr
    newNode.next = next
        
    self.nodeCount += 1
    
    return True


def insertAt(self, pos, newNode):
    if pos < 1 or pos > self.nodeCount + 1:
        return False

    prev = self.getAt(pos - 1)
    return self.insertAfter(prev, newNode)
```

<br>

## 원소 삭제

```python
def popAfter(self, prev):
    curr = prev.next
    
    if curr.next is None:
        self.tail = prev
        
    prev.next = curr.next
    curr.next.prev = curr.prev
        
    self.nodeCount -= 1
    return curr.data


def popBefore(self, next):
    curr = next.prev
        
    if curr.prev is None:
        self.head = next
        
    next.prev = curr.prev
    curr.prev.next = curr.next
        
    self.nodeCount -= 1
    return curr.data


def popAt(self, pos):
    if pos < 1 or pos > self.nodeCount:
        raise IndexError
        
    if pos == 1:
        prev = self.head
        
    else:
        prev = self.getAt(pos - 1)
            
    return self.popAfter(prev)
```


<br>

## 두 리스트 연결

```python
def concat(self, L):
    self.tail.prev.next = L.head.next
    L.head.next.prev = self.tail.prev
    if L.tail:
        self.tail = L.tail
    
    self.nodeCount += L.nodeCount
    return True
```
