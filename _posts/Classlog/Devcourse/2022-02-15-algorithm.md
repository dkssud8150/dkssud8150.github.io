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
# math: true
# mermaid: true
# image:
#   src: /assets/img/autodriving/MV3D/pointcloud.png
#   width: 800
#   height: 500
---

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

- 조합의수

n개의 서로 다른 원소에서 m개를 택하는 경우의 수

이는 특정한 하나의 원소의 입장에서 볼 때, 이 원소를 포함하는 경우와 그렇지 않은 경우를 따로 계산해서 더하는 것이다.

이는 재귀로서

def combi(n,m):

return combi(n-1,m) + combi(n-1,m-1)

로 표현이 가능하다. 하지만 여기서 종결 조건이 없기 때문에 이를 설정하여 다시 표현하면

def combi(n,m):

if n == m: return 1

elif m == 0: return 1

else: return combi(n-1,m) + combi(n-1,m-1)

- 하노이탑

재귀함수로 짜보아라

# 6강: 알고리즘의 복잡도

- 시간 복잡도 (Time Complexity)

문제의 크기와 이를 해결하는 데 걸리는 시간 사이의 관계

- 공간 복잡도 (Space Complexity)

문제의 크기와 이를 해결하는 데 필요한 메모리 공간 사이의 관계

이번 강의에서는 시간 복잡도에 대해서만 강의한다.

시간 복잡도에도 종류가 나뉜다.

- 평균 시간 복잡도 (Average Time Complexity)

임의의 입력 패턴을 가정했을 때 소요되는 시간의 평균

- 최악 시간 복잡도 (Worst-case Time Complexity)

가장 긴 시간을 소요하게 만드는 입력에 따라 소요되는 시간

<br>

Big-O Notation

점근 표기법(asymptotic notation)의 하나로 어떤 함수의 증가 양상을 다른 함수와의 비교로 표현( 알고리즘의 복잡도를 표현할 때 흔히 쓰임 )

O(logn), O(n), O(n^2), O(2^n) 등으로 표기한다. 이는 입력의 크기가 n일 때,

- O(logn): 입력의 크기의 로그에 비례하는 시간 소요
- O(n): 입력의 크기에 비례하는 시간 소요

O(n) - 선형 시간 알고리즘

n개의 무작위로 나열된 수에서 최댓값을 찾기 위해 선형 탐색 알고리즘을 적용한다. 여기서 최댓값을 찾기 위해서는 끝까지 다 살펴보기 전까지는 알 수 없게 된다. 따라서 O(n)에 해당한다.

O(logn) - 로그 시간 알고리즘

n개의 크기 순으로 정렬된 수에서 특정 값을 찾기 위해 이진 탐색 알고리즘을 적용한다.

O(n^2) - 이차 시간 알고리즘

삽입 정렬이 있다. 알고리즘이 복잡하기에 간단하게만 정리한다. 알고리즘의 실행 시간이 n의 제곱에 비례하기 때문에 10개를 푸는 시간이 s 라면, 100개를 푸는 시간은 100 * s 가 된다.

O(nlogn) - 병렬 정렬

정렬할 데이터를 반씩 나누어 각각을 정렬시킨다. 이는 O(logn), 정렬된 데이터를 두 묶음씩 합친다. 이는 O(n). 따라서 O(nlogn)이 된다.

<br>

# 7강: 연결 리스트 (Linked Lists)

## 추상적 자료구조 (Abstract Data Structures)

- Data: 정수, 문자열, 레코드 ...
- A set of perations: 삽입, 삭제, 순회, 정렬, 탐색

1개의 노드에는 값인 data와 다음을 가르키는 link(next) 값이 포함되어 있다. 노드는 숫자뿐만 아니라 문자, 레코드 등 다양하게 들어올 수 있다.

연결 리스트에는 기본적으로 처음과 끝을 가르키는 head와 tail, 총 길이를 가르키는 # of nodes가 있어야 한다.

초기 node로서 코드는

```

class Node:

def __init__(self, item):

self.data = item

self.next = None

```

그 후 연결 리스트를 구성하기 위해

```

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

이번 강의에서는 1,2,3만 할 것이다. 대부분의 연결 리스트에서 인덱스는 0이 아닌 1부터 시작한다.

특정 원소 참조하는 코드는

```

def getAt(self, pos):

if pos <= 0 or pos > self.nodeCount: return None

i = 1

current = self.head

while i < pos:

current = current.next

i += 1

return current

```

일반 배열과 비교를 해보면

저장 공간:

- 배열: 연속된 위치
- 연결리스트: 임의의 위치

특정 원소 지칭:

- 배열: index를 통한 매우 간편함
- 연결리스트: 선형탐색과 유사하게 처음부터 다 탐색해야 함

따라서 배열은 O(1), 연결리스트는 O(n)의 복잡도를 가지는데, 다른 장점이 있기 때문에 사용한다.

<br>

# 8강: 연결 리스트 (2)

이번 강의에서는 4,5,6을 할 것이다.

- 연결 리스트 연산 - 원소의 삽입

```

def insertAt(self, pos, newNode):

prev = self.getAt(pos - 1)

newNode.next = prev.next

prev.next = newNode

self.nodeCount += 1

```

이 때, pos는 1 <= pos <= nodeCount + 1 의 범위 안에 있어야 할 것이다.

두 가지 주의 사항이 존재하는데

1. 삽입하려는 위치가 리스트 맨 앞일 때
- prev 없음
- Head 조정 필요
1. 삽입하려는 위치가 리스트 맨 끝일 때
- Tail 조정 필요

따라서 코드는

```

def insertAt(self, pos, newnode):

if pos < 1 or pos > self.nodeCount + 1: return False

if pos == 1:

newnode.next = self.head

self.head = newnode

elif pos == self.nodeCount + 1: self.tail = newnode

else:

if pos == self.nodeCount + 1:

prev = self.tail # 맨 뒤 일 경우 처음부터 읽을 필요없이 tail만 사용

else:

prev = self.getAt(pos - 1)

newnode.next= prev.next

prev.next= newnode

self.nodeCount += 1

return True

```

이 때, 맨 앞에 삽입하는 경우는 O(1), 맨 끝에 삽입하는 경우도 O(1), 중간에 삽입하는 경우가 O(n)이 된다.

- 연결 리스트 연산 - 원소의 삭제

def popAt(self, pos):

인데, 이때도 주의 사항은

1. 삭제하려는 node가 맨 앞의 것일 때
- prev 없음
- head 조정 필요
1. 리스트 맨 끝의 node를 삭제할 때
- tail 조정 필요

맨 끝의 node를 삭제하는 경우 그 전의 pos -1에 대한 값을 가져올 방법이 없으므로 다 순회해야 한다.

맨 앞에 삽입하는 경우는 O(1), 중간 또는 맨 끝에 삽입하는 경우는 O(n)에 해당한다.

- 연결 리스트 연산 - 두 리스트의 연결

```

def concat(self,L):

self.tail.next = L.head

if L.tail:

self.tail = L.tail

self.nodeCount += L.nodeCount

```

이때는 L이 None이 아닌지 판단해야 하기 때문에, if를 삽입했다.

<br>

**# 9강: 연결리스트 (3)**

매번 할 때마다 getAt를 통해 리스트를 순회하는 것은 너무 불필요하다. 따라서 insertAt,popAt가 아닌 insertAfter(prev, newnode),popAfter를(prev) 통해 이전의 값을 넣으면 그 다음에 노드를 지칭하도록 만들고자 한다. 그러기 위해 0 index에 None인 dummy node를 만들어준다.

```

class LinkedList:

def __init__(self):

self.nodeCount = 0

self.head = Node(None)

self.tail = None

self.head.next = self.tail

```

이를 통해 dummy node를 생성하고, 그 노드가 원래의 1 index를 가르키도록 만든다.

- 리스트 순회

```

def traverse(self):

answer = []

curr = self.head

if curr.data == None: return answer

answer.append(curr.data)

while curr.next != None:

curr = curr.next

answer.append(curr.data)

return answer

''' ---> dummy 추가 '''

def traverse(self):

answer = []

curr = self.head

while curr.next:

curr = curr.next

answer.append(curr.data)

return answer

```

- k번째 원소 얻기

```

def getAt(self, pos):

if pos <= 0 or pos > self.nodeCount: return None

i = 1

current = self.head

while i < pos:

current = current.next

i += 1

return current

''' ---> dummy 추가 '''

def getAt(self, pos):

if pos < 0 or pos > self.nodeCount: return None

i = 0

current = self.head

while i < pos:

current = current.next

i += 1

return current

```

- 원소 삽입

```

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

```

def insertAt(self, pos, newnode):

if pos < 1 or pos > self.nodeCount + 1: return False

if pos != 1 and pos == self.nodeCount + 1: prev = self.tail # pos가 1이라면 nodeCount가 0, 즉 빈 리스트이기 때문에 prev가 존재하지 않을 것이다. 따라서 예외 처리

else: prev = self.getAt(pos - 1)

return self.insertAfter(prev,newnode)

```

prev가 상황마다 달라지기 때문에 그에 따른 prev를 각각 정의해준 후 insertAfter한 듯하다.

- 원소 삭제

고려 사항들은

1. prev가 마지막 노드일 때(prev.next == None)
- 삭제할 노드가 없다.
- return None
1. 리스트의 맨 끝의 노드를 삭제할 때(curr.next == None)
- tail 조정 필요
- 두 리스트 연결

여기서는 dummy를 뺀 첫번째 값들부터 연결시켜야 하기 때문에

```

self.tail.next = L2.head.next

self.tail = L2.tail

```

로 연결시켜준다.

```

def concat(self,L):

self.tail.next = L.head.next

if L.tail:

self.tail = L.tail

self.nodeCount += L.nodeCount

```

<br>

**# 10강: 양방향 연결 리스트**

```

class Node:

def __init__(self,item):

self.data = item

self.prev = None

self.next = None

```

여기서는 처음과 끝에 dummy node를 만들어야 한다.

```

class DoublyLinkedList:

def __init__(self, item):

self.nodeCount = 0

self.head = Node(None)

```

<br>

**# 11강: 스택**

- x.isempty() : 스택이 비어있는지 판단한다.
- x.peek() : 스택에 가장 나중에 저장된 데이터 원소를 참조한다.

스택: 자료를 보관할 수 있는 선형 구조

단, 넣을 때는 한 쪽 끝에서 밀어 넣어야 하는 push 연산, 꺼낼 때는 같은 쪽에서 뽑아 꺼내야 하는 pop 연산에 대한 제약이 있다.

이를 후입선출(LIFO - Last-In-First-Out) 특징을 가지는 선형 자료구조라 한다.

<br>

**# 12강: 수식의 후위 표기법**

중위 표기법

- 연산자가 피연산자들의 사이에 위치

```

(A + B) * (C + D)

1    3    2

```

후위 표기법

- 연산자가 피연산자들의 뒤에 위치

```

A B + C D + *

1     2 3

```

각 연산자의 우선순위를 비교하는데, 스택에 있는 연산자가 읽어온 연산자보다 우선순위가 높거나 같으면 pop, 아니면 push한다.

# 13강: 스택의 응용 - 후위 표기 수식 계산

후위 표기법으로 된 식을 계산하는 방법

중위 표기법 → 후위 표기법 → 후위 계산

이를 변환할 때는 이제 문자열이 아닌 리스트에 추가하여 리스트를 리턴할 것이다. 피연산자를 받으면 스택에 추가하고, 연산자를 만날 경우 맨 위의 2개를 빼내서 연산자에 맞게 계산한 후 다시 스택에 집어넣는다. 그렇게 마지막까지 간 후 스택에 담긴 값은 1개일 것이고, 이 값이 최종값이 된다.

```
def splitTokens(exprStr):
    tokens = []
    val = 0
    valProcessing = False
    for c in exprStr:
        if c == ' ':
            continue
        if c in '0123456789':
            val = val * 10 + int(c)
            valProcessing = True
        else:
            if valProcessing:
                tokens.append(val)
                val = 0
            valProcessing = False
            tokens.append(c)
    if valProcessing:
        tokens.append(val)

    return tokens

```

이는 중위 표현식을 받아 리스트로 변환하는 함수이다. 가장 중요한 것은 10진법을 표기해야 하기 때문에, 10진법을 구현하는 것을 주의하며 구현한다.

```
def infixToPostfix(tokenList):
    prec = {
        '*': 3,
        '/': 3,
        '+': 2,
        '-': 2,
        '(': 1,
    }

    opStack = ArrayStack()
    postfixList = []

    for s in tokenList:
        if type(s) is int:
            postfixList.append(s)
        else:
            if opStack.isEmpty():
                opStack.push(s)
            else:
                if s == ')':
                    while prec[opStack.peek()] > prec['(']:
                        postfixList.append(opStack.pop())
                    opStack.pop()
                else:
                    if prec[opStack.peek()] >= prec[s] and s != '(':
                        postfixList.append(opStack.pop())
                    opStack.push(s)

    while not opStack.isEmpty():
        postfixList.append(opStack.pop())

    return postfixList

```

이는 지난번 강의에서 했던 것처럼 중위표현식을 후위표현식으로 변환하는 함수이다. 다른 점은 지난번에는 문자열이었으나, 이번에는 리스트로 만들어 리스트를 리턴한다.

```
def postfixEval(tokenList):
    valStack = ArrayStack()
    lists = []
    for t in tokenList:
        if type(t) is int:
            valStack.push(t)
				else:
						a = valStack.pop()
            b = valStack.pop()
		        if t == '*':
	            valStack.push(b*a)
        elif t == '/':
	            valStack.push(b/a)
        elif t == '+':
	            valStack.push(b+a)
        elif t == '-':
	            valStack.push(b-a)

    return valStack.pop()

def solution(expr):
    tokens = splitTokens(expr)
    print("tokens: {}".format(tokens))
    postfix = infixToPostfix(tokens)
    print("postfix: {}".format(postfix))
    val = postfixEval(postfix)
    return val

```

최종적으로 후위 표현식을 계산하는 함수로 그냥 2개를 pop한 후 계산하여 다시 스택에 넣는 식이다.

solution에서 모든 함수를 실행시키면 결과를 얻을 수 있다.

# 14강: 큐

큐: 자료를 보관할 수 있는 선형 구조

단 넣을 때는 한 쪽 끝에서 밀어 넣어야 하는 enqueue연산, 꺼낼 때는 반대 쪽에서 뽑아 꺼내야 하는 dequeue연산의 제약이 있다.

이를 선입선출(FIFO - First-In Fisrt-Out) 특징을 가진다.

대기열 문제가 이에 해당된다.

연산의 종류

- size(): 현재 큐에 들어 있는 데이터 원소의 수를 구함
- isEmpty(): 현재 큐가 비어 있는지를 판단
- enqueue(): 데이터 원소 x를 큐에 추가
- dequeue(): 큐의 맨 앞에 저장된 데이터 원소를 반환 (제거)
- peek(): 큐의 맨 앞에 저장된 데이터 원소를 반환 (제거 x)

이때, dequeue()만 O(n)이고, 나머지는 O(1)의 복잡도를 가진다. 이는 맨 앞의 원소를 꺼내게 되면 나머지 원소들이 앞으로 당겨져야 하기 때문에, 순차적인 방식이 된다.

# 15강: 환형 큐(Circular queue)

큐의 활용:

- 자료를 생성하는 작업과 그 자료를 이용하는 작업이 비동기적(asynchronously)으로 일어나는 경우
- 자료를 생성하는 작업이 여러 곳에서 일어나는 경우
- 자료를 이용하는 작업이 여러 곳에서 일어나는 경우
- 자료를 생성/이용 둘 다 여러 곳에서 일어나는 경우
- 자료를 처리하여 새로운 자료를 생성하고, 나중에 그 자료를 또 처리해야 하는 작업의 경우

환형 큐(circular queue) : 정해진 개수의 저장 공간을 빙 돌려가며 사용

정해진 개수의 공간이므로 큐가 가득 차면 더이상 원소를 넣을 수 없다.

isfull()이라는 큐의 데이터가 꽉 차있는지 확인하는 함수를 더 추가한다.

마지막 원소 저장된 공간을 rear, 처음을 front라 한다.

전체 공간을 정의하는 self.maxcount = n, data를 None으로 초기화하기 위한 self.data를 정의한다.

# 16강: 우선순위 큐(Priority Queues)

우선순위 큐: 큐가 FIFO 방식을 따르지 않고, 원소들의 우선순위에 따라 큐에서 빠져나오는 방식

활용:

- 운영체제의 CPU 스케줄러

구현:

- 서로 다른 방식이 가능함
    - Enqueue할 때 우선순위 순서를 유지하도록
    - Dequeue할 때 우선순위 높은 것을 선택

이때는 1번이 조금 더 유리하다. 2번은 O(n)의 복잡도를 가지지만, 1번은 이미 줄지어 집어넣었기 때문에 맨 뒤나 맨 앞만 보면 되기 때문이다.

- 서로 다른 두 가지 재료
    - 선형 배열
    - 연결 리스트

시간 적으로 볼 때는 연결 리스트가 유리하다. 하지만 메모리 적으로는 연결 리스트가 많이 들기 때문에 적합한 방법을 사용하는 것이 좋다.

# 17강: 트리(trees)

트리: 정점(node)와 간선(edge)을 이용하여 데이터의 배치 형태를 추상화한 자료 구조

- 뿌리를 root node, 이파리를 leaf node라 한다.
- 이는 각각 노드에 대해 부모(parent), 자식(child)로도 표현할 수 있다.

같은 부모 아래 자식들을 sibling이라 한다.

부모의 부모를 조상(ancestor), 자식의 자식을 후손(deancestor)

child는 많을 수 있지만, 부모는 오직 1개이다.

- 노드의 수준(level)

뿌리를 level0이라 하고, 내려갈수록 level이 1씩 증가한다.

이때 뿌리를 0과 1 중 하나로 둘 수 이는데 0이 더 편하다.

- 트리의 높이(height) = 최대 수준(level) + 1

깊이(depth)라고도 한다.

- 부분트리(서브트리 -subtree)

각각의 부분에 대해 서브트리라고도 부를 수 있다.

- 노드의 차수(degree) = 자식(서브트리)의 수

degree는 각 노드의 자식의 수와도 같다.

맨 마지막의 node 즉 degree=0인 노드를 leaf nodes라고 부른다.

- 이진 트리(binary tree)

모든 노드의 차수가 2이하인 트리

- 포화 이진트리 (full binary tree)

모든 레벨에서 노드들이 채워져 있는 이진 트리 = 모든 차수가 2

높이가 k이고, 노드의 개수는 2^k - 1

- 완전 이진트리 (complete binary tree)

높이가 k인 완전 이진트리의 경우

레벨 k-2 까지는 모든 노드가 2개의 자식을 가진 포화 이진 트리

레벨 k-1 까지는 2개씩이 아니더라도 왼쪽부터 노드가 순차적으로 채워져 있는 트리

# 18강: 이진 트리(binary tree)

연산의 종류

- size(): 현재 트리에 포함되어 있는 노드의 수를 구함
- depth(): 현재 트리의 깊이 (또는 높이) 를 구함
- traverse()

```
# 이진 트리 노드의 기본 구조
class Node:
	def __init__ (self,item):
		self.data = item
		self.left = None
		self.right = None

```

- size

```
# 이진 트리 초기화
class BinaryTree:
	def __init__ (self,r):
		self.root = r

```

```
# 재귀적 방법 가능
# 오른쪽 tree size + 왼쪽 tree size + 1(자신)
class Node:
	def size(self):
		l = self.left.size() if self.left else 0
		r = self.right.size() if self.right else 0
		return l + r + 1

```

```
class BinaryTree:
	def size(self):
		if self.root: return self.root.size()
		else: return 0

```

- depth

```
# 왼쪽과 오른쪽 중 더 큰 것 고름

```

- Traversal
    - 깊이 우선 순회
        - 중위 순회(in-order traversal): 왼 → 자신 → 오
        - 전위 순회(pre-order traversal): 자신 → 왼 → 오
        - 후위 순회(post-order traversal): 왼 → 오 → 자신
    - 넓이 우선 순회

# 19강: 이진 트리 넓이 우선 순회(breadth first traversal)

수준(level)이 낮은 노드를 우선으로 방문

같은 수준의 노드들 사이에는 부모 노드의 방문 순서에 따라 방문하고, 왼쪽 자식 노드를 먼저 방문

설계

- 큐를 사용
1. 빈 리스트 traversal, 빈 큐 q 선언
2. 빈 트리가 아니면, root node를 q에 추가 (enqueue)
3. q가 비어있지 않는 동안
    1. q에서 node를 추출
    2. node 방문
    3. node의 왼, 오 자식 q에 추가
4. q가 빈 큐가 되면 종료

# 20강: 이진 탐색 트리(binary search trees)

모든 노드에 대해

- 왼쪽 서브트리에 있는 데이터는 모두 현재의 노드 값보다 작고
- 오른쪽 서브트리에 있는 데이터는 모두 현재의 노드 값보다 큰

성질을 만족하는 이진 트리

이진 탐색과 비슷하지만, 트리의 장점은 데이터 원소의 추가, 삭제가 용이

단점은 공간 소요가 크다.

추상적 자료구조

각 노드는 (key, value)의 쌍으로 되어 있음

연산의 종류

- insert(key, data): 트리에 주어진 데이터 원소를 추가
- remove(key): 특정 원소를 트리로부터 삭제
- lookup(key): 특정 원소를 검색
- inorder(): 키의 순서대로 데이터 원소를 나열
- min(),max(): 최소 키, 최대 키 원소를 탐색

```
# 기본 노드 구성
class Node:
	def __init__(self,key,data):
		self.key = key
		self.data = data
		self.left = None
		self.right = None

class BinSearchTree:
	def __init__(self):
		self.root = None

```

```
# inorder traversal
class Node:
	def inorder(self):
		#다 동일하나
		traversal.append(self)
		# self자체를 append하여 노드들의 리스트를 구함

```

```
# min
class Node:
	def min(Self):
		if self.left:
			return self.left.min()
		else:
			return self # 계속 내려가다가 더 내려갈 곳이 없으면 자신이 최솟값이므로

```

- lookup

입력은 찾으려는 대상의 키

리턴은 찾은 노드와 부모 노드(없으면 둘다 None)

```
class BinSearchTree:
	def lookup(self,key):
		if self.root:
			return self.root.lookup(key)
		else:
			return None,None

class Node:
	def lookup(self,key,parent=None):
		if key < self.key:
			if self.left:
				return self.left.lookup(key,self) # self.left 의부모는 self이므로
			else:
				return None,None
		elif key > self.key:
				...
		else: return self, parent

```

# 21강: 이진 탐색 트리(2)

원소 삭제

- 키를 이용해서 노드를 찾는다.
    - 해당 키가 없으면, 삭제할 것도 없음
    - 찾은 노드의 부모 노드도 알고 있어야 함
- 찾은 노드를 제거하고도 이진 탐색 트리의 성질을 만족하도록 트리의 구조를 정리

입력: 키

출력: 삭제한 경우 True, 해당 키가 없는 경우 False

```
class BinSearchTree:
	def remove(self,key):
		node,parent = self.lookup(key)
		if node:
			...
			return True
		else: return False

```

트리 구조의 유지

- 삭제하는 노드가
    - 말단 노드인 경우
    - 하나의 자식을 가진 경우
    - 둘의 자식을 가진 경우

# 22강: 힙(heap)

힙: 이진 트리의 한 종류

1. 루트 노드가 언제나 최댓값 또는 최솟값을 가짐
    - 최대 힙, 최소 힙
2. 완전 이진 트리여야 함

이진 탐색 트리와의 비교

1. 둘 다 원소들은 안전히 크기 순으로 정렬되어 있음
2. 특정 키 값을 가지고 빠르게 원소를 찾을 수 있는가?
    - 이진 탐색트리 - O
    - 힙 - X
3. 부가 제약 조건
    - 힙의 경우 완전 이진 트리여야 함

최대 힙의 연산의 종류

- **init**(): 비어있는 최대 힙을 생성
- insert(item): 새로운 원소를 삽입
- remove(): 최대 원소(root node)를 반환 및 제거

노드 번호m을 기준으로

- 왼쪽 자식의 번호: 2*m
- 오른쪽 자식의 번호: 2*m + 1
- 부모 노드의 번호: m // 2

완전 이진 트리이므로 노드의 추가/삭제는 마지막 노드에서만 이루어진다.

```
class Maxheap:
	def __init__(Self):
		self.data = [None]

```

0번 인덱스는 버리기 위해 none

삽입

1. 트리의 마지막 자리에 새로운 원소를 임시로 저장
2. 부모 노드와 키 값을 비교하여 위로 이동시킴

복잡도

- 원소의 개수가 n인 최대 힙에 새로운 원소 삽입 후 부모 노드와 계속 비교

→ log2(n)

두 변수 값을 바꾸는 방법

```
a,b = b,a

```

# 23강: 힙(2)

최대 힙에서 원소를 삭제

1. 루트 노드의 제거 → 이것이 원소들 중 최댓값
2. 트리 마지막 자리 노드를 임시로 루트 노드의 자리에 배치
    
    → 완전 이진 트리를 만들기 위함
    
3. 자식 노드들과의 값 비교 → 아래로 이동
    
    → 자식이 둘이라면, 자식 중 더 큰 값 선택
    

복잡도

- 원소의 개수가 n인 최대 힙에서 최대 원소 삭제

→ 자식 노드들과의 대소 비교 최대 회수: 2 x log2(n)

최악 복잡도 O(logn)의 삭제 연산

최대/최소 힙의 응용

- 우선 순위 큐
    - enqueue할 때 느슨한 정렬을 이루고 있도록 함
    - dequeue할 때 최댓값을 순서대로 추출
- 힙 정렬
    - 정렬되지 않은 원소들을 아무 순서로나 최대 힙에 삽입 O(logn)
    - 삽입이 끝나면 힙이 비게 될 때까지 하나씩 삭제 O(logn)
    - 원소들이 삭제된 순서가 원소들의 정렬 순서
    - 정렬 알고리즘의 복잡도: O(nlogn)

```python
# 힙 정렬의 코드 구현
def heapsort(unsorted):
    H = MaxHeap()
    for item in unsorted:
        H.insert(item)
    sorted = []
    d = H.remove()
    while d:
        sorted.append(d)
        d = H.remove()
    return sorted
```


c++ 아직 너무 약하다

long long, string에 대해서만 알아도 lv2까지는 풀 것 같다.
    
    LLONG_MIN, LLONG_MAX, 계산을 long long으로 한정하기 위한 1*LL
    
    string(메모리 용량 정의), vector(메모리 용량 정의)
    
- next_permutation
    
    do {
    
    } while (next_permutation(dungeons.begin(), dungeons.end()));
    

행렬 테두리 회전, 피로도, 모음 사전,