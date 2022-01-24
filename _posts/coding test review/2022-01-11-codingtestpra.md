---
layout:   post
title:    "Coding Test[Python] - 연습문제"
subtitle: "python level2 연습문제"
category: Classlog
tags:     coding-test practice
---

1. this ordered seed list will be replaced by the toc
{:toc}

프로그래머스에 올라와 있는 코딩테스트를 진행하면서 공부한 내용 및 막혔던 부분들에 대해 리뷰하고자 합니다. 저는 거의 대부분을 python을 사용하였습니다. 

이 글은 python을 통한 스택/큐에 대해 리뷰하고 다른 것들을 참고하시려면 아래 링크를 참고해주세요

* [스택/큐](https://dkssud8150.github.io/classlog/codingteststack.html)
* [힙](https://dkssud8150.github.io/classlog/codingtestheapq.html)
* [해시](https://dkssud8150.github.io/classlog/codingtesthash.html)
* [정렬](https://dkssud8150.github.io/classlog/codingtestsort.html)
* [완전탐색](https://dkssud8150.github.io/classlog/codingtestsearch.html)
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
1. 124나라의 숫자
2. 가장 큰 정사각형 찾기
3. 올바른 괄호
4. 다음 큰 숫자
5. 땅따먹기
6. 숫자의 표현
7. 최댓값과 최솟값
8. 최솟값 만들기
9. 피보나치 수
10. 행렬의 곱셈
11. JadenCase 문자열 만들기
12. N개의 최소공배수

<br>

# 124나라의 숫자

## 참고 코드

```python
def solution(n):
    answer = ''
    
    k = ['1','2','4']
    
    while n > 0:
        n -= 1
        answer = k[n%3] + answer
        n //= 3
        print(answer, n)
        
    return answer
```

```
입력값 〉	13
출력 〉	
1 4
11 1
111 0


입력값 〉	12
출력 〉
4 3
44 0
```

str형으로 하는 것은 생각했으나 좀 더 간단하게 푸는 방법을 몰랐다. 저 코드에 의하면 일단 124의 나라이므로 k를 선언한 후, 나의 경우 for문으로 0부터 시작했다. 그렇게 되면 너무 많은 경우의 수가 생겼다. while을 통해 n을 3으로 나눈 나머지에 대해 k[n%3]을 넣고, 문자이므로 그냥 더하여 문자열을 계속 추가해준다. 나는 1의 자리부터 채울 생각을 했지만, 앞에서부터 채워넣어나가니 편안한 듯하다.

<br>

# 가장 큰 정사각형 찾기

## 나의 전체 코드

```python

```




<br>

# 올바른 괄호

## 나의 전체 코드

```python
def solution(s):
    k=0
    if s[0] == ")": return False
    for i in s:
        if i == "(": k += 1
        else: 
            if k > 0: k -= 1
    
    return k==0
```

올바른 괄호 쌍이 되기 위해서는 "("이 먼저 나온 후 ")"가 나와야 한다. 따라서 "("가 나오면 +1을 하고 앞에 "("가 있다는 의미인 k가 >0인 상태에서만 ")"가 나왔을 때 -1을 해주어 닫혔다는 것을 표현해준다. 이렇게 하면 대부분의 입력값이 통과되지만, 예외적으로 ")))()"가 될 때는 올바르지 않지만, 올바르다고 잘못 판단되어진다. 따라서 예외적으로 ")"가 먼저 나왔을 때는 바로 false을 리턴한다.

<br>

# 다음 큰 숫자

## 나의 전체 코드

```python
def solution(n):
    answer = 0
    
    n2 = (bin(n))[2:]
    print(n2, type(n2))
    
    c1 = 0
    for i in n2: 
        if i == '1': c1 += 1
    
    k = n
    
    while True:
        c2 = 0
        k+=1
        k2 = (bin(k))[2:]
        for i in k2:
            if i == '1': c2 += 1
        if c1 == c2: break
    
    return k
```

bin이라는 함수는 2진수로 바꿔주는 함수이다. 이를 실행하면 0b + 2진수코드를 받을 수 있다. 그래서 [2:]를 해주어 앞의 0b를 제거한다. 그 후 이 2진수에서 1의 갯수를 찾기 위해 c1 변수를 통해 갯수를 구한다. 그 후 n보다 더 큰 숫자를 찾기 위해 무한 while문을 사용하여 1씩 더해가며 찾는다. 그렇게 구해진 k들의 2진수값에서 1의 갯수가 맞는 가장 작은 수를 찾게 되면 반복문을 멈추고 리턴한다.

<br>

### bin, oct, hex

* 2진수
2진수는 bin함수로 구하고, 추출하게 되면 맨 앞에 `0b`가 붙는다.

```python
>>> bin(42)
0b101010
```

다시 원래대로 되돌리기 위해서는 `str()`이나 `int(2진수,2)`를 사용하면 된다. 원래 사실 int인수 뒤에는 디폴트 값으로 10이 되어 있다. 즉, 10진수값으로 추출하겠다는 의미다.

```python
>>> str(0b101010)
42
>>> int(0b101010,2)
42
```

* 8진수
8진수는 잘 사용하지 않지만, oct함수로 구하고, 추출하면 `0o`가 붙는다.

```python
>>> oct(42) 
0o52
```

```python
>>> str(0o52)
42
>>> int(0o52,8)
42
```

* 16진수

```python
>>> hex(42)
0x2a
```

* [참고 블로그](https://www.daleseo.com/python-int-bases/)


# 땅따먹기

##




<br>

# 숫자의 표현

## 나의 전체 코드

```python
def solution(n):
    answer = 0

    for i in range(n):
        c = i
        ing = 0

        while ing < n:
            c+=1
            ing = ing + c

        if ing == n: answer += 1 

    return answer
```

이중 반복문을 사용하여 돌렸지만, 이는 다른 사람의 코드를 참고했을 때 1줄로 표현이 가능하다.

```python
def solution(n):
    return len([i for i in range(1,n+1,2) if n % i == 0])
```

이는 등차수열 합 공식을 사용한 것이라 한다. 

*[참고 블로그](https://gkalstn000.github.io/2021/01/21/%EC%88%AB%EC%9E%90%EC%9D%98-%ED%91%9C%ED%98%84/)

<br>

# 최댓값과 최솟값

## 나의 전체 코드

```python
def solution(s):
    answer = ''
    s = list(map(int,s.split(' ')))
    return str(min(s)) + " " + str(max(s))
```

입력이 ' '로 숫자들이 구분되어 있어 ' '로 구분한 후 이를 간단하게 int list로 만들기 위해 list(map())을 사용했다. 이 것에 대해 최솟값과 최댓값을 다시 문자열로 바꾸어 리턴한다.

<br>

# 최솟값 만들기

```python
def solution(a,b):
    return sum([a*b for a,b in zip(sorted(a), sorted(b,reverse=True))])
```

최댓값과 최솟값을 이어 곱해주면 가장 작은 수가 될 것이라 생각했다. 그래서 이를 코드화 한 것이다.


<br>

# 피보나치 수

```python
def solution(n):
    s = [0,1]
    for i in range(1,n): s.append(s[i-1] + s[i])
    
    return s[-1] % 1234567
```

피보나치 수는 F(n) = F(n-1) + F(n-2) 의 식을 가지고 있다. 따라서 이를 계속 append해주면서 공식을 만들었다.

만든 것들에서 가장 뒤의 값에 1234567(이건 왜 하는건지 모름..)을 나누어주면 된다.

<br>

# JadenCase 문자열 

## 나의 전체 코드

두 가지 방법이 있다. 하나는 다소 지저분해보이나 쉽게 떠올릴 수 있는 방법이고, 두번째는 함수를 활용하는 것이다.

```python
def solution(s):
    return " ".join([(i[0].upper() + i[1:].lower()) if i != '' else '' for i in s.split(' ')])
```

```python
def solution(s):
    return ' '.join(i.capitalize() for i in s.split(' '))
```

첫번째부터 보면 ' '단위로 분할한 후 빈 문자열이 아닐 경우 맨 앞 문자열은 대문자로, 다른 것은 소문자로 변경해준 후 list를 str형태로 바꾸어 출력한다.

두번째는 `capitalize()`를 사용하는 것으로 이 함수는 첫번째 글자가 숫자인지 문자인지 구분하지 않는 `title()`과는 조금 더 세심?한 느낌이 있다. `title()`의 경우 앞에 숫자가 있으면 뒤의 문자열 첫번째꺼를 대문자로 변경하기 때문에 예외가 많지만, `capitalize()`는 숫자인지 문자인지 구분하여 맨 앞에 있는 글자만 대문자로 만든다.

<br>

# N개의 최소공배수

## 나의 전체 코드

```python
def solution(arr):
     
    a = 1
    for i in arr: a *= i
    print(a)
    c=1
    while c != a:
        c+=1
        z=0
        for i in arr:
            if c % i == 0: 
                z += 1
                pass
            else: break
            if z == len(arr): return c
```

조금 지저분하게 풀었다. 일단 모든 숫자를 다 곱한 후 그 숫자까지 반복을 하는데, 모든 arr의 원소들이 다 나눠지는 숫자가 되면, 즉 c%i==0이 모든 원소가 다 되는 상황(z=len(arr))이 되면 멈추고 이를 반환한다.

이 때, `gcd` 함수를 사용할 수 있다.

### gcd

파이썬의 최대공약수를 구하는 함수로 math라이브러리에 포함되어 있다. 최대공약수는 둘 이상의 정수의 공약수 중에서 가장 큰 것을 말한다.

```python
import gcd
>>> math.gcd(66, 22, 11) 
11 
```

이를 활용해 다시 코드를 짠다면 다음과 같다.

```python
from math import gcd

def solution(arr):
    answer = arr[0]
    for i in arr: answer = i * answer // gcd(i,answer)
    return answer
```

동일하게 모든 list의 원소들을 다 곱해주는데, i와 answer의 최대공약수를 나누어준다. `arr = [5,6,8,14]`일 때, `gcd(i,answer)`를 매 반복마다 출력해보면 다음과 같다.

```python
[5, 6, 8, 14]
출력 〉	
i:5 answer:5 gcd(i,answer): 5
i:6 answer:5 gcd(i,answer): 1
i:8 answer:30 gcd(i,answer): 2
i:14 answer:120 gcd(i,answer): 2
```

최대공약수만을 나눈다는 것은 동일한 공약수를 1번만 곱하겠다는 의미와 같다. 즉, 8과 30이 있을 때 8을 곱할 건데 이전에 이미 2를 곱했었기 때문에 2를 중복으로 곱하지 않기 위해 나누어준다.


* lcm
추가적으로 `math.lcm`함수는 최소공배수를 찾아주는 함수다. 따라서 저 문제는 lcm으로 풀면 한 줄이면 가능하다. 하지만 3.9버전부터 사용가능하다

```python
import math
def solution(arr):
    return math.lcm(arr)
```


<br>

<br>

## + 추가적으로 알아본 함수/표현

### zip(*iterable)

`zip(*list)`를 사용하게 되면 행이 아닌 열 단위로 값을 볼 수 있다.

```python
>>> x = [[0,0,0],[1,1,1]]
>>> for i in zip(*x): print(i)
[[0,1],[0,1],[0,1]]
```



### range(n,0,-1)

반복문을 n부터 0까지 반복하고자 할 때는 `for i in range(n,0,-1)`로 작성하면 된다.


### pow
`pow(x,y)`는 x의 y 제곱한 결과를 추출한다.


### filter
`filter(함수, 반복가능한 자료형)`은 반복 가능한 자료형이 함수에 입력됬을 때 반환 값이 참인 것만 묶어서 리턴해준다.

```python
def positive(x):
    return x > 0

print(list(filter(positive, [1, -3, 2, 0, -5, 6])))
```

positive함수에 반복가능한 자료형인 list[1,-3,2,0,-5,6]을 하나씩 집어넣어 참인 것만 묶기 때문에 1,2,6 만 추출할 수 있다.


### all
`all(x)`은 반복 가능한 자료형 x를 인수로 받아 x의 요소가 모두 참이면 True, 하나라도 False이면 False를 추출한다. 