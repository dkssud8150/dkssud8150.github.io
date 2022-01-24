---
title:    "Coding Test[Python] - 완전탐색"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-01-10 12:00:00 +0800
categories: [Classlog,coding test review]
tags: [coding test, search]
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

* [스택/큐](https://dkssud8150.github.io/classlog/codingteststack.html)
* [힙](https://dkssud8150.github.io/classlog/codingtestheapq.html)
* [해시](https://dkssud8150.github.io/classlog/codingtesthash.html)
* [정렬](https://dkssud8150.github.io/classlog/codingtestsort.html)
* [연습문제](https://dkssud8150.github.io/classlog/codingtestsearch.html)
* [카카오]

<br>

## Overview

#### 완전탐색

완전탐색에서는 딱히 특정한 방법이 존재하는 것이 아니라 이것저것 시도해봐야 한다. 완전탐색 부분이 수학적인 부분들을 많이 생각해봐야 하는 것 같다. 

* set

```python
>>>s = set()
>>>s = set([1,2,3])
{1,2,3}
>>>s = set("Hello")
{'e','H','l','o'}
```

* [참고 블로그](https://wikidocs.net/1015)

### INDEX
1. 모의고사
2. 소수 찾기
3. 카펫

<br>

# 모의고사

<br>

# 소수 찾기

## 전체 코드

```python
from itertools import permutations
def solution(n):
    a = set()
    for i in range(len(n)):
        a |= set(map(int, map("".join, permutations(list(n), i + 1))))
    a -= set(range(0, 2))
    for i in range(2, int(max(a) ** 0.5) + 1):
        a -= set(range(i * 2, max(a) + 1, i))
    return len(a)
```

이 문제는 혼자 풀지 못해서 다른 사람의 풀이를 보았다. 하지만, 그래도 이해가 되지 않았다. 일단 모르는 기호나 함수가 많았기 때문이다. 또한, for문으로 각각을 list에 넣지 않고, list(numbers)를 하면 바로 list로 만들어진다. 그렇기에 `list(map(int, numbers))`를 하게 되면 for문을 사용하지 않아도 된다.

### set

집합(set)은 set()의 괄호 안에 리스트나 문자열 등을 입력할 수 있다. 또 set()과 같이 비어 있는 집합 자료형을 만들어볼 수도 있다.

```python
>>>s = set()
>>>s = set([1,2,3])
>>>li = list(s)
>>>li
[1,2,3]

>>>tu = tuple(s)
>>>tu
(1,2,3)
```

set은 중복을 허용하지 않기 때문에 자료형의 중복을 제거하기 위한 필터 역할로도 사용할 수 있다.

<br>

set의 유용한 점 중 하나는 교집합, 합집합, 차집합을 구할 수 있다는 것이다.

```python
>>>s1 = set([1,2,3,4,5,6])
>>>s2 = set([4,5,6,7,8,9])
```

* 교집합

```python
>>>s1 & s2
{4,5,6}

>>> s1.intersection(s2)
{4,5,6}
```

s2.intersection(s1)을 사용해도 결과는 같다.

* 합집합

```python
>>> s1 | s2
{1,2,3,4,5,6,7,8,9}

>>> s1.union(s2)
{1,2,3,4,5,6,7,8,9}
```

* 차집합

```python
>>> s1 - s2
{1,2,3}
>>> s2 - s1
{7,8,9}

>>> s1.difference(s2)
{1,2,3}
>>> s2.difference(s2)
{7,8,9}
```

* 값 추가

값을 추가할 때는 `add`를 사용한다.

```python
>>>s1=set([1,2,3])
>>>s1.add(4)
```

여러 개를 추가할 때는 `update`를 사용한다.

```python
>>>s1.update([4,5,6])
```

제거할 때는 `remove`를 사용한다.

```
>>>s1.remove(2)
```

* [참고 블로그](https://wikidocs.net/1015)

<br>

### |=

`|`는 or 연산자이므로 `|`=는 `|`의 결과를 update한다는 뜻이다.

```python
>>> s1 = [1,2,3]
>>> s2 = [4,5,6]

# 결과만 추출
>>> s1 | s2
>>> s1
[1,2,3]

# 결과를 저장
>>> s1 |= s2
>>> s1
[1,2,3,4,5,6]
```

* [참고 블로그](https://stackoverflow.com/questions/3929278/what-does-ior-do-in-python)

### permutations

python의 itertools를 사용하면 순열과 조합을 for문 없이 구현할 수 있다.

* 순열(permutation)

순열이란 몇 개를 골라 순서를 고려해 나열한 경우의 수이다. 즉, 서로 다른 n개 중 r개를 골라 순서를 나열하는 것이고, 그래서 수학에서 `nPr`이라는 기호를 사용한다. 순열은 순서를 고려하는 것이기 때문에 [A,B,C] 리스트를 2개를 골라 나열한다면 [(A,B),(B,C),(B,A),(B,C),(C,A),(C,B)]가 나오게 된다. 

```python
import itertools

>>>array = ['A','B','C']
>>>npr = itertools.permutations(array,2)
>>>print(list(npr))
[('A','B'),('B','C'),('B','A'),('B','C'),('C','A'),('C','B')]
```

* 조합(combination  )

조합이란 순서를 고려하지 않고 나열한 경우의 수이다. 따라서 위에서 ('A','B')와 ('B','A')는 다른 경우지만, 여기서는 같은 경우로 판단한다. combination에서 c를 따 `nCr`의 기호를 사용한다.

```python
import itertools

>>>array = ['A','B','C']
>>>nCr = itertools.combinations(array,2)
>>>print(list(nCr))
[('A','B'),('A','C'),('B','C')]
```

추가로 한가지 더하자면 조건에 맞게 각각을 추가하는 것보다 빼는 것이 시간적으로 더 짧다고 한다.

<br>

### 소수 찾는 법

소수인지 판별하기 위해서 간단하게 생각하면 2~N-1까지 해당 수를 나눠보고 어떠한 값도 나누어지지 않는다면 소수라고 생각할 수 있다. 이를 코드로 표현하게 되면 다음과 같다.

```python
for i in range(2,n+1):
    if n % i == 0: return False
return True
```

하지만 이 방법은 시간이 너무 오래걸릴 수 있다. 그렇다면 어떻게 하면 시간을 줄이면서 소수를 구할 수 있을까?

약수의 특성을 활용하면 연산 횟수를 반으로 줄일 수 있다. 즉, 특정 수n의 약수를 나열해보았을 때 가운데 수를 기준으로 약수가 대칭된다. 16을 예로 들면 16의 약수는 [1,2,4,8,16]으로 4를 기준으로 대칭되고 있다. (1x16, 2x8, 4x4)

이를 활용하여 가운데 값을 기준으로 한쪽만 연산하더라도 다른 쪽의 약수를 구할 수 있다. 중간값은 제곱근, 즉 루트를 취하여 구할 수 있으며, 이 값을 기준으로 왼쪽에 약수가 하나라도 존재하지 않는다면 오른쪽에도 약수가 존재하지 않는다는 말이다. 이를 코드로 표현하게 되면 range(2, int(n ** 0.5) + 1)이 될 것이다. ** 0.5는 제곱근을 뜻하며, sqrt()로 사용해도 무관하다.

```python
for i in range(2, int(n**0.5) + 1):
    if n % i == 0: return False
return True
```

* **에라토스테네스의 체**

이때까지 1개의 수에 대해서만 구해보았다. 하지만 여러 수에 대해 소수 판별을 해줘야 한다면 어떻게 해야 할까?

100~300 사이의 모든 소수를 구해야 한다면 위의 방법으로 하나하나 판별해야 한다면 너무 오래 걸릴 것이다. 이 상황에서 사용할 수 있는 알고리즘 중 하나가 **에라토스테네스의 체** 방식이다. 알고리즘 작동방식은

1. 2~N까지의 범위가 담긴 배열을 만든다.
2. 해당 배열 내의 가장 작은 수 i 부터, i의 배수들을 해당 배열에서 지워준다.
3. 주어진 범위 내에서 i의 배수가 모두 지워지면 i 다음으로 작은 수의 배수를 같은 방식으로 배열에서 지워준다.
4. 더 이상 반복할 수 없을 때까지 2,3번 과정을 반복한다.

이를 코드화 하면 다음과 같다.

```python
for i in range(2, int(max(s)**0.5) +1):
    s -= set(range(i*2, max(s)+1,i))

    print(i, i*2, max(s)+1, int(max(s)**0.5) + 1)
    print(set(range(i*2, max(s)+1, i)))
```

```python
입력값 > [71, 17, 7]
출력 〉	
2 4 72 9 
{4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70}

3 6 72 9 
{6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69}

4 8 72 9 
{32, 64, 36, 68, 8, 40, 12, 44, 60, 16, 48, 20, 52, 24, 56, 28}

5 10 72 9
{65, 35, 70, 40, 10, 45, 15, 50, 20, 55, 25, 60, 30}

6 12 72 9 
{66, 36, 42, 12, 48, 18, 54, 24, 60, 30}

7 14 72 9 
{35, 70, 42, 14, 49, 21, 56, 28, 63}

8 16 72 9 
{32, 64, 40, 16, 48, 24, 56}
```

가장 큰 값을 기준으로 모든 약수를 빼내면 다른 것들은 자동으로 만족하게 된다. 그래서 max(s)값인 71을 기준으로 소수 찾기를 진행한다. 

따라서 `71^0.5 = 8.42 => 8 + 1 = 9` 까지 for문을 진행하며, 각 i마다 i는 소수이므로 i에 2를 곱한 후 i단위로 약수들을 다 제거해준다.

<br>

# 카펫

```python
def solution(b, y):
    answer = []
    yh=0;yw=0
    for i in range(1,int(y**0.5)+1):
        k = y // i
        if y % i== 0:
            if 2*k + 2*i + 4 == b: yh = i; yw = k
            
    answer = [yw+2, yh+2]
    answer.sort(reverse=True)
    
    return answer
```

약수와 비슷하게 가운데 값을 기준으로 약수를 다 구한 후 둘레에 대한 brown 값과 비교하여 맞다고 하면 그것이 yellow의 가로,세로길이가 될 것이다.


