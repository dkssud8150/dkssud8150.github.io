---
title:    "Coding Test[Python] - 재귀함수"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-01-13 12:00:00 +0800
categories: [Classlog,coding test review]
tags: [coding test, recursion]
toc: True
comments: True
math: true
mermaid: true
#image:
#  src: /commons/devices-mockup.png
#  width: 800
#  height: 500
---

<br>

# 재귀함수

## 피보나치 수열

```python
def fibo(n):
    if n <= 1: return 1
    return fibo(n-2) + fibo(n-1)

def fibo_list(n):
    return fibo(n-1)
```

이 코드가 기본적인 피보나치 수열의 코드이다. 하지만 이는 메모리적으로나 시간적으로나 효율이 좋지 않다. 그런 이유를 찾아보면, 계산이 했던 것을 계속해서 다시 계산을 하기 때문에 오래 걸리게 되는 것이다. 그렇다면 우리는 해결책으로 어딘가에 저장해서 그 어딘가에 없다면 추가하고, 있다면 그 값을 불러오기만 하면 되지 않을까?

이에 대해 잘 정리된 블로그가 있어 참고로 넣는다.

*[참고 블로그](https://mong9data.tistory.com/22)

```python
import sys
sys.setrecursionlimit(10000000)

dic = {0:0, 1:1}

def fibonacci(n):
    if n in dic: return dic[n]
    dic[n] = fibonacci(n-2) + fibonacci(n-1)
    return dic[n] % 1234567

def solution(n):
    return fibonacci(n)
```

파이썬은 원래 재귀를 1000번까지로 제한을 한다. `setrecursionlimit`을 사용해 그 제한을 풀어줄 수 있다. 이보다 중요한 것은 `dic`이다. 개별적으로 보게 되면

`dic = {0:0, 1:1}`을 선언하여 공간을 만들어놓고, 피보나치 수는 보통 2부터 진행하고 0일때는 0, 1일때는 1을 출력한다. 따라서 0과 1에 대해서는 입력시켜놓고, fibonachi 재귀함수를 선언하는 것은 동일하나 

```python
if n <= 1: return 1
return fibo(n-2) + fibo(n-1)
```

```python
if n in dic: return dic[n]               # 1
dic[n] = fibonacci(n-2) + fibonacci(n-1) # 2
```

두 부분을 함께 보게 되면, 원래는 `if n <= 1: return 1` 을 통해 end condition을 적용하고, `return` 을 통해 재귀를 계속해준다. 이 대신 딕셔너리에 없으면 2번줄처럼 선언해서 집어넣는다. 하지만 딕셔너리에 값이 존재할 경우에는 그 값을 불러오기만 하면 된다. 100번째 피보나치 수를 찾으면 `354,224,848,179,261,915,075`이 나온다. 이 값을 찾기 위해 첫번째 방식을 사용하면 10초가 넘어가서 실행되지 않는다. 

아래의 방법으로 진행할 경우 둘다 실행은 되는 테스트 케이스인 6번을 기준으로 첫번째는 0.12~0.15ms / 10.2MB , 두번째 방식은 0.01ms / 10.2MB 이 나온다. 무려 12~15배나 빠르다는 것이다.

```python
t1 = Timer("fibonacci", "from __main__ import fibonacci")
t2 = Timer("fibo", "from __main__ import fibo")
print("fibonacci(30): ", t1.timeit(number=1000), "seconds")
print("fibo(30): ", t2.timeit(number=1000), "seconds")
```

이와 같은 방법으로 30번째 피보나치 수를 찾는 것을 1000번 실행을 시켰을 때의 시간을 얻을 수 있다고 한다. 