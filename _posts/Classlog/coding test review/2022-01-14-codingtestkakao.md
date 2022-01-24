---
title:    "Coding Test[Python] - 카카오 및 코딩테스트 기출문제"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-01-14 12:00:00 +0800
categories: [Classlog,coding test review]
tags: [coding-test]
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

# 멀쩡한 사각형

```python
def solution(w,h):
    answer = 0
    
    for i in range(1,h+1):
        c = w
        while True:
            if (h-i) * (w/h) >= c: break
            c -= 1
        answer += c
        if c == 0 : return answer*2
```

나는 이와 같이 코드를 짰는데, 코드를 돌려보면 시간 초과로 53점을 받는다. 그렇다면 시간을 줄일 수 있는 방법을 찾아야 할텐데, 그 방법은 아마 패턴을 찾는 것일 것이다. 

<img src="_site/assets/img/2022-01-14/pattern.png">

이 문제에서 [8,12] 사각형의 대각선이 지나가는 패턴을 보면, [2,3]의 크기로 대각선에 의해 제거되야 하는 사각형의 패턴이 보인다. 이 때, 최소공약수를 구해보면 다음과 같다.

```python
>>>print(gcd(8,12))
4
```

즉, 패턴이 4번 반복된다는 것을 알 수 있다. 

이해가 되지 않는다면 이에 대해 구글링하면 많이 나올 것이다.

아무튼 이렇게 `gcd`함수를 사용하여 패턴을 구했다.

```python
w,h = [8,12]
>>>print(w//gcd(w,h), h//gcd(w,h))
2 3
```

따라서 패턴은 [2,3]의 크기로 4번 반복된다. 그렇다면 [2,3] 크기 안에서 제거해야 할 사각형, 즉 대각선이 지나가는 사각형의 갯수를 구해야 한다. 이는 참고 블로그의 저자는 세로와 가로를 나누어 갯수를 센다. 대각선은 세로줄 기준으로 최소 1칸씩은 차지한다. 따라서 `h//gcd(w,h)`가 되고, 가로줄 기준으로는 1줄에 2개를 차지할 수 있다. 그 갯수는 `w//gcd(w,h) - 1`가 된다. 최종 코드는 다음과 같다.

```python
from math import gcd
def solution(w,h):
    return w * h - (w//gcd(w,h) + h//gcd(w,h) -1) * gcd(w,h)
```

<br>

# 짝지어 제거하기

## 나의 전체 코드

```python
def solution(s):
    arr=[]
    s = list(s)
    for i in range(len(s)):
        if len(arr) == 0: arr.append(s[i])
        else:
            if arr[-1] == s[i]: arr.pop()
            elif arr[-1] != s[i]: arr.append(s[i])
        
    if len(arr) == 0: return 1
    else: return 0
```

처음에는 재귀함수를 사용해서 돌렸는데, 시간이 너무 오래걸려서 다른 방법을 생각했다. 굳이 for문을 계속 돌릴 필요없이 같으면 pop, 다르면 append 하면 0이 되거나 더이상 없을 때까지 돌릴 필요없이 1번에 다 진행시킬 수 있다.

`baabaa`의 경우, b를 arr에 먼저 넣고, 그 다음 a를 가지고 와서 arr 맨 뒤 값인 b와 비교하여 같으면 b를 제거, 다르면 a를 arr에 삽입하는 식이다.


<br>

# 행렬 테두리 회전하기

## 나의 전체 코드

```python
def solution(rows, columns, queries):
    answer = []
    c=1
    num = [[c+i+j for j in range(columns)] for i in range(0,rows*columns,columns)]
    
    for i in queries:
        arr = []
        x1,y1,x2,y2 = i
        x1-=1;y1-=1;x2-=1;y2-=1

         
        a = num[x1][y1:y2+1]

        for c in range(1,y2-y1+1): 
            num[x1][y1+c] = a[0]
            arr.append(a.pop(0))
        
        for i in range(1,x2-x1+1): a.append(num[x1+i][y2])    
            
        for r in range(1,x2-x1+1): 
            num[x1+r][y2] = a[0] 
            arr.append(a.pop(0))
        
        for i in range(y2,y1,-1): a.append(num[x2][i-1]) 
            
        for c in range(y2-1,y1-1,-1): 
            num[x2][c] = a[0]
            arr.append(a.pop(0))
        
        for i in range(x2,x1,-1): a.append(num[i-1][y1])  
        
        for c in range(x2-1,x1-1,-1): 
            num[c][y1] = a[0]
            arr.append(a.pop(0))
        
        arr.append(a[0])
        
        answer.append(min(list(map(int,arr))))
        
    return answer
```

이것이 좋은 코드는 아니지만, 다른 사람의 풀이를 봐도 비슷한듯하다. 회전을 하는 것이기에 이동하기 전 숫자를 미리 추출하여 a 리스트에 넣어놓고, 1칸씩 밀어서 리스트 값을 삽입한다. 나의 경우 사각형 기준 위, 오른쪽, 아래, 왼쪽 순으로 회전시켰다.

<br>

# 메뉴 리뉴얼

## 나의 전체 코드

```python
import itertools 

def solution(orders, course):
    answer = []
    com = []
    
    maxlen_order = len((sorted(orders, key=len,reverse=True))[0])
    
    arr = [list(map(str,i)) for i in orders if len(i) > 1]
    
    for c in course:
        dic = {}

        for i in arr:
            com = ["".join(sorted(co)) for co in list(itertools.combinations(i,c))]
            for a in com:
                if a not in dic: dic[a] = 1
                else: dic[a] += 1  
        
        maxv = 1
        for k,v in dic.items():
            if maxv < v: 
                maxv = v
                maxk = k
        
        if len(maxk) != maxlen_order and len(dic) != 0: 
            answer.append([k for k,v in dic.items() if v==maxv and maxv > 1])
            
    a = []
    for i in answer: 
        for k in i: a.append(k)
    return sorted(a)
```

이는 `itertools.combinations`를 사용한 방법이다. orders를 str형 list로 변환한 후 이를 combinations 즉, 조합 배열을 만든다. 조합 배열이라 함은 list 중에서 c개 만큼을 뽑는데, 순서를 고려하지 않고 추출한다. 예를 들어 [1,2,3]의 리스트가 있고 이를 combinations를 사용해보자.

```python
>>> lists = [1,2,3]
>>> k = list(itertools.combinations(lists,2))
>>> print(k)
[(1,2),(2,3),(1,3)]
```

이렇게 만들어진 리스트를 알파벳들을 조합하여 하나의 문자로 만들어 리스트 com에 넣는다. 그것을 반복문을 사용하여 원소 i가 딕셔너리 dic에 존재하면 갯수를 의미하는 [1]에 +1을 하고, 없으면 삽입한다. 이를 통해 모든 경우의 수를 따진 dic 딕셔너리를 얻는다. 거기서 가장 많이 주문된 조합을 구하여 그것에 대한 조합을 answer에 넣는다. 이렇게 넣으면 [[],[]]의 형태가 되므로 이를 또 풀어줘야 한다.

<br>

# 괄호 변환

## 나의 전체 코드

```python
import sys
sys.setrecursionlimit(1000001)

def correct(p):
    c = 0;cc=0
    u=[]
    
    # 1
    if p == '': return p
    
    
    # 2
    for i in p:
        if i == "(": cc += 1
        elif cc>0 and i ==")": cc-=1
    if cc == 0: return p
    
    for i in p:
        if i == "(": c+=1
        elif i == ")": c-=1
        u.append(i)
        if c == 0: break
    
    u = "".join(u)
    v = p[len(u):]
    
    
    
    # 3
    cu = 0
    for i in u:
        if i == "(": cu += 1
        elif cu>0 and i ==")": cu-=1
    if cu == 0: return u + correct(v)
      

    # 4
    q = u[1:len(u)-1]
    u=[]
    for i in q:
        if i == "(" : i = ")"
        elif i == ")" : i = "("
        u.append(i)
    
    return '(' + correct(v) + ')' + ''.join(u)



def solution(p):
    answer = ''
    
    return correct(p)
```

이 문제는 문제 설명 자체에서 재귀적이라는 표현이 있는 것으로 보아 재귀함수를 사용해야 했다. 문제에 맞춰 코드만 잘 작성하면 알맞게 동작한다. 유의할 것은 `cc`나 `cu`에 대한 반복문을 쓸 때, 즉 올바른 괄호 문자열을 판별할 때 그냥 `elif i == ")"`라고 작성하게 되면 ))(( 에 대해서도 올바르다고 판별하게 나온다. 따라서 cu>0 이라는 표현, 즉 (가 먼저 나온 후 )가 나올 때만 +1을 하여 올바름을 판별한다.

<br>

# [1차]뉴스 클러스터링

## 나의 전체 코드

```python
import math

def solution(str1, str2):
    intersec=[]
    str1 = str1.lower();str2 = str2.lower()
    str1=list(str1);str2=list(str2)
    
    arr1 = [str1[i-1] + str1[i] for i in range(1,len(str1)) if (str1[i-1] + str1[i]).isalpha()]
    arr2 = [str2[i-1] + str2[i] for i in range(1,len(str2)) if (str2[i-1] + str2[i]).isalpha()]
    print(arr1,"  ",arr2,"\n")
    
    if arr1 == [] and arr2 == []: return 65536
    
    arr1_copy = arr1.copy()
    arr2_copy = arr2.copy()

    # 교집합
    inter = []
    for i in arr1:
        if i in arr2_copy:
            inter.append(i)
            arr1_copy.remove(i); arr2_copy.remove(i)
    
    answer = math.floor((len(inter) / len(inter + arr1_copy + arr2_copy)) * 65536)
    return answer
```

일단 `str.isalpha()`는 문자열 str에 문자만 있는지를 판단하여 모두가 문자열이면 True, 1개라도 문자가 아닌 것이 있으면 False를 리턴하게 된다. 처음에는 set 함수만을 사용하려고 노력했다. 하지만 사용하지 않고도 간단하게 풀 수 있는 방법이 많기에 set은 순서와 중복을 제거할 때만 사용하도록 하자. 반대로 문자열이 숫자인지 아닌지를 판별해주는 isdigit함수도 있다.

`copy()`는 복합 객체를 복사하는 것이다. 사용하지 않고 그냥 `arr_copy = arr1`으로 하게 되면 arr_copy의 원소를 변경하게 되면 arr1도 같이 변한다.

### isalnum

이는 문자열 내 특수문자 사용 여부를 판별한다. `c.isalnum()`를 하게 되면 변수형 c가 모든 문자가 문자 또는 숫자인 경우 True, 그렇지 않은 경우 False를 반환한다.

```python
>>>string = "Hello World!!"
>>>new_str = ''.join(char for char in string if char.isalnum())
>>>print(new_str)
HelloWorld
```

<br>

a = list(map(list.__add__, p, x))