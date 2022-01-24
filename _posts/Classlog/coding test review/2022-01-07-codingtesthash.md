---
layout:   post
title:    "Coding Test[Python] - 해시"
subtitle: "key-value 쌍"
category: Classlog
tags:     coding-test hash
---

1. this ordered seed list will be replaced by the toc
{:toc}

프로그래머스에 올라와 있는 코딩테스트를 진행하면서 공부한 내용 및 막혔던 부분들에 대해 리뷰하고자 합니다. 

이 글은 python을 통한 힙에 대해 리뷰하고 다른 것들을 참고하시려면 아래 링크를 참고해주세요

* [스택/큐](https://dkssud8150.github.io/classlog/codingteststack.html)
* [힙](https://dkssud8150.github.io/classlog/codingtestheapq.html)
* [정렬](https://dkssud8150.github.io/classlog/codingtestsort.html)
* [완전탐색](https://dkssud8150.github.io/classlog/codingtestsearch.html)
* [연습문제]
* [카카오]

<br>

## overview

### 해시

해시는 dictionary라는 자료구조를 사용하는 방식을 말하는데, dict을 사용하는 이유는 
* 리스트를 사용하지 못할 때
list['a']가 불가능하기 때문에 dict을 통해 dict['a']를 찾을 수 있다.

* 빠른 탐색이 필요할 때
딕셔너리는 리스트보다 탐색이 빠르다.

* 집계가 필요할 떄
원소의 개수를 세는 문제에서 해시와 collections 모듈의 counter 클래스를 사용하면 빠르게 해결할 수 있다.

### Index

1. 완주하지 못한 선수
2. 전화번호 목록
3. 위장
4. 베스트앨범

<br>

<br>

# 완주하지 못한 선수

## 나의 전체 코드 - list 사용

```python
def solution(participant, completion):
    for i in participant:
        if not i in completion: return i
        elif participant.count(i) != completion.count(i): return i
```

## dict 사용

```python
def solution(participant, completion):
    c = {c:i for i,c in enumerate(completion)}
    p = {p:i for i,p in enumerate(participant)}
    
    for i in p.keys():
        if c.get(i,-1) == -1: return i
        elif participant.count(i) > 1 and participant.count(i) != completion.count(i): return i

```

```python
입력값 〉	["marina", "josipa", "nikola", "vinko", "filipa"], ["josipa", "filipa", "marina", "nikola"]
출력 〉	
dict p:  {'josipa': 0, 'filipa': 1, 'marina': 2, 'nikola': 3}
return :  'vinko'
```

효율성이 4개 안되지만, 이보다 좋은 방법이 떠오르지 않았다.

<br>

### dict

dictionary에서 사용하는 함수들로는 다음과 같다.

* dict.get(key, 특정 값)
dict에서 원소를 가져오는데, 해당하는 값이 없을 경우 keyerror 대신 특정한 값을 리턴한다.

```python
dict = {'가': 3, '나': 1}
dict.get('다', 0) # 0
```

dict에 '다'라는 key가 없다면 0을 출력한다.

* del dict[key]
dict에서 key에 해당하는 값을 삭제한다. dict.pop()함수를 사용해도 된다. 이때 `dict.pop('가',2)`와 같이 적게 되면 '가' key를 삭제하고자 하는데 값이 없는 경우 2를 리턴한다.

* dict.items() / dict.keys() / dict.values()
for문을 사용할 때 key-value를 동시에 반복하기 위해 사용할 수 있다.

```python
for key, value in dict.items():
    print(key,value)
```

하지만, key만 보고 싶을 때는 `dict.keys()`를 할 수 있다.

```python
dict.keys() # 'dict_keys(['가','나'])
```

또한, value만 보고 싶을 때는 `dict.values()`를 할 수 있다.

```python
dict.values() # 'dict_values([3,1])
```

* [참고 블로그](https://yunaaaas.tistory.com/46)

### try/except

코드에서 에러가 발생했을 때, 에러를 핸들링하는 기능이다. try 블럭에서는 실행할 코드를 작성하고, 여기서 에러가 발생했을 때 except를 통해 예외 처리한다. 마지막에 항상 실행되는 문으로 finally도 함께 쓸 수 있다.

```python
value = [1,2]
for i in range(3):
    try:
        sum = value[i]
    except Indexerror as err:
        print("indexerror: ",err)
    except keyerror as err:
        print("keyerror: ",err)
    finally: print("finish!", "  sum: ", sum)
```

```python
finish! sum: 1
finish! sum: 2
indexerror:  list index out of range
finish! sum: 2
```

value[0]과 value[1]은 값이 있지만, value[2]는 값이 없다. 따라서 indexerror가 발생했고, 그것을 예외 처리하여 `indexerror: error 내용` 으로 출력했고, finally를 매 반복마다 출력한 것을 볼 수 있다.

<br>

# 전화번호 목록

## 나의 전체 코드

```python
def solution(phoneBook):
    phoneBook.sort()

    for p1, p2 in zip(phoneBook, phoneBook[1:]):
        print(p1,p2)
        if p2.startswith(p1):
            return False
    return True
```

```python
12 23
23 34
34 45
45 56
```

정렬 후 바로 앞의 문자와만 비교하면 된다.

<br>

# 위장

## 나의 전체 코드

```python
def solution(clothes):
    answer = 1
    b={}
    cloth = {n:k for n,k in clothes}

    for k,v in cloth.items():
        print("k: {} \t v: {} ".format(k,v))
        
        if v not in b: b[v] = 1
        else: b[v] += 1
            
    for k,v in b.items():
        v += 1
        answer *= v
        print(k,v)
        
    answer -= 1
    return answer
```

다른 사람의 풀이를 조금 참고 했는데, 나는 b라는 종류별 수량을 리스트로 또 만들어 거기서 1,2,3,... 개씩 빼면서 나머지의 원소들을 곱하는 식으로 계산하고, 완전 다 입었을 때와 개별로 입었을 때의 값을 더해주었다. 그 코드는 매우 지저분했으며, 예외사항이 너무 많았다. 

하지만, **입지 않는다는 경우의 수를 1씩 더 해주는 방식**으로 경우의 수를 생각해본다면 결과는 똑같이 나온다.

<br>

# 베스트앨범

## 나의 전체 콛

```python
def solution(genres, plays):
    answer = []
    
    sing = dict()
    for g,p in zip(genres, plays):
        if g in sing: sing[g] += p
        else: sing[g] = p
    
    sing = sorted(sing.items(), key=lambda x: x[1],reverse=True)
    sing = [s[0] for s in sing]
    
    print(sing)
    
    for i in sing:
        s = []
        for k in range(len(plays)):
            try:
                if plays[genres.index(i,k)] not in s:s.append(plays[genres.index(i,k)])
            except ValueError as err:
                print(err)
        
        s = sorted(s,reverse=True)
        s = s[:2]
        print(s)
        
        for j in s:
            answer.append(plays.index(j))
    
    return answer
```

```python
입력값 〉	["classic", "pop", "classic", "classic", "pop"], [500, 600, 150, 800, 2500]
출력 〉	
sing: ['pop', 'classic']
pop: [2500, 600]
'classic' is not in list
classic: [800, 500]
```

이 코드는 86.7/100 의 점수를 받았다. 먼저 조회수를 기준으로 정렬한 후 장르만 추출한다. 그 다음 각 장르마다 조회수를 추출하여 크기별로 정렬하고, 2개만 추출하기 때문에 2개까지만 리턴한다.