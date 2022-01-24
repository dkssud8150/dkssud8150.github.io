---
title:    "Coding Test [C++] - 연습문제"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-01-05 12:00:00 +0800
categories: [Classlog,coding test review]
tags: [coding test, pratice]
toc: True
comments: True
math: true
pin: true
mermaid: true
#image:
#  src: /commons/devices-mockup.png
#  width: 800
#  height: 500
---



프로그래머스에 기재되어 있는 c++ coding test에 대해 리뷰하고자 합니다. 그 중에서도 연습문제에 해당하는 문제를 이 곳에 작성할 예정입니다. 다른 것을 참고하시려면 아래 링크를 클릭하시기 바랍니다.

1. [해시](https://dkssud8150.github.io/classlog/codingtestchash.html)
2. [정렬](https://dkssud8150.github.io/classlog/codingtestcsort.html)
3. [완전탐색](https://dkssud8150.github.io/classlog/codingtestcsearch.html)
4. [카카오 블라인드 채용 문제](https://dkssud8150.github.io/classlog/codingtestckakao.html)

<br>

### Index

1. 더 맵게
2. 이중우선순위큐
3. 디스크 컨트롤러

# 2016년

## 나의 전체 코드

```cpp
#include <string>
#include <vector>
#include <iostream>

using namespace std;

string solution(int a, int b) {
    string answer = "";
    
    vector<string> week = {"THU","FRI","SAT","SUN","MON","TUE","WED"};
    
    int days[] = {31,29,31,30,31,30,31,31,30,31,30,31};
    
    int day = 0;
    
    for(int i=0; i<a-1; i++){
        day += days[i];
    }
    day += b;
    
    int c = day%7;
    
    answer = week[c];

    return answer;
}
```

2016년의 경우 1월1일이 금요일이다. 따라서 b=1=day의 경우 FRI 가 출력되어야 한다. 금요일을 시작으로 5/24일까지 날짜를 다 더해서 7로 나눈 후 나머지를 구한다. 



### vector container
vector 컨테이너는 자동으로 메모리가 할당되는 배열이다. 이 구조는 push_back, pop_back을 통해 맨 뒤에 삽입/제거를 사용할 수 있다.

```cpp
vector<int> v2(v1);
```
이 형태는 v2는 v1 vector를 복사하여 생성한다는 것이다.

```cpp
vector<int> v(5, 2);
```
2로 초기화된 5개의 원소를 가지는 vector v를 생성한다.

```cpp
vector<string> v;
```
string으로 된 vector 컨테이너 v를 생성한다.


#### vector에서 사용하는 함수

* v.assign(3,2)
2의 값으로 3개의 원소 할당

* v.at(idx)
v에서의 idx번째 원소를 참조. v[idx]도 사용가능하나 안전하게 사용가능

* v.front() / v.back()
첫번째 / 마지막 원소를 참조

* v.begin() / v.end()
첫번째 / 마지막 원소를 가르킴

* v.push_back(2) / v.pop_back()
마지막 원소 뒤에 원소 2를 삽입 / 마지막 원소를 제거

* v.resize(n) / v.resize(n,3)
크기를 n으로 변경, n 이상의 원소는 default값인 0으로 초기화 / 3으로 지정하게 되면 0 대신 3으로 초기화

* v.size()
v의 원소의 갯수를 리턴

* v2.swap(v1)
v1와 v2의 원소와 capacity를 바꿔줌

* v.insert(2,3,4)
2번째 위치에 3개의 4값을 삽입, 즉 2번째 위치에다가 4값을 3개 추가한다는 것

* v.erase(iter)
iter가 가리키는 원소를 제거, 사이즈는 줄지만 메모리는 그대로 남음

* v.empty()
size가 비었으면 true를 리턴

* [참고 블로그](https://blockdmask.tistory.com/70)

<br>

<br>

# 가운데 글자 가져오기

## 나의 전체 코드 

```cpp
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

string solution(string s) {
    string answer = "";
    string c;
    
    int ceils = ceil(float(s.length())/2);
    
    ceils -= 1;
    
    for(int i=1;i<=ceils;i++) {
        s.erase(s.begin());
        s.pop_back();
    }
    
    cout << ceils <<  " " << answer << endl;
    
    return s;
}
```

```cpp
입력값 〉"abcde"
출력 〉ceils: 2    s: c
```

s의 크기를 2로 나누고, 그것을 올림한다. 그렇게 되면 ceils는 3이 되는데, 1뺀 값인 2번을 앞뒤로 뺀다. 이 때 s의 크기가 짝수이면 2로 나누면 정수가 되고, 이에 1을 뺀 값만큼 앞뒤로 빼면 중앙값이 나오게 된다. 하지만,,

이는 1줄이면 바로 출력이 가능하다.

```cpp
    return s.length()&1 ? s.substr(s.length()*0.5,1) : s.substr(s.length()*0.5-1,2);
```

<br>

### substr

```cpp
string.substr(start,length)
```

start는 탐색구간의 시작점, length는 탐색구간의 길이이므로 start ~ start+length 위치에 있는 문자열을 리턴

인자로 음수를 넣으면 뒤에서부터 한다.


<br>

# 같은 숫자는 싫어

## 나의 전체 코드

```cpp
#include <vector>
#include <iostream>

using namespace std;

vector<int> solution(vector<int> arr) 
{
    vector<int> answer;
    
    for(int i=0; i<arr.size();i++) {
        answer.push_back(arr[i]);
        if(i>0 && arr[i-1] == arr[i]) {
            answer.pop_back();
        }
    }
    
    return answer;
}
```

이 또한 1줄로 가능하다.

```cpp
arr.erase(unique(arr.begin(), arr.end()),arr.end());
vector<int> answer = arr;
```

### unique

unique는 연속된 중복 원소를 vector의 제일 뒷부분으로 보내버린다.

```cpp
vector<int> v
1  1  1  2  2  2  3
```

```cpp
unique(v.begin(),v.end())
1  2  3  2  2  2  3
```

이처럼 중복된 값은 뒤로 보내는데, 문제는 중복된 값 그대로 보내는 것이 아니라 원래 그 자리에 있던 원소 값이 들어가게 된다. 즉, 중복된 값을 제거한 원소는 [1,2,3] 인데, 이 뒤에 자리를 보면 [2,2,2,3]이라는 원래 자리에 있던 원소가 들어가 있는 것을 볼 수 있다.

우리는 중복된 값을 제거한 vector가 필요하기 때문에 필요 없는 뒷부분은 제거해준다.

```cpp
v.erase(unique(v.begin(), v.end()), v.end());
```

중복된 값을 뒤로 보내준 뒤 반환되는 값은 vector의 중복된 값의 첫번째 위치로 반환된다. 따라서 이 부분부터 뒷부분을 다 제거한다. unique는 문자도 정렬하여 사용할 수 있다.

```cpp
vector<int> v
a  a  a  b  b  b  c
```

```cpp
v.erase(unique(v.begin(),v.end()),v.end())
a  b  c
```

* [참고 블로그](https://dpdpwl.tistory.com/39)

<br>

# 나누어 떨어지는 숫자 배열

## 나의 코드

```cpp
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

vector<int> solution(vector<int> arr, int divisor) {
    vector<int> answer;
    
    for(int i=0;i<arr.size();i++) {
       if (arr.at(i) % divisor == 0) {
           answer.push_back(arr.at(i));
       }
    }
    
    sort(answer.begin(),answer.end());
    
    if(answer.size() == 0) {
        answer.push_back(-1);
    }
    
    return answer;
}
```

### push_back()
python의 append 역할을 하는 함수로 push_back을 사용할 수 있다.

### sort()
sort를 할 때는 `#include <algorithm>`을 해줘야 한다.
`sort(배열의 포인터, 배열의 포인터 + 배열의 크기)` 로 되어있지만, 대부분은 `sort(vector.begin(), vector.end())` 로 사용한다. 즉 vector의 첫번째부터 끝까지 정렬한다.

내림차순으로 하고 싶을 경우 `sort(vector.begin(), vector.end(), desc)` 로 사용한다.

<br>

# 두 정수 사이의 합

```cpp
#include <string>
#include <vector>

using namespace std;

long long solution(int a, int b) {
    long long answer = 0;
    
    if(a>b) {
        int n = a;
        a = b;
        b = n;
    }
    
    for(int i=a;i<=b;i++) {
        answer += i;
    }
    
    return answer;
}
```

### long long
long long 은 int형 연산에서 초과되는 범위를 다룰 때 사용한다.

변수 자료형에는 int, char, long long, float, double 등이 있다.

