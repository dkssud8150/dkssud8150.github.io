---
title:    "Coding Test [C++] - 정렬"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-01-09 12:00:00 +0800
categories: [Classlog,coding test review]
tags: [coding test, sort]
toc: True
comments: True
math: true
mermaid: true
#image:
#  src: /commons/devices-mockup.png
#  width: 800
#  height: 500
---


프로그래머스에 기재되어 있는 c++ coding test에 대해 리뷰하고자 합니다. 그 중에서도 연습문제에 해당하는 문제를 이 곳에 작성할 예정입니다. 다른 것을 참고하시려면 아래 링크를 클릭하시기 바랍니다.

1. [해시](https://dkssud8150.github.io/classlog/codingtestchash.html)
2. [연습문제](https://dkssud8150.github.io/classlog/codingtestcpra.html)
3. [완전탐색](https://dkssud8150.github.io/classlog/codingtestcsearch.html)
4. [카카오 블라인드 채용 문제](https://dkssud8150.github.io/classlog/codingtestckakao.html)

<br>

## overview

### Index
1. k번째 수


# k번째수

```cpp
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

vector<int> solution(vector<int> array, vector<vector<int>> commands) {
    vector<int> answer;
    vector<int> arr;
    
    for(int i=0;i<commands.size();i++) {
        arr = array;
        cout << arr[commands[i][0]-1] << endl;
        cout << "arr" << arr[commands[i][0]+commands[i][2]-2] << endl;
        sort(arr.begin() + commands[i][0] -1, arr.begin() + commands[i][1]);
        answer.push_back(arr[commands[i][0]+commands[i][2]-2]);  
    }
        
    return answer;
}
```

```cpp
입력값 〉	[1, 5, 2, 6, 3, 7, 4], [[2, 5, 3], [4, 4, 1], [1, 7, 3]]
출력 〉	
start point: 5
answer point: 6
start point: 6
answer point: 6
start point: 1
answer point: 2
```

### vector 함수

vector을 정렬하고 원하는 값들을 추출한다. 

다양한 함수들은 [연습문제](https://dkssud8150.github.io/classlog/codingtestcpra.html)에 적혀있으니 참고하고, 거기에 없는 부분들만 체크하고자 한다.

* vector.remove_if(bool)

```cpp
bool predicate(int num){
    return num>=100 && num<=200;
}

vector.remove_if(predicate)
```

* [참고 블로그](https://blockdmask.tistory.com/76)

<br>

# 가장 큰 수

```cpp
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

bool compare(string a, string b) {
    return (a + b > b + a);
}

string solution(vector<int> numbers) {
    string answer = "";
    vector<string> str;
    
    for (int i : numbers)
        str.push_back(to_string(i));
    
    sort(str.begin(), str.end(), compare);
    
    for (auto i=str.begin();i<str.end();i++) {
        answer += *i;
    }
    
    if(answer[0] == '0'){
        answer = '0';
    }
    
    return answer;
}
```

다른 사람의 코드를 참고했는데, sort에 bool 함수를 집어넣고, 개별로 서로 비교하여 정렬하는 것이 핵심인 것 같다.

<br>

### const

const는 값을 선언할 수 있도록 도와주는 키워드다. 즉, 값을 변경할 수 없게 한다. 따라서 const는 값을 한 번 설정하고, 그 값을 유지하면서 사용할 때 필요하다. 

포인터와 const를 함께 사용하게 되면, 상수를 가르키는 포인터(* ptr)가 가르키는 공간은 수정할 수 없는(const) 공간이지만, 상수 변수의 주소를 가르키는 포인터는 수정할 수 있다.

```cpp
int value = 5, value2 = 11;
const int * ptr = &value;
// *ptr = 10; // error! can't change const value
value = 10; // ok!
std::cout << value << " " << *ptr << std::endl; // 10 10
```

* [참고 블로그](https://dydtjr1128.github.io/cpp/2020/01/08/Cpp-const.html)


### bool

bool 변수 선언

```cpp
bool tr = true;
bool fa = false;
```

bool 함수 선언
```cpp
bool b(a,b) {
    if(a == b) return true;
    return false;
}
```

### sort

```cpp
bool compare(string a, string b) {
    return (a + b > b + a);
}

sort(str.begin(), str.end(), compare);
```

compare 함수를 만들어서 sort의 세번째 인자 값으로 넣게 되면, 해당 함수의 반환 값에 맞게 정렬한다. 즉, true가 되도록 정렬하겠다는 의미이다.


### * 포인터

포인터는 특정 값을 저장하는 것이 아니라 메모리 주소를 저장하는 변수다. 

```cpp
int value = 5;
int * ptr = &value;
```

ptr은 값으로 value 변수 값의 주소를 가지고 있다. 그러므로 ptr을 value 변수를 '가리키는' 값이라고 할 수 있다.

### &



### to_string
`to_string(숫자)`를 하게 되면 숫자가 str로 변경된다.

### stoi()
string을 int로 변환하기 위한 std::stoi()가 있다. 그 외에도 stol()(str to long long), stod()(str to double), stof()(str to float)등도 있다.

```cpp
int i = std::stoi(int_val);
```

<br>

# H-index

## 나의 전체 코드

```cpp
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;



int solution(vector<int> citations) {
    int answer = 0;
    
    vector<int> answ;
    
    sort(citations.begin(),citations.end(),greater<int>());
    
    for(auto i : citations)
        cout << i << endl;
    
    for(int i=0;i<citations.size();i++) {
        cout << i << " " << citations[i] << endl;
        if(citations[i] <= i) return i;
    }
}
```

```cpp
입력값 〉	[10, 8, 5, 4, 3]
출력 〉	
10
8
5
4
3

0 10
1 8
2 5
3 4
4 3
```

정렬한 후 차례대로 비교했을 때 i가 증가한다는 것은 특정 값 이상인 논문이 i개 있다는 말과 같다. 즉 i가 3일때는 4보다 큰 값이 3개 있다는 것이고, i가 4일 때 3보다 큰 값이 4개 있으므로 i=4가 h의 최댓값이 된다.

### greater<int>()

sort는 기본적으로 오름차순으로 정렬한다. 내림차순으로 정렬하고자 하면, greater<int>()를 사용하면 된다.

`sort(start,end,greater<int>())` 을 하게 되면 내림차순으로 정렬된다.

