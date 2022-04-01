---
title:    "Coding Test[C++] - 완전탐색"
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

프로그래머스에 기재되어 있는 c++ coding test에 대해 리뷰하고자 합니다. 그 중에서도 연습문제에 해당하는 문제를 이 곳에 작성할 예정입니다. 다른 것을 참고하시려면 아래 링크를 클릭하시기 바랍니다.

1. [해시](https://dkssud8150.github.io/classlog/codingtestchash.html)
2. [정렬](https://dkssud8150.github.io/classlog/codingtestcsort.html)
3. [연습문제](https://dkssud8150.github.io/classlog/codingtestcpra.html)
4. [카카오 블라인드 채용 문제](https://dkssud8150.github.io/classlog/codingtestckakao.html)

<br>

## overview

### 완전탐색

수학적으로 알고리즘을 생각한 후 그것을 코드로 표현하는 문제가 많은 듯하다.



<br>

### Index

1. 모의고사
2. 사전순 부분 문자열

<br>

# 모의고사

## 나의 전체 코드

```cpp
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

vector<int> solution(vector<int> answers) {
    vector<int> answer;
    
    vector<int> f = {1,2,3,4,5};
    vector<int> s = {2,1,2,3,2,4,2,5};
    vector<int> t = {3,3,1,1,2,2,4,4,5,5};
    
    vector<int> count(3);
    
    for(auto i=0;i<answers.size();i++) {
        if(answers[i] == f[i%f.size()]) count[0]++;
        if(answers[i] == s[i%s.size()]) count[1]++;
        if(answers[i] == t[i%t.size()]) count[2]++;
    }
    
    int count_max = *max_element(count.begin(), count.end());
    
    for(int i=0;i<count.size();i++) {
        if(count[i] == count_max) answer.push_back(i+1);
    }
    
    return answer;
}
```

### *max_element()

`max_element(start,end)`는 vector 중에서 [start, end) 구간에서 가장 큰 값을 추출하는데, 이는 cout << endl; 에서 사용이 되기 때문에 포인터와 함께 써야 한다.

```cpp
cout << max_element(count.begin(), count.end()) << endl;

int count_max = *max_element(count.begin(), count.end());
```

<br>

<br>

# 사전순 부분문자열

- string과 사전순의 특성을 사용하여 진행

```cpp
#include <iostream>
#include <vector>

using namespace std;

// 테스트를 위한 변수 선언
string ss{ "xyb" };

// 미리 메모리 크기를 지정해서 변수 선언
string answer{ "" };

string solution(string ss, string answer)
{	
	// 출력값을 담을 변수 선언
	string lists;

	for (auto s : ss) {
		// 에러를 방지하기 위해 비어 있는지를 체크하고, 
		// 현재 값이 순서상 가장 맨 뒤가 될 때까지 맨 뒤 값들을 뺀다.
		while (!lists.empty() && lists.back() < s)
			lists.pop_back();

		// 처리가 완료된 리스트에 삽입
		lists.push_back(s);
	}
	return lists;
}

int main()
{	
	// 정답 확인을 위한 변수 선언
	string correction;

	// 입력 인자를 통해 정답을 입력
	cin >> correction;

	// 출력한 리스트와 정답이 같으면 correct, 아니면 No를 출력
	if (solution(ss, answer) == correction)
		cout << "correct" << endl;
	else
		cout << "No" << endl;
}
```

이 문제에서는 특별한 함수나 익숙치 않은 메서드를 사용한 것이 아니라, 사전 순에 따라 알고리즘을 생각해내는 것이 중요했다.