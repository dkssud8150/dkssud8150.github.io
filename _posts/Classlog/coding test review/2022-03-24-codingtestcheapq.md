---
title:    "Coding Test[C++] - 덱(deque)"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-03-24 01:10:00 +0800
categories: [Classlog,coding test review]
tags: [coding test, deque]
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
4. [연습문제](https://dkssud8150.github.io/classlog/codingtestcpra.html)
5. [카카오 블라인드 채용 문제](https://dkssud8150.github.io/classlog/codingtestckakao.html)

<br>

# 회전하는 큐

```cpp
#include <iostream>
#include <deque>
#include <vector>
#include <algorithm>

using namespace std;

int num, N, M, n;

deque<int> dq;

int main(void) {
	//  테스트를 위해 작성한 lists
	//	vector<vector<int>> lists{{1,2,3},{2,9,5},{27,16,30,11,6,23},{1,6,3,2,7,9,8,4,10,5}};

	//  lists의 몇번째 데이터인지 입력
	//	cout << "Number : ";
	//	cin >> num;

	//  입력한 숫자에 대한 vector 가져오기
	//	vector<int> index = lists[num-1];

	//  배열 크기와 출력하고 싶은 원소 개수 입력
	cin >> N >> M;

	//  N 크기의 배열 생성
	for (auto i = 1; i <= N; ++i) {
		dq.push_back(i);
	}

	int cnt{ 0 };

	for (auto k = 0; k < M; k++) {
		// 빼내고 싶은 원소 번호를 입력
		cin >> n;

		// 빠른 탐색을 위해 원소 번호의 위치 계산
		int range = distance(dq.begin(), find(dq.begin(), dq.end(), n));

		// deque의 크기 지정
		int size = dq.size();

		// 원소의 위치가 deque의 크기/2 보다 작다면 앞에서 탐색
		if (range <= int(size / 2)) {

			// 맨 앞에서 빼낼 수 있으므로 원하는 원소가 맨 앞에 올때까지 뒤로 미루기
			while (n != dq.front()) {
				dq.push_back(dq.front());
				dq.pop_front();
				++cnt;
			}
		}

		// 원소 위치가 deque의 크기/2 보다 크다면 뒤에서 탐색
		else {
			// 빼낼 수 있는 곳이 맨 앞뿐이므로 빼내고 싶은 원소를 맨 앞으로 이동할때까지 앞으로 이동시키기 
			while (n != dq.front()) {
				dq.push_front(dq.back());
				dq.pop_back();
				++cnt;
			}
		}

		// 탐색이 끝나면 맨 앞에 원하는 원소가 있을 것이므로 빼내기
		dq.pop_front();
	}
	cout << cnt;
}
```

lists에 대한 것들은 다 테스트를 위해 작성한 부분들이다. lists를 사용한 정확한 코드는 다음과 같다.

```cpp
#include <iostream>
#include <deque>
#include <vector>
#include <algorithm>

using namespace std;

int num, N, M, n;

deque<int> dq;

int main()
{
    vector<vector<int>> lists{{1,2,3},
                                {2,9,5},
                                {27,16,30,11,6,23},
                                {1,6,3,2,7,9,8,4,10,5}};

    cout << "list Number : ";
    cin >> num;

    vector<int> index = lists[num-1];

	cin >> N >> M;

	for (auto i = 1; i <= N; ++i) {
		dq.push_back(i);
	}

	int cnt{ 0 };

	for (auto n : index) {

		int range = distance(dq.begin(), find(dq.begin(), dq.end(), n));

		int size = dq.size();

		if (range <= int(size / 2)) {
			while (n != dq.front()) {
				dq.push_back(dq.front());
				dq.pop_front();
				++cnt;
			}
		}
		else {
			while (n != dq.front()) {
				dq.push_front(dq.back());
				dq.pop_back();
				++cnt;
			}
		}

		dq.pop_front();
	}
	cout << cnt;
}
```

### deque

회전하는 자료형을 사용하기 위해 deque를 사용했다. deque에서는 vector에 있는 push_back()과 pop_back()이외에, push_front(), pop_front()도 지원한다. 그리고 `++i` 처럼 앞에다 ++를 작성해준 이유는 메모리 관점에서 뒤에 작성하게 되면 메모리상에 i가 존재하고, i+1이 또 생성되는 반면 앞에다 적어주면 이러한 불필요를 줄일 수 있기 때문이다.

<br>

### distance()

시작위치부터 특정 원소의 위치를 추출해주는 함수인데, 간단하게 특정원소의 위치를 찾아준다. 사용 인자는 다음과 같다.

```cpp
int std::distance<iterator _First, iterator _Last);
```

- _First : 거리를 측정할 시작 위치
- _Last : 거리를 측정할 마지막 위치
- 반환값 : _Last - _First;

이것과 find 함수를 사용하여 

```cpp
int distance(dq.begin(), find(dq.begin(), dq.end(), n));
```

시작 위치부터 원소 n의 위치까지의 거리를 반환할 수 있다. 예를 들어, [1,2,3,4] 배열이 있고, 3의 위치를 distance로 찾으면 2을 반환해준다. 

<br>

### cin >> m

입력 스트림으로 출력 스트림의 반대로 입력을 받는다.  >> 연산자는 피연산자의 타입에 따라 적절하게 입력값을 가공하여 넣어준다. int나 double을 입력데이터를 받는다면 입력 데이터를 수로 판단하여 가공한다. 위 코드를 실행하면 입력을 받게 되고, 입력을 하면 그 값이 m으로 들어간다. m의 타입은 >> 연산자가 알아서 판단해준다.

