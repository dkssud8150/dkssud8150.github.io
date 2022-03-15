---
title:    "[데브코스] 5주차 - OpenCV abstract and install "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-14 17:27:00 +0800
categories: [Classlog, devcourse]
tags: [opencv, devcourse]
toc: True
comments: True
image:
  src: /assets/img/dev/week5/day1/main2.jpg
  width: 800
  height: 500
---

<br>

# OpenCV 개요

openCV란 Open Source로 개발되고 있는 컴퓨터 비전과 머신러닝에 대한 소프트웨어 라이브러리다.

[공식 사이트](https://www.opencv.org)

## OpenCV를 사용하는 이유
- 무료 사용 : BSD/Apache 2 라이선스를 따르기 때문에 학교/회사 등으로 상업적 사용도 가능
- multiple interface: c, c++, python, java 등 많은 곳에서도 사용가능
- multiple platform: windows, linux,mac OS, IOS, Android
- optimized: 최신 CPU 버전을 지원, 멀티 코어 프로세싱, openCL/CUDA 지원
- 구글 지도, 침입자 감지, 로봇 네비게이션 등 많은 곳에서 사용

## OpenCV 역사

- 1998년: intel 주도로 시작하여 오픈 소스로 개발되었다. 
- 2006년: C로 구현되어 함수, 구조체를 가지고 있었다.
- 2009년: C++로 전환하여 클래스로 영상 데이터를 다루기 시작했다.
- 2015년: OpenCV 프로젝트 구조 개선, GPU, IPP 활용 확대
- 2017년: DNN 모듈 지원
- 2018년: C++ 11 지원, DNN 지원 강화
- 2021년: OPENCV 4.5.5 개발

<br>

## OpenCV 구성

### OpenCV 모듈
- OpenCV는 모듈(module)이라고 부르는 다수의 라이브러리들의 집합이다.
- OpenCV 모듈은 크게 메인 모듈과 추가 모듈로 나눌 수 있다.

<br>

1. 메인 모듈
- 핵심 기능, 널리 사용되는 기능, 기반 기능
- https://github.com/opencv/opencv

| 모듈 이름 | 설명 |
| --- | --- |
| **core** | 행렬, 벡터 등의 opencv 핵심 클래스와 연산 함수를 정의 |
| **imgcodecs** | 영상 파일 입출력 |
| **imgproc** | 필터링, 기하학적 변환, 색 공간 변화 등의 영상 처리 기능 |
| **objdetect** | 얼굴, 보행자 검출 등의 객체 검출 |
| **highgui** | 영상의 화면 출력, 마우스 이벤트 처리 등의 사용자 인터페이스 기능 |
| ml | 머신러닝 알고리즘 |
| dnn | 심층 신경망, DNN 기능 |
| java, js, python | java, js, python 인터페이스를 지원 | 
| video | 옵티컬 플로우, 배경 차분 등의 동영상 처리 기술 |
| videoio | 동영상 파일 입출력 |
| **world** | 여러 OpenCV 모듈을 포함하는 하나의 통합 모듈 |

<br>

2. 추가 모듈
- 최신 기능, 널리 사용되지 않는 기능, 특허가 걸려있는 알고리즘, HW 의존적 기능(CUDA)
- https://github.com/opencv/opencv_contrib/

| 모듈 이름 | 설명 |
| --- | --- |
| cudaxxx | cuda를 이용해서 컴퓨터 비전을 빠르게 동작 |

<br>

### 컴퓨터 비전 처리 과정과 필요한 OpenCV 모듈 구성

1. 영상 입출력 : core,videoio, imgcodecs
2. 전처리 : imgproc, photo
3. 특징 추출 : imgproc, features2d
4. 객체 검출,영상 분할 : imgproc, objdetect
5. 분석(객체 인식, 포즈 추정, 움직임 분석, 3D 재구성) : calib3d, video, stitching, ml, dnn
6. 화면 출력, 최종 판단 : highgui, ml ,dnn

이 모든 모듈을 통합한 `world`라는 모듈을 사용하면 된다. 필요하지 않은 것도 포함되어 용량이 커지긴 하나 편리하기 때문에 이를 많이 사용한다.

<br>

> openCV 관련 사이트
>
> opencv 사이트 : [http://opencv.org/](http://opencv.org/)
>
> opencv 4.x에 대한 도움말 : [https://docs.opencv.org/4.x/](https://docs.opencv.org/4.x/)
>
> opencv에 대한 질문과 답변 사이트 : [https://forum.opencv.org/](https://forum.opencv.org/)
>
> opencv 메인 모듈 깃허브 : [https://github.com/opencv/opencv/](https://github.com/opencv/opencv/)
>
> opencv 추가 모듈 깃허브 : [https://github.com/opencv/opencv_contrib/](https://github.com/opencv/opencv_contrib/)
>
> 페이스북 그룹 : [https://www.facebook.com/groups/opencvprogramming](https://www.facebook.com/groups/opencvprogramming)

<br>

## OpenCV 설치

OpenCV 설치 : opencv 헤더 파일, LIB 파일, DLL 파일을 컴퓨터에 생성하는 작업

앞의 두 파일은 OpenCV 프로그램을 빌드할 때 필요한 파일이고, DLL 파일은 OpenCV 프로그램을 실행할 때 필요한 파일이다.

- 설치 방법
1. 설치 실행 파일 이용
    - 장점
        - 설치가 빠르고 간단
        - 미리 빌드된 DLL, LIB 파일 제공
    - 단점
        - OpenCV 추가 모듈이 미지원
        - Windows 64비트 운영 체제만 지원
2. 소스 코드 직접 빌드
    - 장점
        - 자신의 시스템 환경에 최적회된 DLL, LIB 파일을 생성 가능
        - 원하거나 원치 않는 옵션을 선택이 가능 (extra modules, parallel_for backend, etc)
    - 단점
        - 빌드 작업이 복잡하고 시간이 오래 걸린다.


### 1. 설치 실행 파일 활용

1.설치 파일 다운로드 : [https://opencv.org/releases](https://opencv.org/releases) 또는 [https://github.com/opencv/opencv/releases](https://github.com/opencv/opencv/releases) 로 가서 자신의 운영체제 클릭하면 opencv-4.5.5-vc14_vc15.exe 파일 다운로드이 가능하다.

<img src="/assets/img/dev/week5/day1/opencvsite.png" width="50%">
<img src="/assets/img/dev/week5/day1/opencvsite2.png" width="50%">

위치를 정하고 extract를 클릭하면 된다.

```markdown
opencv
    ⊢ build     // dll, lib파일, header파일이 저장
        ⊢ include
            ∟ opencv2
                ∟ ... // header들
        ∟ x64
            ⊢ vc14  // visual studio 15 
            ∟ vc15  // visual studio 17을 이용해서 빌드된 파일들
                    // 2022에서도 사용 상관 없음
                ⊢ bin   // dll 파일들 존재
                    ⊢ ...
                    ⊢ ~55.dll // 릴리즈용으로 빌드된 파일
                    ∟ ~55d.dll // 디버깅용으로 빌드된 파일
                ∟ lib
                    ⊢ ...
                    ⊢ ~55.lib // 릴리즈용으로 빌드된 파일
                    ∟ ~55d.lib // 디버깅용으로 빌드된 파일
	∟ sources   // opencv 메인 모듈 소스 코드
```

<br>

2.환경 변수 등록

2-1. 환경 변수 새로 만들기

opencv를 편리하게 사용하기 위해서는 opencv 폴더 위치를 시스템 환경 변수에 등록해야 한다. 그래야 opencv를 사용하는 프로그램이 dll파일을 찾아서 정상적으로 실행된다.

`설정 -\> 시스템 -\> 정보 -\> 고급 시스템 설정 -\> 환경 변수` 로 이동해서 \<user-id\>에 대한 사용자 변수 새로 만들기 선택

<img src="/assets/img/dev/week5/day1/envsetting.png">

<img src="/assets/img/dev/week5/day1/envsetting2.png">

- 변수 이름: OPENCV_DIR
- 변수 값: C:\opencv\build

2-2. 환경 변수 PATH 편집

이를 하는 이유는 다른 라이브러리를 사용할 때 더 효과적으로 사용할 수 있기 때문이다.

<img src="/assets/img/dev/week5/day1/envsetting3.png" width="50%">
<img src="/assets/img/dev/week5/day1/envsetting4.png" width="50%">

이 후 다 확인하면 등록이 된다. 등록 확인을 위해 명령 프롬포트를 켜서 `opencv_version`을 타이핑하면 버전이 나온다.

```bash
> opencv_version
4.5.5
```

<br>

<br>

# OPENCV 프로그램 개발 환경 설정

## HelloCV 프로젝트 만들기

1.Visual C++ 새 프로젝트 만들기

2.빈 프로젝트 클릭

3.프로젝트 이름: HelloCV, 위치는 임의이 폴더, 하단의 `솔루션 및 프로젝트를 같은 디렉토리에 배치` 를 클릭
    - 솔루션은 여러 개의 프로젝트를 관리할 수 있는 컨테이너라고 할 수 있다.
    - 새 프로젝트를 만드는데, 이 프로젝트를 담기 위한 솔루션이 만들어진다.
    - 좀 복잡한 프로그램을 만들 경우 솔루션을 다른 곳에 만들기도 한다.

4.만들기 클릭

<img src="/assets/img/dev/week5/day1/mkpro.png">

<br>

5.생성된 파일로 가보면 sln 파일이 있다. 이것이 솔루션 파일, vcxproj 파일이 프로젝트 관리 파일이다. 나머지는 visual studio에서 관리를 위한 파일이다.

6.프로젝트 -\> 새 항목 추가 -\> c++ 파일을 클릭

<br>

7.아래 소스코드 작성

```cpp
#include <iostream>
#include "opencv2/opencv.hpp" // opencv관련 헤더 파일, 
// 이는 설치한 opencv/bulid/include/opencv2/opencv.hpp 가 있다.

int main()
{
	std::cout << "Hello OpenCV " << CV_VERSION << std::endl;
}
```

<br>

8.opencv 헤더 파일 위치 지정

현재에는 아직 opencv.hpp를 인식하지 못하여 불러오지 못한다. 그래서 직접 지정을 해줘야 한다.

- 프로젝트 -\> HelloCV 속성 -\> 구성 속성 -\> C/C++ -\> 일반
- 추가 포함 디렉토리 항목에 `$(OPENCV_DIR)\include` 입력

이 때, 절대적 위치를 작성해줄 수 있지만, 이전에 시스템에 환경 변수를 등록했었다. `$(OPENCV_DIR)`이란 OPENCV_DIR이라는 이름의 환경변수를 불러오겠다는 것이다. 환경 변수를 지정해주지 않았다면 오류가 날 것이다.

여기서 상단에 구성을 보면 활성(debug) 이외에도 releases가 있을 것이다. 디버그 모드는 개발하는 동안 디버깅하면서 버그를 찾고 에러를 수정하는 모드이고, 릴리즈모드는 개발이 끝나서 프로그램을 배포할 때 사용하는 모드다. dedug를 클릭해서 수정한다. 

또한, 구성 옆에 그 옆에 플랫폼도 x64가 맞는지 잘 확인한다.

<br>

9.opencv lib 파일 위치 지정

- 프로젝트 -\> HelloCV 속성 -\> 구성 속성 -\> 링커 -\> 일반
- 추가 라이브러리 디렉토리 항목에 `$(OPENCV_DIR)\x64\vc15\lib` 입력

<br>

10.추가 종속성

- 프로젝트 -\> HelloCV 속성 -\> 구성 속성 -\> 링커 -\> 입력
- 추가 종속성 항목에 화살표를 클릭하고 편집을 클릭한 후 `opencv_world455d.lib`를 입력한다. 이 때, 구성이 debug이어서 455d.lib을 지정해줬다. 만약 릴리즈 모드라면 d를 뺀 `455.lib`을 입력해야 한다.

릴리즈모드도 따로 동일하게 해줘야 한다. 

<br>

이렇게 설정을 마치고 솔루션 빌드를 누르면 아래 오류가 있는지 확인할 수 있다. 오류가 없으면 `C:\coding\opencv\HelloCV\x64\Debug`에 HelloCV.exe 파일이 생성되었다고 나올 것이다. 실제로 파일 탐색기로 들어가보면 파일이 존재하고 이를 더블클릭하면 바로 꺼지면서 확인이 어렵다. 

따라서 파일 주소창에 `cmd`를 입력하면 프롬포트에 현재 주소 기준으로 켜질 것이다. 여기서 `HelloCV.exe`를 입력하면 버전이 나온다.

```bash
> HelloCV.exe
Hello OpenCV 4.5.5
```

또는 편리하게 상단에 디버그 -\> 디버깅 시작 또는 디버깅하지 않고 시작 을 누르면 바로 나온다. 그러나 현재는 종단점을 설정하지 않아서 바로 프로세스가 종료된다.

<br>