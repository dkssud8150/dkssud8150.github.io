---
title:    "[데브코스] 5주차 - OpenCV Print image file and Setting for OpenCV "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-14 20:15:00 +0800
categories: [Classlog, devcourse]
tags: [opencv, devcourse]
toc: True
comments: True
image:
  src: /assets/img/dev/week5/day2/main.png
  width: 800
  height: 500
---

<br>

# 영상 파일 불러오기

1. 가지고 있던 lenna.bmp 파일을 불러와서 화면에 출력하는 OpenCV 예제 프로그램
2. 편의상 이전 강의에서 작성한 HelloCV 프로젝트에 필요한 소스 코드를 추가

아래는 lenna.bmp 파일이다.

<img src="/assets/img/dev/week5/day2/lenna.bmp" cation="lenna.bmp">

<br>

현재 폴더에 있는 lenna.bmp 파일을 불러와서 화면에 출력하는 OpenCV 예제 코드다.

```cpp
#include <iostream>             
#include "opencv2/opencv.hpp"   // OpenCV 관련 헤더 파일을 include

using namespace cv;             // cv와 std 네임스페이스를 사용하도록 설정
using namespace std;            // 원래는 std::, cv:: 으로 사용해야 하지만 이를 사용하지 않기 위해

int main()
{
    Mat img = imread("lenna.bmp"); // lenna.bmp 파일을 불러와서 img에 저장
                                    // mat(matrix)은 행렬을 표현하는 클래스

    if (img.empty()) { // 영상 파일 불러오기를 실패하면 에러 메시지를 출력하고 프로그램을 종료
        cerr << "Image load failed!" << endl; 
        return -1;
    }

    namedWindow("Image"); // image 라는 이름의 새 창을 만듦
    imshow("Image", img); // 여기에 img 영상을 출력
    waitKey();           // 키보드 입력이 있을 때까지 프로그램을 대기, 
    destroyAllWindows(); // 키 입력이 있으면 모든 창을 닫고 종료
}
```

이 코드를 실행하면 아래와 같이 나올 것이다.

<img src="/assets/img/dev/week5/day2/imple.png">

<br>

## OpenCV 주요 함수 설명

1.영상 파일 불러오기 

```cpp
Mat imread(const String& filename, int flags = IMREAD_COLOR);
```

- filename : 불러올 영상 파일 이름 e.g. "lenna.bmp", "C:\\lenna.bmp"
- flags : 영상 파일 불러오기 옵션 플래그
    - IMREAD_UNCHANGED : 영상 속성 그대로 읽어오기 e.g. 투명한 PNG 파일은 4채널(B,G,R,α)
    - IMREAD_GRAYSCLAE : 1채널 grayscale 영상으로 읽기
    - IMREAD_COLOR(default) : 3채널 BGR 컬러 영상으로 읽기
- 반환값 : 불러온 영상 데이터 (Mat 객체)

<br>

2.비어 있는 Mat 객체 확인

```cpp
bool Mat::empty() const
```

- 반환값 : rows, cols, data 멤버 변수가 0이면 true 반환

<br>

3.영상 파일 저장하기

```cpp
bool imwrite(const String& filename, InputArray img, const std::vector<int>& params = std::vector<int>());
```

- filename : 저장할 영상 파일 이름, 파일 이름에 포함된 확장자를 분석하여 해당 파일 형식으로 저장된다.
- img : 저장할 영상 데이터 (Mat 객체)
- params : 파일 저장 옵션 지정 (속성 & 값의 정수 쌍)
  - 예를 들어, JPG 압축율을 90%로 하고자 하면 {IMWRITE_JPEG_QUALITY,90} 을 지정한다.
- 반환값 : 정상적으로 저장하면 true, 실패하면 false

<br>

```cpp
imwrite("lenna.png", img);
```

이를 실행하면 현재 폴더에 lenna.png 파일이 생겨날 것이다.

<br>

4.새 창 띄우기

```cpp
void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);
```

- winname : 창 고유 이름, 이 이름으로 창을 구분한다.
- flags : 창 속성 지정 플래그
  - WINDOW_NORMAL : 영상 크기가 창 크기에 맞게 지정됨
  - WINDOW_AUTOSIZE(default) : 창 크기가 영상 크기에 맞게 자동으로 변경됨
  - WINDOW_OPENGL : OpenGL 이 지원됨

> 이 namedwindow는 반드시 선언해야 하는 것은 아니다. 저 아래에 imshow라는 코드가 image라는 창을 여는데, 이가 존재하지 않는다면 기본 형태의 창을 생성하고 출력한다.

<br>

5.창 닫기

```cpp
void destroyWindow(const String& winname);
void destroyAllWindows();
```

- winname : 닫고자 하는 창 이름

> 참고사항 - 일반적인 경우 프로그램 종료 시 운영 체제에 의해 열려 있던 모든 창이 자동으로 닫힌다. 그래서 destroyAllwindows()를 사용하지 않아도 작동한다.

<br>

6.창 위치 지정

```cpp
void moveWindow(const String& winname, int x, int y);
```

- winname : 창 이름
- x,y : 이동할 위치 좌표 

<br>

7.창 크기 지정

```cpp
void resizeWindow(const String& winname, int width, int height);
```

- winname : 창 이름
- width, height : 변경할 창 크기

> 참고사항 - 윈도우가 WINDOW_NORMAL 속성으로 생성되어야 동작한다.

<br>

8.영상 출력하기

```cpp
void imshow(const String& winname, InputArray mat);
```

- winname : 영상을 출력할 대상 창 이름
- mat : 출력할 영상 데이터 (Mat 객체)

- 영상 출력 방식
  - 8bit unsigned: 픽셀 값 그대로 출력
  - 16bit unsigned or 32bit **integer**: 픽셀 값을 255로 나눠서 grayscale값으로 출력
  - 32bit(float) or 64bit(double) floating point: 픽셀값에 255를 곱해서 grayscale값으로 출력
  - \*\* 따라서 8bit 형태로 출력해야 안전하게 출력할 수 있다.

> 참고사항 - 만약 winname에 해당하는 창이 없으면 WINDOW_AUTOSIZE 속성의 창을 새로 만들고 영상을 출력한다.
>
> 실제로는 waitkey() 함수를 호출해야 화면에 영상이 나타난다.

<br>

> visual studio에서 코드 번호줄 오른쪽에 회색을 클릭하면 해당 줄에 빨간색 점이 생길 것이다. 이를 종단점이라 하는데, 즉, 해당 코드 전까지 실행되다가 멈추겠다는 것이다. 이를 하려면 `디버깅 시작`을 누르게 되면 해당 줄에서 멈추게 된다. 이 다음부터 디버그 -\> 프로시저 단위 실행을 하면 해당 줄을 실행하는 것이다. 이렇게 하면 빨간 점 또는 그 밑에 화살표가 생긴다. 이는 단위 실행을 하면 이 화살표의 줄이 실행된다는 것이다.

<br>

9.키보드 입력 대기

```cpp
int waitKey(int delay = 0);
```

- delay : 밀리초 단위의 대기 시간 delay <= 0 이면 무한히 기다린다.
- 반환값 : 눌린 키의 아스키 값, 키가 눌리지 않으면 1

> 참고 사항 - waitkey()함수는 opencv 창이 하나라도 있어야 정상 동작한다.
>
> imshow() 함수 호출 후에 waitkey()함수를 호출해야 영상이 화면에 나타난다.
>
> 주요 특수 키의 아스키 값 : ESC==27, ENTER==13, TAB==9

```cpp
while (True) {
  if (waitKey() == 27) // q키 == 'q',  spacebar == ' '
    break;
}
```

<br>

<br>

# OpenCV API 도움말

[opcnCV 도움말 사이트](http://docs.opencv.org/)

도움말 웹페이지에서 우측 상단 검색창을 활용하여 검색하면 된다.

<br>

<br>

# 이미지 파일 형식 변환 프로그램 제작

영상 파일을 다른 형식으로 변환하여 저장하는 프로그램을 작성해보고자 한다.

명령행 인자로 입력 받아야 한다.

```cpp
#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    // 명령행 인자 개수가 3개보다 작으면 사용법을 출력하고 종료
    if (argc < 3) {
        cout << "Usage: ocvrt.exe <src_image> <dst_image>" << endl;
        return 0;
    }

    // 첫번째 이미지 파일을 imread() 함수로 읽어서 img 변수에 저장
    Mat img = imread(argv[1]);

    // 두번째 이미지 파일 이름으로 img 영상을 저장
    // 저장이 제대로 되면 ret 변수에 true를, 실패하면 false를 저장
    bool ret = imwrite(argv[2], img);

    if (ret) {
        cout << argv[1] << " is successfully saved as " << argv[2] << endl;
    }
    else {
        cout << "File save failed!" << endl;
    }
}
```

<img src="/assets/img/dev/week5/day2/implement.png">

- [참고 블로그](https://mndd.tistory.com/141)

<br>

<br>

# Visual Studio에서 OpenCV 편하게 사용하기

## ImageWatch 확장 프로그램

ImageWatch란 OpenCV Mat 데이터를 이미지 형태로 보여주는 visual studio 확장 프로그램이다. OpenCV 프로그램 디버깅시 유용하다.

**ImageWatch 설치**
1. Visual studio 메뉴에서 [확장] -\> [확장 관리] 선택
2. 우측 상단 검색창에 `opencv` 입력
3. Image Watch for Visual Studio 항목에서 [다운로드] 클릭
4. Visual Studiop를 재시작하면 설치된다.

<br>

**사용 방법**
1. HelloCV.cpp 파일에서 **imread()** 이후 코드에 종단점(F9) 설정 후 디버깅 시작(F5)
2. [보기] -\> [다른 창] -\> [Image Watch] 메뉴 선택
3. Image Watch 창에서 Mat 형식의 변수를 이미지 형태로 확인 가능
4. 확대/축소 및 픽셀 값 확인 가능

<img src="/assets/img/dev/week5/day2/imagewatch.png">

<br>

## OpenCV 프로젝트 템플릿의 정의

프로젝트 템플릿이란?
- 프로젝트 속성, 기본 소스 코드 등이 미리 설정된 프로젝트를 자동으로 생성하는 기능
- Visual Studio의 템플릿 내보내기 마법사를 통해서 ZIP파일로 패키징된 자신만의 템플릿 파일을 생성할 수 있다.

OpenCV 프로젝트 템플릿이란?
- OpenCV 개발을 위한 추가 포함 디렉토리, 추가 라이브러리 디렉토리, 추가 종속성 등이 미리 설정되어 있는 콘솔 응용 프로그램 프로젝트를 생성
- OpenCV 기본 소스 코드(main.cpp), 테스트 영상 파일(lenna.bmp)파일도 함께 생성이 가능하다.

<br>

## OpenCV 프로젝트 템플릿 만들기

1. OpenCVTemplate 이름의 프로젝트 생성
- main.cpp 파일 추가 & 코드 작성
- lenna.bmp 파일 추가
- 프로젝트 속성에서 OpenCV 설정(debug, releases) -\> 빌드 및 프로그램 동작 확인

<img src="/assets/img/dev/week5/day2/templateimple.png">

2. [프로젝트] -\> [템플릿 내보내기] 메뉴 선택
- 프로젝트 템플릿 선택 -\> 템플릿 이름 및 설명 작성 후 [마침]

<img src="/assets/img/dev/week5/day2/temexport.png">
<img src="/assets/img/dev/week5/day2/temexport2.png">
<img src="/assets/img/dev/week5/day2/temexport3.png">

3. `c:\Users\<user_id>\Documents\Visual Studio 2022\Templates\PorjectTemplates\` 폴더에 있는 OpenCVTemplates.zip 파일을 수정
- main.cpp & lenna.bmp 파일 추가 (안되어 있을 수도 있기에)
- \+ MyTemplate.vstemplate 파일 편집 

4. Visual Studio에서 새 프로젝트를 만들 때 해당 프로젝트 템플릿을 선택하여 사용

<img src="/assets/img/dev/week5/day2/export.png">

이 때 설명이 너무 단순하기에 더 자세하게 커스터마이징할 수 있다. **MyTemplate.vstemplate**를 수정해주면 된다. 이를 텍스트 편집기로 들어가면 된다.

아래는 원래 우리가 가지고 있던 내용이다.

<img src="/assets/img/dev/week5/day2/vstemplate.png">

<br>

```xml
<VSTemplate Version="3.0.0" xmlns="http://schemas.microsoft.com/developer/vstemplate/2005" Type="Project">
  <TemplateData>
    <Name>OpenCV 콘솔 응용 프로그램</Name>
    <Description>OpenCV 콘솔 응용 프로그램을 생성합니다.</Description>
    ...
      <ProjectItem ReplaceParameters="false" TargetFileName="main.cpp">main.cpp</ProjectItem>
      <ProjectItem ReplaceParameters="false" TargetFileName="lenna.bmp">lenna.bmp</ProjectItem>
    </Project>
  </TemplateContent>
</VSTemplate>
```

name태그와 description을 수정해주고, projectitem부분이 잘 생성되어 있는지 확인하고, 잘 되어 있지 않으면 추가해준다.

이것들을 수정하고 다시 zip으로 묶어 놓는다. 그 후 visual studio에 들어가서 프로젝트 생성을 보면 아래와 같이 잘 나오는 것을 볼 수 있다.

<img src="/assets/img/dev/week5/day2/complete.png">

<br>

<br>

