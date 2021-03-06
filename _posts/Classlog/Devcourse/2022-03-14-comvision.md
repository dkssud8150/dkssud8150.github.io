---
title:    "[데브코스] 5주차 - Computer Vision abstract and image analysis about memory allocation "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-14 13:34:00 +0800
categories: [Classlog, devcourse]
tags: [computer-vision, devcourse]
toc: True
comments: True
image:
  src: /assets/img/dev/week5/day1/main.png
  width: 800
  height: 500
---

<br>

# 개요

## 컴퓨터 비전(computer vision)이란?

컴퓨터를 이용하여 정지 영상 또는 동영상으로부터 의미 있는 정보를 추출하는 방법을 연구하는 학문이다.

<br>

## 컴퓨터 비전과 영상 처리(image processing)

영상(image)란 사진이나 비디오 등을 다 포함한 것을 말한다. 동영상을 말할 때는 video라고 한다. 영상 처리는 영상을 입력으로 받아 화질을 개선하겆나 크롭핑하여 다시 영상을 출력하는 것을 말하고, 거기서 객체를 인식하는 등의 추가 작업을 컴퓨터 비전이라 한다. 그러나 뜻이 거의 동일하기에 동일시 하는 곳도 많다.

<br>

## 컴퓨터 비전의 역사

1960년 MIT The summer vision project 연구가 컴퓨터 비전의 시초이다. 이 때, 위성으로부터 전송 받은 달 표면 사진의 화질을 복원했다. 현재는 2012년에 딥러닝 모델인 AlexNet이 나오게 되면서 딥러닝이 기하급수적으로 발전되었다.

<br>

- 컴퓨터 비전 vs 휴먼 비전

사람은 주변의 영향을 많이 받기 떄문에 정확한 그곳의 색상을 보기 보다는 주변의 상황과 함께 판단해서 어디가 어두운지, 밝은지를 판단한다. 그러나 컴퓨터는 해당 위치의 픽셀을 추출해서 판단하기 때문에 객관적으로 판단할 수 있을 것이다.

<br>

# 컴퓨터 비전의 응용 분야

## 영상의 화질 개선

카메라로 찍은 사진을 더욱 선명하게 만들거나 색상을 원하는 형태로 변경하는 등의 작업을 한다. RAW 영상의 변환, 사진앱의 필터, 잡은 제거, HDR, 초해상도 등을 할 수 있다.

<img src="/assets/img/dev/week5/day1/superresolution.jpeg">

## 내용 기반 영상 검색 (content-based image/video retrieval)

영상에 존재하는 사람, 사물, 색상 정보 등을 인식하여 유사한 영상을 자동으로 찾아주는 시스템이다. 이를 비주얼 검색(visual search)라고도 한다.

<img src="/assets/img/dev/week5/day1/retrieval.jpg">

## 얼굴 검출 및 인식

얼굴 검출(face detection): 영상에서 얼굴의 위치와 크기를 찾는 기법이다.

얼굴 인식(face recognition): 검출된 얼굴이 누구인지를 판단하는 기술이다.
- 미세한 표정 변화도 감지한다.
- 조명 변화, 안경 착용, 헤어 스타일 변화에 의해 정확도가 낮아질 수는 있다.

<img src="/assets/img/dev/week5/day1/facerecog.jpg">

## 의료 영상 처리

x-ray, CT에 사용되는 영상처리이다. 영상의 화질 개선, 영상의 자동 분석을 할 수 있다.

<br>

## 광학 문자 인식

영상에 있는 텍스트를 인식한다. 이를 OCR(optical character recognition)이라 한다. 번역, 자동차 번호판 인식 등을 할 수 있다.

<br>

## 마커 인식

정해진 형태의 마커를 인식하여 숨겨진 정보를 추출할 수 있다. 2D 바코드, QR 코드를 인식할 수 있다.

<br>

## 영상 기반 증강 현실

카메라로 특정 사진을 가리키면 관련된 정보가 증강되어 나타나는 기술이다. 마커 기반과 비 마커 기반으로 나뉜다.

<br>

## 머신 비전(machine vision)

주로 산업계에서 제품의 위치 확인, 측정, 불량 검사 등을 위해 사용되는 영상 기반 기술이다. 공장의 자동화를 촉진시킬 수 있다. 빠른 처리 시간, 높은 정확도, 객관성이 장점이다. 카메라, 렍, 조명, 필터, 영상 보드, 영상 처리 소프트웨어 등으로 구성되어 있다.

<br>

## 인공지능 서비스

입력 영상을 객체와 배경으로 분할하여 객체와 배경을 인식하고 그 후 상황을 인식할 수 있다.
 이를 바탕으로 로봇과 자동차의 행동을 지시할 수 있다.

computer vision + sensor fusion + deep learning 이라 할 수 있다. 이를 상용화한 것이 amazon go\/ 구글,테슬라의 자율 주행 자동차가 있다.

<br>

<br>

# 영상 데이터의 구조와 특징

## 영상(image)이란?

픽셀이 바둑판 모양의 격자에 나열되어 있는 형태로 2차원 행렬 형태를 가지고 있다. 이 때 픽셀이랑 영상의 기본 단위이다.

<img src="/assets/img/dev/week5/day1/pixel.png">

<br>

## Grayscale Image

영상 데이터에는 크게 2가지 종류가 있는데, 하나는 grayscale, 즉 흑백 사진처럼 색상 정보가없이 오직 밝기 정보만으로 구성된 영상을 말한다. 밝기 정보를 256단계로 표현한다.

<img src="/assets/img/dev/week5/day1/grayscale.jpg">

- 그레이스케일 영상의 픽셀값 표현

<img src="/assets/img/dev/week5/day1/grayscalebar.jpeg">

그레일스케일 영상에서 하나의 픽셀은 0~255 사이의 정수 값을 가진다. 0이 검정색, 255가 흰색을 나타낸다.

grayscale level: 그레일 스케일의 픽셀이 가질 수 있는 범위를 나타내는 것으로 [0,255] 또는 [0,256) 이라 표현할 수 있다. 즉 []는 포함, ()는 미포함이라 할 수 있다.

<br>

이를 c/c++에서는 unsigned char로 표현할 수 있다. 이는 1byte 공간을 사용한다.

```cpp
typedef unsigned char BYTE;     // windows
typedef unsigned char uint8_t;  // linux
typedef unsigned char uchar;    // opencv
```

이 unsigned char을 그냥 사용하지 않고, 편의성과 통일성을 위해 typedef를 사용하여 쓴다. 각각 사용하는 곳이 다르다.

<br>

- 픽셀 값 분포의 예

<img src="/assets/img/dev/week5/day1/cameraman.jpg">

이는 영상 처리에서 유명한 사진인 `camera man` 사진이다. 이 때, 픽셀 값을 보면 어두운 값이 0에 가깝고, 밝을수록 255에 가까울 것이다. 그리고 자세히 보게 되면 우리는 구분하지 못하지만, 값이 10~20정도 차이가 나는 것을 볼 수 있다.

<br>

## truecolor image

컬러 사진처럼 다양한 색상을 표현할 수 있는 영상을 말한다. R/G/B 색 성분을 각 256 단계로 표현한다.

<img src="/assets/img/dev/week5/day1/truecolor.png">

<br>

- 트루컬러 영상의 픽셀값 표현

R/G/B 색 성분의 크기를 각각 0.~255 범위의 정수로 표현한다. 0은 없는 상태, 255는 가득있는 상태를 말한다.

<img src="/assets/img/dev/week5/day1/truecolorcircle.png">

<br>

이를 c/c++에서는 unsigned char로 자료형 3개 있는 배열 또는 구조체로 표현한다. 따라서 3byte를 차지하게 된다.

```cpp
class RGB
{
    unsigned char R;
    unsigned char G;
    unsigned char B;
}
```

<br>

- 픽셀 값 분포의 예

<img src="/assets/img/dev/week5/day1/truecolorpixelvalue.png">

이 사진을 보면, 각각의 픽셀이 3개의 값을 가지고 있다. 코의 부분에서는 빨간색이 높게 나오고 있고, 오른쪽부분은 파란색이 높게 나오고 있다.

<br>

## 영상에서 사용되는 좌표계

<img src="/assets/img/dev/week5/day1/imageaxis.png">

영상에서 사용되는 좌표계는 좌측 상단을 0,0으로 기준을 잡는다. 그리고 0,0부터 시작하므로 크기가 (w,h)의 이미지, 즉 `w-by-h image`라 할 때, 마지막은 (w,h)가 아닌 (w-1,h-1)의 좌표를 가진다. 이를 2차원 행렬로 표현하기도 한다. M행, N열을 가진 M x N 행렬이 있다고 할 때, 이를 `m-by-n matrix`라 한다. 

<img src="/assets/img/dev/week5/day1/imagematrix.png">

여기서 주의해야 할 것은 영상에서는 w, 즉 가로를 먼저 작성하지만, 행렬로 표현할 때는 행, 즉 세로를 먼저 작성한다. 그래서 640x480 이미지라 하면 영상으로 표현되어 있는지 행렬로 표현되어 있는지를 알아야 한다. 이것이 틀릴 경우 출력이 이상하게 나오거나 오류가 날 것이다.

<br>

<br>

# 영상 데이터의 표현 방법

이미지는 대체로 2차원 행렬로 이루어져 있기에 2차원 배열을 생성하면 될 것이다.

## 정적 2차원 배열 생성

```cpp
unsigned char a[480][640] {};
```

- unsigned char: 1바이트 사용(0~255 사이의 정수 표현)
- 2차원 배열 전체 크기만큼의 메모리 공간이 연속적으로 할당된다. (640x480 = 307200 bytes)
- 단점:
    - 배열의 크기를 미리 알고 있어야 한다. -> 다양한 크기의 영상을 표현하기에 부적절하다.
    - stack 영역에 메모리를 할당하기에 대략 1MB까지만 할당이 가능하다.
    - => 그래서 방법은 잘 사용하지 않고, 동적 메모리 할당을 사용한다.

<img src="/assets/img/dev/week5/day1/2matrix.png">

<br>

## 동적 2차원 배열 생성

```cpp

int w = 640;
int h = 480;

unsigned char** p;
p = new unsigned char*[h];
for (int i = 0; i < h; i++) {
    p[i] = new unsigned char[w] {};
}
```

- 행 단위로만 연속된 메모리 공간이 보장된다.
- 프로그램 동작 중 다양한 크기의 영상을 생성할 수 있다.
- heap 영역에 메모리를 할당하므로서 x86의 경우 2GB까지 할당이 가능하고, x64의 경우 8TB까지 가능하다.

<br>

### 코드 자세히 분석

```
1. unsigned char** p;
2. p = new unsigned char*[h];
3. for (int i = 0; i < h; i++) {
4.     p[i] = new unsigned char[w] {};
}
```

1.unsigned char 2차원 포인터 p를 선언 -\> local 영역에 로컬 변수 형태로 포인터 변수가 생성된다.

<br>

2.new 연산자를 통해서 unsigned char 포인터 타입을 h개 만큼 할당 -\> h개만큼의 포인터 변수가 생성되고 그 위치를 p가 가리키도록 할당된다. 그렇게 되면 `p[0]~p[h-1]`까지 접근할 수 있게 된다.

<img src="/assets/img/dev/week5/day1/pointp.png">

<br>

3.w개만큼의 unsigned char 메모리 공간을 h만큼 할당하게 된다.

<img src="/assets/img/dev/week5/day1/pointph.png">

<br>

- 동적 2차원 배열 원소 접근 방법

할당을 한 후에는 2중 for문을 사용해서 각각의 픽셀값에 접근할 수 있다.

```cpp
// 2차원 배열 p의 모든 원소 값을 10씩 증가
for (int y=0;y < h; y++) { // y 좌표
    for (int x=0; x < w; x++) { // x 좌표
        p[y][x] = p[y][x] + 10;
    }
}
```

좌측 상단부터 오른쪽으로 간 후 다음 줄 \-\> 오른쪽 \-\> 다음줄 ... 순서로 된다.

<br>

## 동적 2차원 배열 메모리 해제

동적으로 할당한 2차원 배열은 사용이 끝난 후에는 반드시 해제를 해야한다. 이때 delete를 사용한다. 동적 2차원 배열 생성의 역순으로 해제해야 한다.

```cpp
for (int y = 0; y < h; y++) {
    delete[] p[i];
delete[] p;
}
```

이 때, delete에 `[]`를 반드시 같이 사용해야 한다. 그리고 2차원 각각의 행에 대한 메모리를 해제하고, p의 포인터에 대해 한 번 더 해제해야 한다.

<br>

## 대용량 1차원 메모리 할당 후 영상 데이터 저장

그러나 영상 데이터를 2차원이 아닌 1차원으로 메모리를 할당하여 저장할 수도 있다.

```cpp
int w = 10, h = 10;
unsigned char * data = new unsigned char[w * h] {};
...

delete[] data; // 메모리 해제
```

10x10 영상데이터를 저장한다고 했을 때, 10*10=100개의 unsigned char 메모리 공간을 할당하고 모든 픽셀 데이터를 저장한다. 순서는 위와 같이 좌측 상단부터 다음 줄 \-\> 오른쪽 \-\> 다음줄 ... 순서로 저장한다.

- 특정 좌표 (x,y) 위치 픽셀 값 참조

```cpp
unsigned char& p1 = *(data + y*w + x);
```

포인터 연산을 사용해서 w는 영상의 가로 크기, data는 시작 좌표를 의미한다.

<br>

## 간단한 형태의 영상 데이터 저장 클래스 생성

```cpp
class MyImage
{
public:
    MyImage() : w(0), h(0), data(0) {} // default, 즉 init

    MyImage(int _w, int _h) : w(_w), h(_h) { // w,h 두 개의 정수 값을 받는 생성자에서는 w,h를 초기화
        data = new unsigned char[w * h] {}; // new 연산자를 이용해서 데이터 공간을 메모리 할당, 그 시작 주소를 data가 가르키도록 했다.
    }

    ~MyImage() { // 소멸자
        if (data) delete[] data; // 메모리 할당이 있었으면 그것을 삭제하도록
    }

    unsigned char& at(int x, int y) { // at을 이용해서 (x,y) 좌표에 있는 픽셀값을 반환하도록 하는데, 참조(&)로 반환하도록 해서 읽어올 뿐만 아니라 픽셀값을 설정할 수도 있도록 만듦
        return *(data + y * w + x);
    }

public:
    int w, h; // w: 영상 가로 크기, h: 영상 세로 크기
    unsigned char* data; // 픽셀 데이터를 저장하기 위해 동적 할당한 메모리 공간의 시작 주소를 가리킬 포인터 변수
}
```

<br>

<br>

# 영상 파일의 형식과 특징

## BMP 파일 구조

비트맵(bitmap)
- 비트(bit)들의 집합(map) -\> 픽셀의 집합
- 영상의 전체 크기에 해당하는 픽셀 정보를 그대로 저장한다.
    - 장점: 표현이 직관적이고 분석이 용이하다.
    - 단점: 메모리 용량을 많이 차지한다. 영상의 확대/축소시 화질 손상이 심하다.
- 사진, 포토샵
- 비트맵의 종류
    - 장치 의존 비트맵(DDB): 출력 장치(화면, 프린터 등)의 설정에 따라 다르게 표현된다.
    - 장치 독립 비트맵(DIB): 출력 장치가 달라져도 항상 동일하게 출력된다. **BMP 파일은 Windows 환경에서 비트맵을 DIB형태로 저장한 파일 포맷이다.**

벡터 그래픽스(vector graphics)
- 점과 점을 연결해 수학적 원리로 그림을 그려 표현하는 방식
- 이미지 크기를 확대/축소해도 화질이 손상되지 않음
- 폰트, 일러스트레이터

<img src="/assets/img/dev/week5/day1/bitmapandvector.jpeg">

<br>

아래는 BMP 파일 구조를 나타낸 것이다.

<img src="/assets/img/dev/week5/day1/bitmapstructure.png">

1.비트맵 파일 헤더

이는 비트맵 파일에 대한 정보를 담고 있다.

파일 헤더의 구조체는 다음과 같다. WORD의 경우 2byte, DWORD는 4byte크기의 자료형이다. 그러므로 이는 전체 14 bytes로 표현되는 구조체다.

```cpp
typedef struct tagBITMAPFILEHEADER {
    WORD    bfType;     // 이 파일이 bmp파일인지 아닌지를 나타내는 지시자, 'B', 'M', 0x42/0x4D 와 같이 16진수로 기록이 된다.
    DWORD   bfsize;     // BMP 파일 크기를 4byte크기로 저장된다
    WORD    bfReserved1;// 현재 사용되지 않는 플래그
    WORD    bfReserved2;// 현재 사용되지 않는 플래그
    DWORD   bf0ffBits;  // 비트맵 비트까지의 오프셋으로 비트맵 파일 헤더로부터 픽셀 데이터가 있는 위치까지의 거리를 나타낸다.
} BITMAPFILEHEADER;
```

2.비트맵 정보 헤더

여기서는 비트맵 영상에 대한 정보를 담고 있다.

```cpp
typedef struct tagBITMAPINFOHEADER {
    DWORD   biSize;             // BITMAPINFOHEADER 크기
    LONG    biWidth;            // 비트맵 가로 크기, 4bytes
    LONG    biHeight;           // 비트맵 세로 크기
    WORD    bitPlanes;          // 항상 1을 저장
    WORD    bitBitCount;        // 한 픽셀의 컬러를 표현하기 위해 사용되는 비트 수를 나타낸다. truecolor의 경우 24, grayscale의 경우 8을 가진다.
    DWORD   biCompression;      // BI_RGB, 대부분의 경우 0
    DWORD   biSizeImage;        // 대부분의 경우 0
    LONG    biXPelsPerMeter;    // 대부분의 경우 0
    LONG    biYPelsPerMeter;    // 대부분의 경우 0
    DWORD   biClrUsed;          // 대부분의 경우 0
    DWORD   biClrImportant;     // 대부분의 경우 0
} BITMAPINFOHEADER;
```

3.색상 테이블/팔레트

여기서는 비트맵에서 사용되는 색상 정보를 담고 있다.

RGBQUAD는 4byte로 구성되어 있다. 

```cpp
typedef struct tagRGBQUAD {
    BYTE    rgbBlue;        // Blue
    BYTE    rgbGreen;       // Green
    BYTE    rgbRed;         // Red
    BYTE    rgbReserved;    // 사용되지 않는 공간인 1, 이를 사용하는 이유는 4의 배수로 맞춰주기 위함이다. 이는 메모리를 조금 더 빠르게 사용할 수 있기 때문에 사용한다.
} RGBQUAD;
```

이는 256컬러 이하의 비트맵에서만 존재한다. 그래서 grayscale 비트맵에서는 존재하고, truecolor 비트맵에서는 존재하지 않는다.

- 그레일스케일 비트맵
    - (0,0,0,0),(1,1,1,0)...,(255,255,255,0) 의 총 256개의 색상을 가진다.
    - 전체 4 * 256 = 1024bytes의 색상 테이블을 가지고 있게 된다.
- 트루컬러 비트맵
    - 이 색상 테이블을 가지지 않고 마지막 픽셀 데이터에 색상 정보를 저장한다.

4.픽셀 데이터

- 그레이스케일 비트맵: RGBQUAD 배열에 이미 정의되어 있기 때문에 배열의 인덱스 값을 저장
- 트루컬러 비트맵: 색상 테이블이 없기 때문에 (B,G,R) 순서로 픽셀 값을 저장

일반적으로 상하가 뒤집힌 상태로 저장된다. (**bottom-up**)

또한, 효율적인 데이터 관리를 위해 영상의 가로 크기를 4의 배수로 맞춰서 저장한다.

그레이스케일의 경우

4bytes x 3bytes

| | | | |
| --- | --- | --- | --- |
| idx | idx | idx | 0 |
| idx | idx | idx | 0 |
| idx | idx | idx | 0 |

트루컬러의 경우

12bytes x 3bytes

| | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| B | G | R | B | G | R | B | G | R | 0 | 0 | 0 |
| B | G | R | B | G | R | B | G | R | 0 | 0 | 0 |
| B | G | R | B | G | R | B | G | R | 0 | 0 | 0 |

<br>

<br>

<img src="/assets/img/dev/week5/day1/gray_4x4.png" width="400px">

이러한 그레이스케일 파일이 있다고 하자. 이를 바이너리 편집기로 열면 아래와 같이 출력된다.

<img src="/assets/img/dev/week5/day1/graybmp.png" width="50%">
<img src="/assets/img/dev/week5/day1/graydesc.png" width="50%">

맨 앞 줄의 숫자는 빼고, 그 다음부터 확인하면 된다. 오른쪽이 숫자들을 분석한 사진이다.

- bfType: bmp파일이므로 42, 4D
- bfSize: bmp파일의 크기
- biSize: 28 00 00 00, 16진수이므로 32+8이므로 bitmapinfoheader의 크기는 40byte
- biWidth, biHeight: 가로,세로가 4x4
- biBitCount: 1픽셀을 표현하기 위해 8bit 즉, 2^8=256가지의 색상을 나타낼 수 있다.
- 그레이스케일이므로 biClrImportant에서 infoheader가 끝나고 rgbquad를 4x256 = 1024byte만큼의 색상 테이블이 나타나게 된다. 0부터 FF(255)까지 나타나있다.
- 파란색: 실제 픽셀값에 대한 정보를 역순으로 나타내게 됨. 즉 FF FF FF FF는 맨 아래, 흰색을 나타낸다. 근데 중요한 것은 FF가 스케일값 자체를 나타내는 것이 아니라 **색상 테이블 중에서 255번째에 있는 색상을 가지고 있다**는 것을 나타낸다.

<br>

<img src="/assets/img/dev/week5/day1/color_4x4.png" width="400px">

<img src="/assets/img/dev/week5/day1/colordesc.png">

- bfType: bmp파일이므로 42, 4D
- biSize: 32+8=40byte
- biWidth,biHeight: 4x4
- biBitCount: 1=16 + 8 == 24bit, 즉 1픽셀당 3bytes
- 트루컬러이므로 색상 테이블이 없다.
- [80(B) 80(G) 80(R)] 는 16진수이므로 [128, 128, 128] 즉 회색
- FF 00 00: Blue 가 4개
- 00 FF 00: Green 가 4개
- 00 00 FF: Red 가 4개

<br>

<br>

이 bmp파일을 구체화시켜서 보고자 한다.

1. visual studio 에서 새 프로젝트 생성 -> window 데스크톱 애플리케이션
2. 이름은 각자 작성하고, `솔루션 및 프로젝트~` 를 체크한다.

생성하게 되면 기본적인 소스코드를 만들어준다. 여기서 간단하게 코드를 살펴보면

- mwinmain : 클래스를 등록
- initinstance: createwindow라는 함수를 이용해서 실제 윈도우를 만들고, showwindow를 사용해서 화면에 보여준다. 
- myregisterclass: 기본적인 윈도우 스타일에 대한 정보를 기록하고 있다. 이를 registerclassexw라는 것으로 보낸다. 이는 운영체제에 이러한 윈도우를 만들 것이라는 것을 등록한다.
- winproc: 윈도우 메시지를 처리하는 함수다.
    - wm_command: 어떤 메뉴를 선택했을 때 처리하는 케이스
    - wm_paint: 화면에 어떤 그림을 그릴 때 사용하는 케이스
    - wm_destroy: 프로그램이 종료될 때 실행되는 케이스
- about: 도움말 대화상자

솔루션 빌드를 하고, 디버깅하지 않고 시작을 누르면 창이 하나 생성되는 것을 볼 수 있다.

<br>

그래서 이 WM_COMMAND가 끝나는 부분에 왼쪽 마우스가 눌렸을 때에 대한 메시지를 처리하도록 하고자 한다. 

```cpp
# include <stdio.h>     // 파일에 대한 처리를 위해 맨 위에 # include <stdio.h> 를 추가

case WM_LBUTTONDOWN:
{
    FILE* fp = NULL;
    fopen_s(&fp, "cat.bmp", "rb"); // 현재 폴더에 있는 cat.bmp 열기

    if (!fp)
        break;

    BITMAPFILEHEADER bmfh; // fileheader에 대한 정보를 담을 구조체 선언
    BITMAPINFOHEADER bmih; // infoheader에 대한 정보를 담을 구조체 선언

    fread(&bmfh, sizeof(BITMAPFILEHEADER), 1, fp); // 불러온 파일에서 bitmapfileheader 크기만큼을 불러와서 bmfh에 저장
    fread(&bmih, sizeof(BITMAPINFOHEADER), 1, fp); // 불러온 파일에서 bitmapfileheader 크기만큼을 불러와서 bifh에 저장

    LONG nWidth = bmih.biWidth;         // bitmapinfoheader중에서 가로 추출
    LONG nHeight = bmih.biHeight;       // bitmapinfoheader중에서 세로 추출
    WORD nBitCount = bmih.biBitCount;   // bitmapinfoheader중에서 각각의 픽셀이 몇 비트로 표현되고 있는가에 대한 정보 추출

    DWORD dwWidthStep = (DWORD)((nWidth * nBitCount / 8 + 3) & ~3); // 하나의 행을 표현하기 위해 필요한 메모리 크기를 계산하는 것, 가로 크기 * 각 byte 크기를 한 후, `+ 3) & ~3)은 이 값보다 같거나 큰 4의 배수값을 구하는 방법이다.
    DWORD dwSizeImage = nHeight * dwWidthStep; // 영상의 세로크기와 위의 값을 곱하면 전체 픽셀 데이터를 저장하기 위해 필요한 메모리 공간의 크기를 구할 수 있다.

    DWORD dwDibSize;
    if (nBitCount == 24)    // 트루 컬러의 경우
        dwDibSize = sizeof(BITMAPINFOHEADER) + dwSizeImage; // 전체 Dibsize라고 해서 infoheader와 전체 영상 데이터의 크기를 더한다.
    else    // 그레이스케일의 경우
        dwDibSize = sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * (1 << nBitCount) + dwSizeImage; // 색상 테이블의 크기까지 추가해서 전체 Dibsize를 구할 수 있다.

    BYTE* pDib = new BYTE[dwDibSize]; // 메모리 공간을 동적 할당

    // fseek 함수를 이용해서 파일의 맨 처음에서 비트맵 파일 헤더 크기만큼 이동하므로 비트맵 정보 헤더 위치로부터 DIB크기(dwDibSize)만큼을 파일로부터 읽는다.
    fseek(fp, sizeof(BITMAPFILEHEADER), SEEK_SET); 
    
    // pDib 메모리에는 비트맵 정보 헤더와 색상테이블(팔레트), 픽셀 데이터 정보가 저장된다.
    fread(pDib, sizeof(BYTE), dwDibSize, fp);

    LPVOID lpvBits;
    if (nBitCount == 24) // 두 가지를 따로 계산하는 이유는 이 두개의 포인터 값을 이용해서 setDIBitsToDevice라는 win32함수를 사용해서 bitmap을 나타낼 때 사용되기 때문이다.
        lpvBits = pDib + sizeof(BITMAPINFOHEADER); // 실제 픽셀 데이터가 나타나는 주소값을 계산
    else
        lpvBits = pDib + sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * (1 << nBitCount);

    HDC hdc = GetDC(hWnd);
    int x = LOWORD(lParam); // winRroc에 있는 인자값에 있는 것으로 실제 마우스가 클릭된 x좌표를 파싱하는 코드
    int y = HIWORD(lParam); // winRroc에 있는 인자값에 있는 것으로 실제 마우스가 클릭된 y좌표를 파싱하는 코드
    ::SetDIBitsToDevice(hdc, x, y, nWidth, nHeight, 0, 0, 0, nHeight, lpvBits,
        (BITMAPINFO*)pDib, DIB_RGB_COLORS); // 출력하는데 필요한 정보들을 전달
    ReleaseDC(hWnd, hdc);

    delete[] pDib;
    fclose(fp);
}
break;
```

<br>

## BMP/JPG/GIF/PNG 파일 형식의 특징

1. BMP
- 픽셀 데이터를 압축하지 않고 그대로 저장 -\> 파일 용량이 큰 편
- 파일 구조가 단순해서 별도의 라이브러리 도움 없이 파일 입출력 프로그래밍이 가능
- 파일의 크기가 중요하지 않은 연산의 경우 이를 사용한다.

2. JPG/JPEG
- 주로 사진과 같은 컬러 영상을 저장
- 손실 압축(lossy compression)
- 압축률이 좋아서 파일 용량이 크게 감소 -\> 디지털 카메라 사진 포맷으로 주로 사용됨
- 정밀한 영상 처리나 컴퓨터 비전에서는 픽셀 값이 조금만 바뀌어도 성능이 차이가 나서 선호는 하지 않는다. 

3. GIF
- 256 색상 이하의 영상을 저장 -\> 일반 사진을 저장 시 화질 열화가 심하다.
- 무손실 압축(lossless compression)
- 움직히는 GIF 지원
- 영상 처리에서는 잘 사용하지 않는다.

4. PNG
- Portable Network Graphics
- 무손실 압축 (컬러 영상도 무손실 압축)
- 알파 채널(투명도/불투명도)를 지원한다.
- 파일의 크기가 중요하지 않을 때 사용한다.

<br>

## 영상 데이터 크기 분석

- 그레이스케일 영상: (가로 크기) x (세로 크기) bytes
    - (512x512) 의 경우 512x512=26211bytes
- 트루컬러 영상: (가로 크기) x (세로 크기) x 3 bytes
    - FHD(1820x1080) 의 경우 1920x1080x3 = 6220800bytes
    - 이를 30fps로 1분 재생하려면 6MB x 30fps = 180MB x 60sec = 1GB

| | <img src="/assets/img/dev/week5/day1/lenna_gray.png" width="33%"> | <img src="/assets/img/dev/week5/day1/sky.png" width="33%"> | <img src="/assets/img/dev/week5/day1/tree.png" width="33%"> |
| --- | --- | --- | --- |
| 속성 | 521x512 Grayscale | 1920x1080 Truecolor | 1920x1080 Truecolor |
| BMP | 263,222 | 6,220,854 | 6,220,854 |
| PNG | 167,488 | 2,730,645 | 4,081,084 |
| JPG(95%) | 90.965 | 512,220 | 1,098,200 |
| JPG(80%) | 37,923 | 213,879 | 542,790 |

bytes 변환하는 방법으로는 cv2::imwrite()함수를 사용하면 된다. 나는 그냥 python으로 구해보았다.

```python
from pathlib import Path

name = 'lenna_gray'
filenames = ['./' + name+'.bmp', './'+ name+'.png','./'+name+'_95p.jpg','./'+name+'_80p.jpg']

for f in filenames:
    file_size =Path(f).stat().st_size
    print(f.split('/')[-1]," size is:", file_size,"bytes")
```

```markdown
lenna_gray.bmp  size is: 263222 bytes
lenna_gray.png  size is: 167488 bytes
lenna_gray_95p.jpg  size is: 90965 bytes
lenna_gray_80p.jpg  size is: 37923 bytes
```

lenna_gray.bmp 파일의 크기는 비트맵 크기 `14` + infoheader크기 `40` + 팔레트 크기 `1024` + 가로x세로 `512x512` = `263,222` 로 구할 수 있다.

sky.bmp나 tree.bmp의 경우는 14 + 40 + 1920x1080 = 6,220,854bytes 가 된다.

jpg의 경우 압축률을 정할 수 있다. 그래서 위의 경우 95%, 80%로 지정했다. 80%가 더 큰 압축률을 의미한다.

sky와 tree의 크기가 다른 이유는 픽셀값의 변화가 다소 작은 경우 압축이 더 많이 되기 때문에 sky가 크기가 더 작다. 이처럼 **픽셀값의 변화가 작은 경우 저주파 성분이 강하다**고 하고, **변화가 큰 경우 고주파 성분이 강하다**고 할 수 있다.

<br>

<br>

