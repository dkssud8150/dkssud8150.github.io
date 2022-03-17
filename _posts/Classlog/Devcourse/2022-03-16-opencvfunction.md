---
title:    "[데브코스] 5주차 - OpenCV Function with Mat class and useful feature "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-16 13:01:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, event]
toc: True
comments: True
image:
  src: /assets/img/dev/week5/day3/main.jpg
  width: 800
  height: 500
---

<br>

# 유용한 OpenCV 함수

## 행렬 합/평균/최대,최소

### 행렬 합 구하기

```cpp
Scalar sum(InputArray src);
```

- src : 입력 행렬, 1~4채널
- 반환값 : 행렬 원소들의 합

- **예제 코드**

```cpp
uchar data[] = {1, 2, 3, 4, 5, 6};
Mat mat1(2,3,CV_8UC1, data);
/* 1 2 3; 4 5 6 */

int sum1 = (int)sum(mat1)[0]; // 21 == val[0]
```

<br>

### 행렬 평균

```cpp
Scalar mean(InputArray src, InputArray mask = noArray());
```

- src : 입력 행렬, 1~4채널
- mask : 마스크 영상, noArray()는 mat 클래스를 비어있는 상태와 같음, 마스크라는 것은 전체 행렬에서 원소의 값이 0이 아닌 것만 수행하도록 하는 것, 즉 mask에서 0과 1이 섞여 있을 때 1인 곳의 src 위치인 것만 계산한다는 것, 그렇다면 src와 mask는 크기가 같아야 한다.
- 반환값 : 행렬의 평균 값

- **예제 코드**

```cpp
Mat img = imread("lenna.bmp", IMREAD_GRAYSCALE);

double mean = mean(img)[0]; // 124.0 그레일 스케일이므로 scalar에서 val[0]만 0이 아니고, 1,2,3은 0으로 채워져 있을 것이다. 그래서 val[0]만 불러옴. 만약 IMREAD_COLOR이라면 
// Scalar m = mean(img);
// m[0] = Blue, m[1] = Green, m[2] = Red
// 에 대한 평균
```

<br>

### 행렬 최댓값/최솟값

```cpp
void minMaxLoc(InputArray src, double* minVal, double* maxVal = 0, Point* minLoc = 0, Point* maxLoc = 0, InputArray mask = noArray());
```

- src : 입력 영상, 단일 채널만 가능
- minVal, maxVal : 최솟값/최댓값 변수 포인터(필요없으면 NULL 지정)
- minLoc, maxLoc : 최솟값/최댓값 위치 변수 포인터(필요없으면 NULL 지정)
- mask : 마스크 영상, mask 행렬 값이 0이 아닌 부분에서만 연산을 수행

- **예제 코드**

```cpp
Mat img = imread("lenna.bmp", IMREAD_GRAYSCALE);

double minv, maxv;
Point minLoc, maxLoc;
minMaxLoc(img, &minv, &maxv, &minLoc, &maxLoc); // 25, 245, [508,71], [116,273]
```

여기서 아래와 같이 작성할 경우 

```cpp 
double* minv;
mixMaxLoc(img, *minv, ...);
```

`*`의 피연산자가 포인터여야 하는데 double형식으로 되어 있다고 나온다. 또한, 인수 목록이 일치하는 오버로드된 함수 minMaxLoc의 인스턴스가 없다고 나온다.

<br>

## 행렬의 자료형(타입) 변환

```cpp
void Mat::convertTo(OutputArray m, int rtype, double alpha=1, double beta=0) const;
```

> ::란 Mat의 전역함수가 아닌 멤버함수라는 것, 그래서 'Mat.convertTo()' 형태로 작성해야 한다는 것이다.

- m : 출력 영상(행렬)
- rtype : 원하는 출력 행렬 타입, CV_8UC1, CV_32FC1 등등
- alpha : 추가적으로 곱할 값
- beta : 추가적으로 더할 값

- **예제 코드**

```cpp
Mat img = imread("lenna.bmp", IMREAD_GRAYSCALE); // CV_8UC1

Mat fimg;
img.convertTo(fimg, CV_32FC1); // img를 CV_32FC1형태로 바꾸어 fimg로 저장한다.
```

<br>

## 행렬의 정규화

```cpp
void normalize(InputArray src, InputOutputArray dst, double alpha = 1, double beta = 0, int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray());
```

- src : 입력 행렬(영상)
- dst : 출력 행렬, src와 같은 크기
- alpha : 노름 정규화인 경우 목표 노름(norm)값, norm_minmax인 경우에는 최솟값
- beta : norm_minmax인 경우 최댓값
- norm_type : 정규화 타입, NORM_INF, NORM_L1, NORM_L2, **NORM_MINMAX** 중 하나를 지정
- dtype : 출력 행렬의 타입
- mask : 마스크 영상

- **예제 코드**

```cpp
Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

Mat dst;
normalize(src, dst, 0, 255, NORM_MIXMAX); // 모든 픽셀값을 0~255로 픽셀값을 정규화시킨다는 것, 위에서 봤듯이 최솟값이 25, 최댓값이 245였기에 이를 조금 더 0~255로 정규화시킴
```

<br>

## 색 공간 변환 함수

```cpp
void cvtColor(InputArray src, OutputArray dst, int code, int dstCn = 0);
```

- src : 입력 영상
- dst : 출력 영상
- code : 색 변환 코드 COLOR_BGR2GRAY, COLOR_BGR2RGB, COLOR_BGR2HSV, COLOR_BGR2YCrCb 등등, [opencv 문서 페이지](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab) 참고
- dstCn : 결과 영상의 채널 수, 0이면 자동 결정됨

- **예제 코드**

```cpp
Mat src = imread("lenna.bmp"); // 3channel

Mat dst;
cvtColor(src, dst, COLOR_BGR2GRAY); // 3 -> 1channel
```

<br>

## 채널 분리/병합

### 채널 분리

```cpp
void splic(const Mat& src, Mat* mvbegin); // 참조긴하나 그냥 Mat 타입 변수를 지정하면 될듯
void split(InputArray src, OutputArrayOfArrays mv);
```

- src : (입력) 다채널 행렬
- mvbegin : (출력) Mat 배열의 주소
- mv : (출력) 행렬의 벡터, vector<Mat>

- **예제 코드**

```cpp
Mat src = imread("lenna.bmp");

vector<Mat> planes;
split(src, planes);
```

1개의 다채널 행렬을 여러 개의 단일 채널로 분리해주는 것이다.

<br>

### 채널 병합

```cpp
void merge(const Mat* mv, size_t const, OutputArray dst);
void merge(InputArrayOfArrays mv, OutputArray dst);
```

- mv : (입력) 1채널 Mat 배열 또는 행렬의 벡터
- count : (mv가 Mat타입의 배열인 경우) Mat 배열의 크기
- dst : (출력) 다채널 행렬

합치려는 영상을 배열로 받을 수도 있고, 벡터 Mat타입을 받을 수도 있다. 

- **예제 코드**

```cpp
Mat src = imread("lenna.bmp");

vector<Mat> planes;
split(src, planes);

swap(planes[0], planes[2]); //planes[0](blue)과 planes[2](red)를 바꿔주는 것

Mat dst;
merge(planes, dst); // blue와 red가 바뀐 상태로 출력될 것이다.
```

<img src="/assets/img/dev/week5/day3/merge.png">

<br>

<br>

# 유용한 OpenCV 함수

## 연산 시간 측정 방법

### 연산 시간 측정

대부분의 영상 처리 시스템은 대용량 영상 데이터를 다루고 복잡한 알고리즘 연산을 수행한다. 영상 처리 시스템 각 단계에서 **소요되는 연산 시간을 측정**하고 시간이 오래 걸리는 부분을 찾아 개선하는 **시스템 최적화 작업**이 필수적이다. 그러므로 OpenCV에서도 연산 시간 측정을 위한 함수를 지원한다.

- getTickCount, getTickFrequency

```cpp
int64 t1 = getTickCount();

my func();

int64 t2 = getTickCount();
double ms = (t2 - t1) * 1000 / getTickFrequency();
```

- getTickCount() : tick이라는 발생 횟수를 세는 것, tick은 초의 단위가 아니기 때문에 frequency를 사용해서 구해야 한다.
- getTickFrequency() : 이를 그대로 나눠주면 상당히 작게 나와서 1000을 곱해서 ms로 구한다.

> 연산 시간 측정은 릴리즈 모드에서 수행해야 한다.

<br>

- TickMeter 클래스

개념적으로 복잡하고, 사용이 번거로워서 새로운 클래스를 사용하면 편하다. 이는 직관적인 인터페이스를 제공하고, getTickCount와 getTickFrequency를 조합해서 시간을 측정한다.

TickMeter의 정의는 다음과 같다.

```cpp
class TickMeter
{
    public:
        TickMeter();    // 기본 생성자

        void start();   // 시간 측정을 시작할 때 사용
        void stop();    // 시간 측정을 멈출 때 사용
        void reset();   // 시간 측정을 초기화할 때 사용

        double getTimeMicro() const;    // 연산 시간을 마이크로 초 단위로 반환
        double getTimeMilli() const;    // 연산 시간을 밀리초 단위로 반환
        double getTimeSec() const;      // 연산 시간을 초 단위로 반환
        ...
}
```

- 예제 코드

```cpp
TickMeter tm;
tm.start():

func1();

tm.stop();
cout << "func1(): " << tm.getTimeMilli() << "ms. " << endl;

tm.reset(); // reset을 하지 않으면 앞의 시간과 합쳐진다.

tm.start();

func2();

tm.stop();
cout << "func2(): " << tm.getTimeMilli() << "ms. " << endl;
```

<br>

## ROI 영역과 마스크 연산

### ROI

ROI : region of interest
- 영상에서 특정 연산을 수행하고자 하는 임의의 부분 영역

### 마스크 연산

opencv는 일부 함수에 대해 ROI 연산을 지원하며, 이 때 마스크 영상(mask image)를 인자로 함께 전달해야 한다. 마스크 영상으 CV_8UC1타입(grayscale), 마스크 영상의 픽셀 값이 0이 아닌 위치에서만 연산을 수행한다. 보통 마스크 영상으로는 0 또는 255로 구성된 이진 영상(binary image)를 사용한다.

<br>

```cpp
void Mat::CopyTo(InputArray m, InputArray mask) const;
void copyTo(InputArray src, OutputArray dst, InputArray mask);
```

멤버 함수
- m : 출력 영상, 만약 `*this`(자기 자신)와 크기 및 타입이 같은 m을 입력으로 지정하면 **m을 새로 생성하지 않고 연산을 수행**하고, 그렇지 않으면 m을 새로 생성하여 연산을 수행한 후 반환한다.
- mask : 마스크 영상, CV_8U.0이 아닌 픽셀에 대해서만 복사 연산을 수행한다.

전역 함수
- src : 입력 영상
- mask : 마스크 영상
- dst : 출력 영상

<br>

전체 동작은 src는 원본 데이터, dst는 배경 데이터, mask는 물체에 대한 마스크, 즉 물체 이외의 부분은 0으로 되어 있다. 이들을 합치면 영상 합성이 된다.

```cpp
Mat src = imread("airplane.bmp", IMREAD_COLOR);
Mat mask = imread("mask_plane.bmp", IMREAD_GRAYSCALE); // 회색이나 실제로는 컬러이므로 grayscale로 불러와야 한다.
Mat dst = imread("field.bmp", IMREAD_COLOR);

//copyTo(src, dst, mask);
src.copyTo(dst, mask); // src의 mask즉 0이 아닌 부분들만 dst에 복사해서 합성

imshow("src", src);
imshow("dst", dst);
imshow("mask", mask);
waitKey();
```

<img src="/assets/img/dev/week5/day3/combine.png">

이 src와 dst는 같은 크기와 같은 타입이어야 한다. 그리고 위의 코드는 mask 이미지가 있는 경우에 가능하다. 만약 mask가 없다면 png 파일에 알파 채널이 있는지 확인하기 위해 GIMP 프로그램을 통해 열어본다.

알파채널이 있다면, [B,G,R,α]에서 앞에 [B,G,R]은 일반적인 컬러 형태의 Mat 클래스 객체로, α채널은 그레이스케일의 Mat 클래스 객체로 만들면 된다. 그리고 중요한 것은 배경 이미지와 물체 이미지의 크기가 다르다면 자신이 물체를 넣을 위치를 잘라서 크기를 맞춰주어야 한다.

<br>

```cpp
Mat src = imread("cat.bmp", IMREAD_COLOR);
Mat logo = imread("opencv-logo-white.png", IMREAD_UNCHANGED); // unchanged로 하면 알파가 없어진다.
```

<img src="/assets/img/dev/week5/day3/pngimread.png">

```cpp
vector<Mat> planes;
split(logo, planes);

Mat mask = planes[3]; // 알파 채널은 mask로 지정
```

<img src="/assets/img/dev/week5/day3/pngimread.png">
<img src="/assets/img/dev/week5/day3/mask.png">

```cpp
merge(vector<Mat>(planes.begin(), planes.begin() + 3), logo); // 첫번째 인자는 vectorMat을 새로 만드는데, planes 맨 앞부터 3개만 사용해서 vector<Mat> 객체를 새로 만들어서 logo로 출력
```
<img src="/assets/img/dev/week5/day3/split.png">

```cpp
Mat crop = src(Rect(10, 10, logo.cols, logo.rows));
```

<img src="/assets/img/dev/week5/day3/crop.png">

<br>

```cpp
logo.copyTo(crop, mask); // cv 함수로 mask연산이 옵션으로 되어 있다. 즉, 0이 아닌 값들에 대해서만 logo를 copy하여 crop에 추가한다.

imshow("src", src);
waitKey();
destroyAllWindows();
```

<img src="/assets/img/dev/week5/day3/final.png">