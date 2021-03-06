---
title:    "[데브코스] 6주차 - OpenCV specific color and edge extraction "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-24 13:01:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, OpenCV]
toc: True
comments: True
# image:
#   src: /assets/img/dev/week6/day3/main.jpg
#   width: 500
#   height: 500
---

<br>

# 특정 색상 영역 추출

RGB, HSV, YCrCb 등의 색 공간에서 각 색상 성분의 범위를 지정하여 특정 색상 성분만 추출할 수 있다. RGB는 계산이 이상하게 나올 수 있으므로 HSV나 YCrCb를 사용한다.

```cpp
void inRange(InputArray src,InputArray lowerb, InputArray upperb, OutputArray dst);
```

- src : 입력 행렬
- lowerb : 하한 값 (Mat or Scalar)
- upperb : 상한 값 (Mat or Scalar)
- dst : 입력 영상과 동일 크기, CV_8UC1 타입, 범위 안에 들어가는 픽셀 값만 255로 설정
    - 단일 채널 -\> dst =  lowerb <= src <= upperb
    - 다중 채널 -\> dst1 = lowerb3 <= src1 <= upperb1
                    dst2 = lowerb2 <= src2 <= upperb2
                    dst3 = lowerb1 <= src3 <= upperb3


노란색만 추출하고 싶을 때, HSV 기준으로 5~30 정도 또는 15~40 정도의 값을 추출하면 된다.

```cpp
int pos_hue1 = 5, pos_hue2 = 30, pos_sat1 = 200, pos_sat2 = 255;
Mat src, src_hsv, dst, dst_mask;

if (argc < 2) {
    src = imread("flower1.png", IMREAD_COLOR);
} else {
    src = imread(argv[1], IMREAD_COLOR);
}

cvtColor(src, src_hsv, COLOR_BGR2HSV);

Scalar lowerb(pos_hue1, pos_sat1, 0);
Scalar upperb(pos_hue2, pos_sat2, 255);
inRange(src_hsv, lowerb, upperb, dst_mask);

cvtColor(src, dst, COLOR_BGR2GRAY); // gray로 변환하여 연산하고자 함
cvtColor(dst, dst, COLOR_GRAY2BGR); // 나중에 출력하는 것에 색을 입히고 싶으므로 bgr로 변형해놓는다.
src.copyTo(dst, dst_mask); // 마스크에 대한 0이 아닌 값만 복사

imshow("dst_mask", dst_mask);
imshow("dst", dst);
imshow("src", src);
waitKey();
```

<br>

<br>

# 히스토그램 역투영 (histogram backprojection)

앞서 사용했던 방법에서 원색을 찾는 것이라면 HSV가 편하지만, Hue값이 애매하게 지정해야만 한다면 조금 어려울 수 있다. 그래서 이 색과 비슷한 것들만 추출해달라는 방법을 사용하면 편하다. **히스토그램 역투영**은 주어진 히스토그램 모델(영상에서 특정 사각형을 쳐서 그 안의 픽셀들을 추출)에 영상의 픽셀들이 얼마나 일치하는지를 검사하는 방법을 말한다. 임의의 색상 영역을 검출할 때 효과적이다. 이 때, YCrCb로 변환시켜서 적용해야 잘 추출된다.

특정 사각형의 픽셀들을 2차원 히스토그램을 그린 후 그레이스케일 범위로 정규화를 한 후 그래프를 그리게 되면 작은 점으로 표시될 것이다. 이 때, 앞서 배운 inrange는 마스크 영상으로 0 or 255이지만, 이번에는 0~255사이의 값들이 나오게 되어 결과 영상을 만든다.

이 때 추출된 색상에 가우시안을 적용해서 추출하면 조금 더 부드러운 영상을 얻을 수 있다.

<br>

<img src="/assets\img\dev\week6\day4\selectroi.png">

- 2차원 히스토그램

```cpp
Mat hist;
int channels[] = {1, 2};
int cr_bins = 128; int cb_bins = 128;
int histSize[] = {cr_bins, cb_bins};
float cr_range[] = {0, 256};
float cb_range[] = {0, 256};
const float* ranges[] = {cr_range, cb_range};

// 부분 영상에 대한 히스토그램 계산
calcHist(&crop, 1, channels, Mat(), hist, 2, histSize, ranges);
```

컬러 영상의 히스토그램을 그려야 하기 때문에 조금 복잡하다. 먼저 ycrcb로 변환시킨 후 선택할 영역을 크롭한 후 그 공간에서 히스토그램을 하는데, y는 무시하고 crcb만 구할 것이므로 1,2만 출력하고, 히스토그램 사이즈를 정한 후, 범위를 0~255이지만 (0,256)으로 지정한 후 calcHist를 계산한다.

hist는 2차원으로 나올 것이다.

- 역투영

```cpp
Mat backproj;
calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);
imshow("back", backproj);
```

<img src="/assets\img\dev\week6\day4\backproject.png">

- 마스크 복사

```cpp
src.copyTo(dst, backproj);
```

copyto는 0이 아닌 픽셀들에 대해 복사하므로 다 수행된다.

- selectROI

```cpp
Mat src = imread("cropland.png", IMREAD_COLOR);
Rect rc = selectROI(src);

Mat src_ycrcb;
cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

Mat crop = src_ycrcb(rc);
```

selectROI라는 함수를 사용하면 window에서 사용자가 관심 공간을 지정해줄 수 있다. 지정된 공간을 잘라서 여기에 대해서만 관심을 두겠다는 것이다.

<br>

- **역투영 함수**

```cpp
calcBackProject(const Mat* images,int nimages,const int* channels,InputArray hist,OutputArray backproject, const float** ranges, double scale = 1, bool uniform = true)
```

- images : 여러 Mat을 넣어 줄 수 있다. 
- nimages : 몇 개의 Mat을 사용했는지에 대한 값
- 몇 채널만 사용했는지
- hist : calcHist에서 출력된 hist
- backproject : 출력 영상
- ranges : calcHist에서 사용했던 ranges

<br>

잘 찾기 위해서는 1장의 이미지만 사용하는 것이 아니라 살색을 찾기 위해서는 살색 영상을 집어넣어서 연산을 해야 한다. 잘 나온 히스토그램은 동그랗게 나올 것이다.

<img src="/assets/img/dev/histogram.png">

여기서 그냥 backprojection을 하면 배경에 있는 이미지들중에서도 비슷한 색상이 있다면 배경도 함께 추출될 수 있다. 그러므로 가우시안 블러를 한 후 이진화를 통해 0과 255로 변수를 재생성하면 좀 더 색상에 부합하는 것을 추출할 수 있다.


```cpp
Mat backproj;
calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);
GaussianBlur(backproj, backproj, Size(), 1.0);
backproj = backproj > 50;
```

또한, 히스토그램을 보기 위해 아래 코드도 실행하여 0~255로 정규화 한 후 윈도우로 출력해본다.

```cpp
calcHist(&ref_ycrcb, 1, channels, mask, hist, 2, histSize, ranges);

Mat hist_norm;
normalize(hist, hist_norm, 0, 255, NORM_MINMAX, -1);
imshow("hist_norm", hist_norm);
```

<br>

- **전체 코드**

```cpp
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	// Calculate CrCb histogram from a reference image

	Mat ref, ref_ycrcb, mask;
//	ref = imread("lenna.bmp", IMREAD_COLOR);
	ref = imread("ref.png", IMREAD_COLOR);
	mask = imread("mask.bmp", IMREAD_GRAYSCALE);
	cvtColor(ref, ref_ycrcb, COLOR_BGR2YCrCb);

	Mat hist;
	int channels[] = { 1, 2 };
	int cr_bins = 128; int cb_bins = 128;
	int histSize[] = { cr_bins, cb_bins };
	float cr_range[] = { 0, 256 };
	float cb_range[] = { 0, 256 };
	const float* ranges[] = { cr_range, cb_range };

	calcHist(&ref_ycrcb, 1, channels, mask, hist, 2, histSize, ranges);

#if 1
	Mat hist_norm;
	normalize(hist, hist_norm, 0, 255, NORM_MINMAX, -1);
	imshow("hist_norm", hist_norm);
#endif

	Mat src, src_ycrcb;
	src = imread("kids.png", IMREAD_COLOR);
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

	Mat backproj;
	calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);
	GaussianBlur(backproj, backproj, Size(), 1.0);
	backproj = backproj > 50;

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
	src.copyTo(dst, backproj);

	imshow("ref", ref);
	imshow("mask", mask);
	imshow("src", src);
	imshow("backproj", backproj);
	imshow("dst", dst);
	waitKey();

	return 0;
}
```

<br>

<br>

# 에지 검출과 영상의 미분

edge : 영상에서 픽셀의 밝기 값이 급격하게 변하는 부분을 말한다. 일반적으로 배경과 객체, 또는 객체와 객체의 경계가 이에 해당한다. 

기본적으로 에지를 검출하는 방법은 영상을 (x,y) 변수의 함수로 간주했을 때 이 함수의 1차 미분값이 크게 나타나는 부분(순간 변화량)을 검출한다. 그러나 픽셀 그래프틑 step function의 형태, 즉 계단 형식으로 정수값으로만 변화한다. 입력 영상에 가우시안 블러를 적용해서 잡음을 제거한 후 에지를 검출하는 것이 바람직하다.

미분값이 threshold(임계값)을 넘어가면 에지라고 할 수 있다.

<br>

## 1차 미분의 근사화

<img src="/assets/img/dev/week6/day4/approximation.png">

- 전진 차분 : [I(x+h) - I(x)] / h
- 후진 차분 : [I(x) - I(x - h)] / h
- 중앙 차분 : [I(x+h) - I(x-h)] / 2h

이 때, 중앙 차분이 가장 잘 근사화가 된다. 마스크를 간단화하기 위해 중앙 차분의 1/2를 빼서 계산한다. 

<br>

영상에서의 ∆x의 최소값은 1pixel이다. 그리고, 영상에서는 x방향, y방향 두 개를 계산해야 한다.

중요한 것은 잡음 제거와 마스크에서의 합이 0이 되면 영상이 전체적으로 어두워지므로 3x3 크기로 만든다. 즉 x방향의 [-1 0 1] 의 마스크가 있다고 하면 

```markdown
가로 방향
[-1 0 1]
[-1 0 1]
[-1 0 1]

세로 방향
[-1 -1 -1]
[0 0 0]
[1 0 1]
```

또는 가우시안 형태를 따서 만든 형태를 많이 사용한다.

```markdown
가로 방향
[-1 0 1]
[-2 0 2]
[-1 0 1]

세로 방향
[-1 -2 -1]
[0 0 0]
[1 2 1]
```

<br>

가로 방향에서의 아래 필터를 사용했을 때의 에지 출력이다.

<img src="/assets/img/dev/week6/day4/edge.png">

```cpp
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	Mat dst(src.rows, src.cols, CV_8UC1);

	for (int y = 1; y < src.rows-1; y++) {
		for (int x = 1; x < src.cols-1; x++) {
            /* 가로 방향 필터 */
			int v1 =  src.at<uchar>(y - 1, x + 1)
				+ src.at<uchar>(y, x + 1) * 2
				+ src.at<uchar>(y + 1, x + 1)
				- src.at<uchar>(y - 1, x - 1)
				- src.at<uchar>(y, x - 1) * 2
				- src.at<uchar>(y + 1, x - 1);

            /* 세로 방향 필터 */
            int v2 = src.at<uchar>(y - 1, x - 1)
				+ src.at<uchar>(y - 1, x) * 2
				+ src.at<uchar>(y - 1, x + 1)
                - src.at<uchar>(y + 1, x - 1)
				- src.at<uchar>(y + 1, x) * 2
				- src.at<uchar>(y + 1, x + 1);
            

            dst.at<uchar>(y, x) = saturate_cast<uchar>(v1);
            dst.at<uchar>(y, x) = saturate_cast<uchar>(v2);
		}
	}

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}
```

여기서 문제점은 미분값이 -가 나올 수도 있는데, 포화연산을 통해 아래부분은 다 날아가게 된다. 따라서 128을 더하여 확인을 해본다.

```cpp
//dst.at<uchar>(y, x) = saturate_cast<uchar>(v1);
dst.at<uchar>(y, x) = saturate_cast<uchar>(v1+128);
```

<img src="/assets/img/dev/week6/day4/edge128w.png">

이는 가로 방향 필터를 사용한 에지 검출이다. 여기서 흰색의 부분은 밝아지는 부분, 검은색의 부분은 어두워지는 부분이고, 회색은 아무것도 아닌 부분이다.

<br>

<img src="/assets/img/dev/week6/day4/edge128h.png">

위의 그림은 세로 방향 필터를 사용한 에지 검출이다. 동일하게 흰색 부분은 밝아지는 부분이고, 검은색 부분은 어두워지는 부분이다.

```cpp
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	Mat dst1(src.rows, src.cols, CV_8UC1);
	Mat dst2(src.rows, src.cols, CV_8UC1);

	for (int y = 1; y < src.rows-1; y++) {
		for (int x = 1; x < src.cols-1; x++) {
			/* 가로 방향 필터 */
			int v1 = src.at<uchar>(y - 1, x + 1)
				+ src.at<uchar>(y, x + 1) * 2
				+ src.at<uchar>(y + 1, x + 1)
				- src.at<uchar>(y - 1, x - 1)
				- src.at<uchar>(y, x - 1) * 2
				- src.at<uchar>(y + 1, x - 1);

			/* 세로 방향 필터 */
			int v2 = src.at<uchar>(y - 1, x - 1)
				+ src.at<uchar>(y - 1, x) * 2
				+ src.at<uchar>(y - 1, x + 1)
				- src.at<uchar>(y + 1, x - 1)
				- src.at<uchar>(y + 1, x) * 2
				- src.at<uchar>(y + 1, x + 1);


			dst1.at<uchar>(y, x) = saturate_cast<uchar>(v1+128);
			dst2.at<uchar>(y, x) = saturate_cast<uchar>(v2+128);
		}
	}

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
	waitKey();
}

```

<br>

에지 검출을 위해서는 이 둘을 함께 사용해야 정확한 에지를 구할 수 있다.



<br>

<br>

# 영상의 그래디언트

함수 f(x,y)를 x축과 y축으로 각각 편미분(partial derivative)하여 벡터 형태로 표현한 것이다.

∇f = [fx fy]

- 그래디언트 크기(magnitude) : |∇f| = sqrt(fx^2 + fy^2)
- 그래디언트 방향(phase) : ϴ = tan^(-1)(fy/fx)

이를 코드화 시키면 다음과 같다.

```cpp
int mag = (int)sqrt(v1 * v1 + v2 * v2);
dst.at<uchar>(y, x) = saturate_cast<uchar>(mag);
```

<br>

<img src="/assets\img\dev\week6\day4\edgegradient.png">

에지만 검출하기 위해서는 이진화를 작성해야 한다.

```cpp
dst = dst > 120;
```

이와 같이 threshold를 주게 되면 에지만 검출하고 나머지는 검은색으로 추출된다.

<img src="/assets\img\dev\week6\day4\edgethreshold.png">

<br>

## 그래디언트 크기와 방향

크기는 픽셀 값의 차이 정도, **변화량**이고, 방향은 픽셀 값이 가장 **급격하게 증가하는 방향**을 의미한다.

sobel연산자를 이용한 미분 함수가 있다.

- **함수**

```cpp
void Sobel(InputArray src, OutputArray dst, iont ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);
```

- src,dst : 입력, 출력 영상, 둘은 같은 크기 및 같은 채넗 수
- ddepth : 출력 영상 깊이
- dx, dy : x방향과 y방향으로의 미분 차수, 대부분 (0,1) 또는 (1,0)을 넣는다.
- ksize : 커널 크기
    - 1 : 3x1 or 1x3
    - CV_SCHARR : 3x3 scharr 커널
    - 3,5,7 : 3x3,5x5,7x7
- scale : option, 연산 결과에 추가적으로 곱할 값
- delta : option, 연산 결과에 추가적으로 더할 값
- borderType : 가장 자리 픽셀 확장 방식

ksize는 거의 필수로 3(default) 작성, CV_SCHARR은 scharr 필터를 사용하는 방법이다. (굳이는.. 안사용함)

```cpp
void magnitude(InputArray x, InputArray y, OutputArray magnitude);
```

- x : 2D 벡터의 x좌표 행렬, 실수형
- y : 2D 벡터의 y좌표 행렬, x와 같은 크기, 실수형
- magnitude : 2D 벡터의 크기 행렬, X와 같은 크기

<br>

<br>

- sovel마스크를 이용한 에지 검출

```cpp
Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

if (src.empty()) {
    cerr << "Image load failed!" << endl;
    return;
}

Mat dx, dy;
Sobel(src, dx, CV_32FC1, 1, 0);
Sobel(src, dy, CV_32FC1, 0, 1);

Mat mag;
magnitude(dx, dy, mag);
mag.convertTo(mag, CV_8UC1);

Mat edge = mag > 150;

imshow("src", src);
imshow("mag", mag);
imshow("edge", edge);
```

convertTo 를 사용하거나 normalize를 사용

<br>

<br>

## 캐니 에지 검출

### 캐니 에지 검출 알고리즘

openCV에 함수가 잘 되어 있다.

**좋은 에지 검출기란**
- 정확한 검출 : 에지가 아닌 점을 에지로 찾거나 또는 에지를 검출하지 못하는 확률을 최소화
- 정확한 위치 : 실제 에지의 중심을 검출
- 단일 에지 : 하나의 에지는 하나의 점으로 표현

이들을 수행하지 위한 canny edge detector을 사용한다.

<br>

#### 캐니 에지 검출 방법

1. 가우시안 필터링 : 잡음 제거(option)
2. 그래디언트 계산
    - 소벨 마스크를 사용하여 그래디언트의 크기와 방향을 계산
        - 방향을 4구역으로 단순화한다. 즉, 330~23,23~75...
3. NMS(non-maximum suppression)
    - 하나의 에지가 여러 개의 픽셀로 표현되는 현상을 없애기 위해 그래디언트 크기가 local maximum인 픽셀만을 에지 픽셀로 지정
    - 그래디언트 방향에 위치한 두 개의 픽셀과 local maximum을 검사
        - 그래디언트 방향 두개를 선정해서 물체의 수직 방향으로 체크하고자 함 
4. 이중 임계값을 이용한 히스테리시스 에지 트래킹
    - 조명에 의한 에지가 끊기는 부분을 해결하기 위함
    - 두개의 임계값을 사용 :Tlow, Thigh
    - 강한 에지 : ||f|| >= Thigh =\> 반드시 에지로 선정
    - 약한 에지 : Tlow <= ||f|| <= Thigh -\> 강한 에지와 연결된 픽셀만 최종 에지로 선정
    - Tlow >= ||f|| -\> 에지 아닌 것으로 판단

<br>

```cpp
void Canny(InputArray image, OutputArray edges, double threshold1, double threshold2, int aperturesize = 3, L2gradient = false)
```

- image : 입력 영상
- edges : 에지 영상
- threshold1 : 하단 임계값
- threshold2 : 상단 임계값, 1과 2의 비율을 1:2, 1:3을 권장한다.
- aperture size: 소벨 연산을 위한 커널 크기
- L2gradient : L2 norm 사용 여부

<br>

이처럼 실행하면 된다. 함수를 직접 구현하고 싶다면, opencv 소스코드를 참고하면 좋다.

> 소스코드 찾기
찾고자 하는 함수에 정지(F9) 후 디버깅(F5) -\> 프로시저 단위로 실행(F11) -\> outputarray/inputarray가 나오면 빠져나오기(shift+F11) -\> 함수를 찾을 수 없다고 나오면 `opencv/sources/modules/` -\> 이정도만 와도 알아서 인식한다. 안되면 -\> `imgproc/src/canny.cpp/`
>

<br>

<br>