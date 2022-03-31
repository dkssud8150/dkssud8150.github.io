---
title:    "[데브코스] 6주차 - OpenCV Hough transform and using GPU "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-25 13:01:00 +0800
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

# 허프 변환(Hough transform) 직선 검출

2차원 영상 좌표에서의 직선의 방정식을 파라미터 공간으로 변환하여 직선을 찾는 알고리즘이다. 

`y = ax + b =\> b = -xa + y`

<img src="/assets/img/dev/week6/day5/hough.png">

(x,y) 평면에서 한 직선을 표현하는 a',b'가 있고, 이 직선을 지나는 점을 (xi,yi), (xj,yj) 라 한다면 직선의 방정식은 다음과 같다.

- b = -xi*a + yi
- b = -xj*a + yj

이 두 직선은 점 (a',b')를 지난다.

<br>

## 축적 배열

직선 성분과 관련된 원소 값을 1씩 증가시키는 배열

<img src="/assets/img/dev/week6/day5/accumulation.gif">

<br>

직선의 방정식 y= ax+b를 사용할 때의 문제점은 y축과 평행한 수직선은 표현하지 못한다. 그래서 극좌표계 직선의 방정식을 사용한다.

`x*cosϴ + y*sinϴ = ρ`

<img src="/assets/img/dev/week6/day5/houghtransform.png">

극 좌표를 사용할 경우 원래는 xy평면에서의 점은 ab평면에서는 직선, xy평면에서의 직선은 ab평면에서는 점으로 구성되지만, 극좌표계를 사용하면 ϴρ평면에서 곡선으로 나오게 된다. 

직선을 찾기 위해서는 ab평면이든 ϴρ평면이든 그리드 형태로 나눠야 한다. 이 때, 나누는 단위를 정해야 하는데, 예를 들어 ϴ를 0.1씩 나누어 그리드를 그린다면 직선을 찾는 성능은 좋아질 것이지만, 시간이 오래 걸릴 수 있다.

<br>

- **허프 변환 직선 검출 함수**

```cpp
void HoughLines(InputArray image, OutputArray lines, double rho, double theta, int threshold, double srn = 0, double stn = 0, double min_theta = 0, double max_theta = CV_PI);
```

- image : 그레이스케일의 에지 영상
- lines : 직선의 파라미터(rho,theta) 저장할 출력 벡터 vector\<Vec2f>
- rho : 축적 배열에서 rho 값의 간격 e.g. 1.0 -> 1픽셀 간격
- theta : 축적 배열에서 theta 값의 간격 e.g. CV_PI/180 -> 1도 간격
- threshold : 축적 배열에서 직선으로 판단할 임계값
- stn, srn : 멀티스케일 허프 변환에서 rho 해상도를 나누는 값, srn에 양의 실수를 지정하면, rho 해상도와 rho/srn 해상도를 각각 이용하여 멀티스케일 허프 변환을 수행한다. srn과 stn이 모두 0이면 일반 허프 변환을 수행
- min_theta : 검출할 직선의 최소 theta 값
- max_theta : 검출할 직선의 최대 theta 값

<br>

- **코드 구현**

```cpp
Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

Mat src_edge;
Canny(src, src_edge, 50, 150);

vector<Vec2f> lines1;
HoughLines(src_edge, lines1, 1, CV_PI / 180, 250); // 1픽셀, 1도 간격

Mat dst1;
cvtColor(src_edge, dst1, COLOR_GRAY2BGR); // 검출 직선의 색을 입히기 위해 변환

// 반한된 직선의 방정식을 화면에 출력
for (size_t i = 0; i < lines1.size(); i++) {
    float r = lines1[i][0], t = lines1[i][1];
    double cos_t = cos(t), sin_t = sin(t); // cos, sin 은 float타입으로반환되기 때문에 정밀한 측정을 위해 double로 반환한 후 계산
    double x0 = r*cos_t, y0 = r*sin_t;
    double alpha = 1000;

    Point pt1(cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t));
    Point pt2(cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t));
    line(dst1, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
}
```

houghlines를 사용하면 반환된 값을 화면에 출력하는 것이 조금 복잡하다. 그래서 이를 보완하여 나온 함수가 있다.

- **확률적 허프 변환에 의한 선분 검출 함수**

```cpp
void HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength = 0, double maxLineGap = 0)
```

- image : 그레이스케일 에지 영상
- lines : 선분의 시작, 끝 좌표(x1,y1,x2,y2)를 저장할 출력 벡터 vector\<Vec4i>
- rho : 축적 배열에서 rho값의 해상도, 픽셀 단위 1.0 -\> 1픽셀 간격
- theta : 축적 배열에서 theta값의 간격, 라디안 단위 CV_PI/180 -\> 1도 간격
- threshold : 축적 배열에서 직선으로 판단할 임계값
- minLineLength : 검출할 선분의 최소 길이, 특정 길이보다 짧은 것은 검출하지 않을 길이를 정하는 것
- maxLineGap : 직선으로 간주할 최대 에지 점 간격, 직선이 끊어져서 있을 경우 어느 정도의 거리보다 작을 때는 항상 이어주라는 간격을 정하는 것

P는 proportion, 확률에 대한 함수로, 이 함수를 사용하여 반환된 값은 선분, 즉 시작점과 끝점을 받는다. 

<br>

- **코드 구현**

```cpp
vector<Vec4i> lines2;
HoughLinesP(src_edge, lines2, 1, CV_PI / 180, 160, 50, 5);

for (size_t i = 0; i < lines2.size(); i++) {
    Vec4i l = lines2[i];
    line(dst2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
}
```

line들을 화면에 출력하는데 있어서 위의 houghlines 함수보다 훨씬 더 간단해진 것을 볼 수 있다. 들어 있는 것을 바로 그려주면 된다.

<img src="/assets\img\dev\week6\day5\houghoutput.png">

나뭇잎의 경우 울퉁불퉁한 에지도 직선으로 판단되고 있는 것을 볼 수 있다.

<br>

```cpp
TickMeter tm;
tm.start();

vector<Vec2f> lines1;
HoughLines(src_edge, lines1, 1, CV_PI / 180, 250);

tm.stop();
tm.reset();
tm.start();

vector<Vec4i> lines2;
HoughLinesP(src_edge, lines2, 1, CV_PI / 180, 160, 50, 5);

tm.stop();
```

시간 차는 조금 나지만, for문을 돌려 식을 구성하고, 계산하는데도 시간이 걸리기 때문에 비슷하다.

<img src="/assets\img\dev\week6\day5\timeset.png">

<br>

이를 차선 인식에 대입해서 사용해보면 다음과 같이 코드를 작성할 수 있다.

```cpp
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("../data/lane03.bmp");

	imshow("src", src);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	TickMeter tm;
	tm.start();

//	Rect roi = selectROI(src); 
//	Mat roi_crop = src(roi);
//	imshow("crop",roi_crop);

//	int roi_crop_mean = mean(roi_crop)[0];

	vector<Point> pts;
	pts.push_back(Point(70, 350));
	pts.push_back(Point(300, 200));
	pts.push_back(Point(390, 200));
	pts.push_back(Point(640, 350));


	Mat mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
//	mask = roi_crop_mean;

	fillPoly(mask, pts, Scalar(255), LINE_AA);

	Mat crop;
	copyTo(src, crop, mask);

	cvtColor(crop, crop, COLOR_BGR2GRAY);

	Mat crop_edge;
	GaussianBlur(crop, crop, Size(), 1);
#if 0
	crop = crop > 150;
#else
	threshold(crop, crop, 170, 255, THRESH_BINARY);
#endif

	Canny(crop, crop_edge, 50, 150);
	imshow("crop_edge",crop_edge);

	vector<Vec4i> lines;
	HoughLinesP(crop_edge, lines, 1, CV_PI / 180, 30, 20, 100);

//	for (auto line : lines) 
//		cout << line << endl;

	tm.stop();
	cout << tm.getTimeMilli() << "ms." << endl;

	for (Vec4i line : lines) {
		float dx = float(line[2] - line[0]);
		float dy = float(line[3] - line[1]);
		float angle = float(atan2f(dy, dx) * 180 / CV_PI); // x,y 변위를 통해 삼각함수를 사용하여 직선의 각도를 구함
		
		if (fabs(angle) <= 10)
			continue;	// 하늘과 나무 등 차선이 아닌 것들에 대한 직선도 많이 그려지기 때문에 이를 삭제하기 위한 코드

		if (angle > 0)
			cv::line(src, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 0, 255), 1); // 각도가 +이면 빨강
		else
			cv::line(src, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 255, 0), 1); // 각도가 -이면 초록
	}

	imshow("src", src);
//	imshow("dst", dst);

	waitKey();
	return 0;
}

```

<img src="/assets\img\dev\week6\day5\linedetect.png">

이 차선을 동영상에서 잘 인식하기 위해서는 과거의 데이터, 즉 1~2초 전에 라인을 인식한 데이터를 가져와서 찾는 것이 잘 찾아진다. 왜냐하면 현재의 사진에 대해서만 검출한다면 점선인 차선이 너무 띄엄띄엄 있을 경우 잘 인식을 못하기도 하고, 노이즈가 많이 생기기도 한다. 

<img src="/assets\img\dev\week6\day5\linedetect2.png">

또는관심 영역을 설정해서 그 안에서만 검출하는 방법도 있다. 아니면 차선이 거의 흰색이므로 그런 색상 정보를 이용할 수 있다. 그림자의 경우에도 에지로 찾아질 수 있다. 그러므로 도로의 픽셀값보다 커지는(차선이 흰색) 에지에 대해서만 검출을 시키면 될 것이다.

관심영역을 설정하는데는 두가지가 있다.
1. selectROI 함수 사용
    - 사각형만 가능하여 제한적이지만, 간단하고 쉽다.
    <img src="/assets\img\dev\week6\day5\selectROI.png">
    <img src="/assets\img\dev\week6\day5\selectedROI.png"> 

2. **polylines 그린 후 mask 연산하여 관심 영역만 copyTo(src, mask)**
    - 1번보다 구현하기는 복잡하나 원하는 영역 모양으로 구성할 수 있다. 딥러닝을 사용하려면 이 방법을 사용해야 할 것이다.

```cpp
#include <iostream>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("../data/lane03.bmp");

	imshow("src", src);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	TickMeter tm;
	tm.start();

//	Rect roi = selectROI(src); 
//	Mat roi_crop = src(roi);
//	imshow("crop",roi_crop);

//	int roi_crop_mean = mean(roi_crop)[0];

	/* lane03.bmp point */
	vector<Point> pts;
	pts.push_back(Point(70, 350));
	pts.push_back(Point(300, 200));
	pts.push_back(Point(390, 200));
	pts.push_back(Point(640, 350));


	Mat mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
//	mask = roi_crop_mean;

	fillPoly(mask, pts, Scalar(255), LINE_AA);

	Mat crop;
	copyTo(src, crop, mask);

	cvtColor(crop, crop, COLOR_BGR2GRAY);

	Mat crop_edge;
	GaussianBlur(crop, crop, Size(), 1);
#if 0
	crop = crop > 150;
	Canny(crop, crop_edge, 50, 150);
#else
	threshold(crop, crop, 170, 255, THRESH_BINARY);
	Canny(crop, crop_edge, 50, 150);
#endif
	

	vector<Vec4i> lines;
	HoughLinesP(crop_edge, lines, 1, CV_PI / 180, 30, 20, 100);

//	for (auto line : lines) 
//		cout << line << endl;

	tm.stop();
	cout << tm.getTimeMilli() << "ms." << endl;

	for (Vec4i line : lines) {
		float dx = float(line[2] - line[0]);
		float dy = float(line[3] - line[1]);
		float angle = float(atan2f(dy, dx) * 180 / CV_PI);
		
		if (fabs(angle) <= 10)
			continue;

		if (angle > 0)
			cv::line(src, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 0, 255), 1);
		else
			cv::line(src, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 255, 0), 1);
	}

	imshow("src", src);
//	imshow("dst", dst);

	waitKey();
	return 0;
}

```

<img src="/assets\img\dev\week6\day5\polylineline.png">

이 때, roi를 설정해서 직선을 그리게 되면, 배경과 mask사이에도 직선이 그려진다. mask의 배경의 픽셀값으 0이고, 관심 영역안의 픽셀과 차이가 나기 때문이다. 따라서 픽셀 값의 차에 대한 이진화를 추가해야 테두리에 대한 직선을 제거시킬 수 있다.

<br>

<br>

- threshold

threshold를 사용한다는 것은 이진화를 하는 것을 얘기한다. 대체로 특정 값보다 큰 것을 true, 특정 값보다 작은 것을 false로 해서 배경과 물체를 나눈다. 영상마다 threshold를 다르게 적용해야 하는데, 이를 자동으로 조절이 되도록 하는 **오츠 알고리즘**라는 것이 있다.
 
<img src="/assets\img\dev\week6\day5\otsu.gif">
 
입력 영상의 히스토그램에 대해 threshold값을 0부터 255까지 증가시키면서 빨간 곡선이 어떤 형태로 되는지 확인한다.그 후 빨간 곡선이 최대가 되는 값에서 이진화가 가장 잘된다고 판단한다.

이를 OpenCV에 적용하는 방법은 간단하다. threshold를 실행하면 된다.

```cpp
Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

Mat dst1,dst2;
// otsu 이진화 적용
double thres = threshold(src, dst1, 130, 255, THRESH_BINARY);
double thresotsu = threshold(src, dst2, -1,255, THRESH_OTSU);


// 계산된 Threshold 출력
cout << thres << "\n" << thresotsu << endl;

imshow("src", src);
imshow("dst1", dst1);
imshow("dst2", dst2);
waitKey();
}
```

<img src="/assets/img/dev/week6/day5/otsuimage.png">
<img src="/assets/img/dev/week6/day5/otsuthreshold.png">

dst1는 130 threshold를 적용한 것이고, dst2는 otsu알고리즘을 사용한 threshold를 적용한 영상이다. lenna영상에서는 117이 나왔다. threshold를 넘으면 흰색, 넘지못하면 검정색이 될 것이다.픽셀값이 낮은 어두운 부분들은 다 검정색, 픽셀값이 높은 부분들은 흰색이 된다.

이 때, src의 밝기를 낮추게 되면 출력 영상도 많이 달라진다.

```cpp
Mat src3 = src - 100;
Mat dst3;
double thresotsu2 = threshold(src3, dst3, -1, 255, THRESH_OTSU);
imshow("otsuthresholddark", dst3);
```

<img src="/assets/img/dev/week6/day5/darkotsu.png">
<img src="/assets/img/dev/week6/day5/darkotsuthreshold.png">

threshold를 출력해봐도 낮게 나오는 것을 확인할 수 있다.

<br>

[참고 블로그1](https://www.charlezz.com/?p=45285)
[참고 블로그2](https://minimin2.tistory.com/129)

<br>

차선이 검정색인데, 환경이 어두워진다면 에지검출이 잘 되지 않는다. 명암비를 높이거나, canny함수의 최소,최대를 바꾸는 방법을 사용해야 할 것이다. equalize를 수행하면 작은 픽셀값의 차이가 강조가 된다. 대신 노이즈가 증가할수도 있다.

<br>

차선에 적용해보았는데, 마스크를 이진화해서 그런지 차선이 사라졌다. 

```cpp
threshold(crop, crop_thr, -1, 255, THRESH_OTSU);
Canny(crop_thr, crop_edge, 50, 150);
```

<img src="/assets\img\dev\week6\day5/cropthreshold.png">

그래서 원본에 적용을 했더니 차선은 인식이 된다. ROI를 하지 않는 코드에서는 잘 사용될 것 같다.

```cpp
threshold(src, crop_thr, -1, 255, THRESH_OTSU);
Canny(crop_thr, crop_edge, 50, 150);
```

<img src="/assets\img\dev\week6\day5/srcthreshold.png">

<br>

otsu 대신 threshold에 존재하는 다양한 방식을 적용하여 차선인식을 해보았다.

```cpp
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap("../data/test_video.mp4");

	if (!cap.isOpened()) {
		cerr << "video open failed!" << endl;
		return -1;
	}

	int w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));

	Mat mask = Mat::zeros(h, w, CV_8UC1);

	vector<Point> pts;
	pts.push_back(Point(400, 340));
	pts.push_back(Point(680, 340));
	pts.push_back(Point(1200, 720));
	pts.push_back(Point(0, 720));
	pts.push_back(Point(0, 480));

	fillPoly(mask, pts, Scalar(255));

	Mat frame, dst_mask, dst, roi;
	while (true) {
		cap >> frame;

		if (frame.empty())
			break;

		copyTo(frame, roi, mask);

		Mat roi_gray, roi_edge;
		cvtColor(roi, roi_gray, COLOR_BGR2GRAY);

		GaussianBlur(roi_gray, roi_gray, Size(), 1.0);

#if 0
		roi_gray = roi_gray > 100;
#else
		/*  추가로 otsu 알고리즘 및 다양한 threshold 방식을 적용하여 threshold를 적용하여 이진화해보았습니다.*/
		threshold(roi_gray, roi_gray, 100, 255, THRESH_BINARY);
//		threshold(roi_gray, roi_gray, 100, 255, THRESH_BINARY_INV); // 반전된 에지 영상이 출력되지만, 선은 에지에 대한 검출이므로 잘 적용됨
//		threshold(roi_gray, roi_gray, 80, 255, THRESH_TRUNC); // 점선은 인식이 잘 되지 않음
//		threshold(roi_gray, roi_gray, -1, 255, THRESH_OTSU); // -1을 적용함으로서 경계를 지정하지 않음, 차선인식이 잘 되지 않음

		/* adaptivethreshold라는 것도 있어 참고해서 추가해보았지만, mask를 통한 경계선도 같이 인식되어 threshold가 잘 되지 않았습니다. */
//		adaptiveThreshold(roi_gray, roi_gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 5);
//		adaptiveThreshold(roi_gray, roi_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 3);

#endif
		Canny(roi_gray, roi_edge, 50, 150);

		vector<Vec4i> lines;
		HoughLinesP(roi_edge, lines, 1, CV_PI/180, 100, 20, 100);

		for (Vec4i line : lines) {
			float dx = float(line[2] - line[0]);
			float dy = float(line[3] - line[1]);
			float angle = float(atan2f(dy, dx) * 180 / CV_PI);

			if (fabs(angle) <= 10)
				continue;

//			if (line[2] <= 400 || line[2] >= 1190 && line[0] >= 680)
//				continue;

			cv::line(frame, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 0, 255), 2);
		}

		imshow("dst", frame);
		imshow("roi_edge", roi_edge);

		if (waitKey(10) == 27)
			break;
	}
}
```

- THRESH_BINARY

<img src="/assets\img\dev\week6\day5\thresh_binary.png">

<br>

- THRESH_BINARY_INV

<img src="/assets\img\dev\week6\day5\thresh_binary_inv.png">

<br>

- THRESH_TRUNC

<img src="/assets\img\dev\week6\day5\thresh_trunc.png">

<br>

- THRESH_OTSU

<img src="/assets\img\dev\week6\day5\thresh_otsu.png">

<br>

- ADAPTIVE_THRESH_MEAN_C

<img src="/assets\img\dev\week6\day5\adaptive_thresh_mean.png">

<br>

- ADAPTIVE_THRESH_GAUSSIAN_C

<img src="/assets\img\dev\week6\day5\adaptive_thresh_gaussian.png">

<br>

[참고 사이트](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
[참고 사이트](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)

<br>

<br>

# 허프 변환 원 검출

허프 변환을 응용하여 원을 검출할 수 있다. 이는 잘 사용하지 않는 방법이기도 하고, 필요할 때 코드만 가져와서 사용하면 될 것 같다.

`원의 방정식 : (x-a)^2 + (y-b)^2 = c^2 -\> 3차원 축적 평면`

원래는 y=ax+b이므로 2차원 평면에 표시했었다. 그래서 이를 3차원을 사용할 수도 있지 않을까 하는 생각에서 나왔다. 

하지만 OpenCV에서는 원 검출 대신 속도 향상을 위해 **Hough gradient method**를 사용한다.
- 입력 영상과 동일한 2차원 평면 공간에서 축적 영상을 생성한다.
- 에지 픽셀에서 그래디언트 계산
- 그래디언트 방향에 따라 직선을 그리면서 해당 픽셀에 값을 누적한다.
- 여기서는 단순하게 평면을 변환하는 것이 아니라 그래디언트를 구해서 그 직선에 따라 1씩 더하는 것이다.
- 높은 값을 가진 픽셀을 원 중심으로 잡고 에지와 가장 많이 겹치는 적절한 반지름을 검출한다.
- 단점 : 여러 개의 동심원은 검출하지 못한다. 가장 작은 원 하나만 검출한 후 끝난다.

```cpp
void HoughCircles(InputArray image, OutputArray circles, int method, double dp, double minDist, double params1 = 100, double params2 = 100, int minRadius = 0, int maxRadius = 0)
```

- image : 입력 영상, 에지가 아닌 일반 영상
- circles : (cx,cy,r)의 정보를 담을 Mat(CV_32FC3) 도는 vector\<Vec3f>
- method : HOUGH_GRADIENT or HOUGH_GRADIENT_ALT 지정, 이 두개를 사용할 때 각각 param1,param2의 값이 달라지게 되어 주의해야 한다.
    - HOUGH_GRADIENT : 원래 사용하던 버전으로 동심원을 찾지 못했음
    - HOUGH_GRADIENT_ALT : 최신 버전으로 좀 더 잘 검출하고, 동심원도 찾을 수 있게 됨
- dp : 입력 영상과 축적 배열의 크기 비율, 1이면 동일 크기, 2이면 축적 배열의 가로, 세로 크기가 입력 영상의 1/2
- minDist : 검출된 원 중심점들의 최소 거리
- params1 : canny 에지 검출기의 높은 임계값, 이 함수안에서 canny에지 검출을 해주기 때문에 지정하는 값
    - HOUGH_GRADIENT : sobel 을 사용한다. 약 (50,150) 이므로 최대 값을 지정해주기 위해 150
    - HOUGH_GRADIENT_ALT : scharr 을 사용하므로 조금 큰 값을 사용해야 한다. 약 300 정도의 값을 줘야 한다.
- params2 : 축적 배열 임계값, HOUGH_GRADIENT_ALT는 원의 perfectness 값(1에 가까운 실수를 사용)
    - HOUGH_GRADIENT_ALT : 0.8~1.0 값으로 지정
- minRadius, maxRadius : 검출할 원의 최소/최대 반지름

이를 통해 동전 이미지에 대한 검출이 가능하다. 

```cpp
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("coins.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat dst1, dst2;
	cvtColor(src, dst1, COLOR_GRAY2BGR);
	cvtColor(src, dst2, COLOR_GRAY2BGR);

	// HOUGH_GRADIENT
	Mat blr;
	GaussianBlur(src, blr, Size(), 1.0); // 입력 영상을 블러링하여 잡음을 제거해야 한다.

	vector<Vec3f> circles1;
	HoughCircles(blr, circles1, HOUGH_GRADIENT, 1, 10, 150, 30, 10, 50);

	for (size_t i = 0; i < circles1.size(); i++) {
		Vec3i c = circles1[i];
		circle(dst1, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 2, LINE_AA);
	}

	// HOUGH_GRADIENT_ALT
	vector<Vec3f> circles2;
	HoughCircles(src, circles2, HOUGH_GRADIENT_ALT, 1.5, 10, 300, 0.9, 10, 50);

	for (size_t i = 0; i < circles2.size(); i++) {
		Vec3i c = circles2[i];
		circle(dst2, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
}
```

<img src="/assets\img\dev\week6\day5/houghcircle.png">

<br>

<br>

# GPU 활용

## CUDA

CUDA란 GPU에서 수행하는 병렬 처리 알고리즘을 C 프로그래밍 언어를 비롯한 산업 표준 언어를 사용하여 작성할 수 있도록 하는 GPGPU(General Purpose computin on GPU) 기술이다. CUDA는 엔비디아가 개발해오고 있어서 엔비디아 그래픽 카드에서만 가동이 된다.

Main메모리에서 GPU메모리로 데이터를 복사하고, CPU에서 CUDA로 명령을 내려서 실행하고 결과가 GPU메모리에 들어가고, 그것을 다시 Main메모리로 복사한다.

<br>

c언어, c++에서는 gpu를 사용하는 방법으로는 `__x__`처럼 CUDA문법인 `__`를 추가해주면 된다.

```cpp
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main() {
    int a = 2, b=7,c;
    int *d_a,*d_b,*d_c;

    cudaMalloc((void**)&d_a, sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    add<<1,1>>(d_a,d_b,d_c); // cuda 문법 <<>> 코어를 어떻게 쓸지

    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```

openCV에서 CUDA를 사용하려면 2가지를 해야 한다.
1. CUDA Toolkit 설치 : https://developer.nvidia.com/cuda-downloads
2. OpenCV 직접 빌드 : http://youtu.be/Gfl6EylhFvM
    - opencv_contrib 소스 코드 다운로드 및 CMake에서 OPENCV_EXTA_MODULES_PATH 설정
    - CMake 에서 WITH_CUDA 설정 선택
    - CUDA_TOOLKIT_ROOT_DIR 확인

- **CUDA를 이용한 Hough 선분 검출 예제 코드**

자세한 설명이 잘 없고, `opencv/sources/samples/gpu/*`여기에 있는 파일들이 opencv에서 제공하는 gpu를 사용한 샘플 파일들이다.

```cpp
#include "opencv2/cudaimgproc.hpp" // cuda관련 헤더 파일을 모듈별로 포함

int main(void) {
    Mat src = imread("building.jpg",IMREAD_GRAYSCALE);
    Mat edge;
    cv::Canny(src,edge,100,200,3);

    // cuda::GpuMat 클래스 사용
    cuda::GpuMat d_edge(edge);
    cuda::GpuMat d_lines;

    // cuda용으로 구현된 함수 및 클래스 사용
    Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (CV_PI/180.0f), 50, 5);
    hough -> detect(d_edge, d_lines);
}
```

cpu와 gpu의 시간 차이는 대체로 15~20배 정도 차이난다. 그러나 cpu와 gpu의 결과가 다르게 나오기도 한다. 그래서 이에 대한 처리도 추가해서 해줘야 한다.

<br>

<br>

# OpenCL(Open Computing Language)

OpenCV에서 CUDA를 사용하려면 빌드를 해야 하는데, 좀 복잡하고 오래 걸린다. 그래서 그 대신 OpenCL을 사용하기도 한다.

여러 개의 CPU, GPU, DSP 등의 프로세서로 이러우저니 이종 플랫폼에서 동작하는 프로그램 코드 작성을 위한 개방형 범용 병렬 컴퓨팅 프레임워크
- Open,royalty-free,cross-platform,parallel programming, heterogeneous

애플에서 최초 개발해서 Intel, AMD, ARM,nVidia 등이 참여했다.

<br>

GPU-Z 프로그램에서 OpenCL 항목을 확인할 수 있다. 

- **openCL을 사용한 코드**

예전에는 `ocl`이라는 네임스페이스를 통한 **ocl클래스**를 사용했어야 했다. 그러나 3.x 버전으로 발전되면서 Mat 함수를 생성할 때 `UMat`으로만 바꿔주면 openCL을 사용할 수 있게 되었다. 이것만 바꿔줘도 나머지는 gpu를 사용하는 방식으로 작동하게 된다.

```cpp
VideoCapture cap("../data/test_video.mp4");
CascadeClassifier fd("haar.xml");

UMat frame, gray,blr,dst; // Mat frame, gray,blr,dst;
vector faces;

while (true){
    cap >> frame; 

    TickMeter tm;
    tm.start();

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blr,Size(), 2);
    bilateralFilter(blr,dst,-1,10,5);

    tm.stop();
    Cout << tm.getTimeMilli() << endl;

    imshow("dst",dst);
    imshow("frame",frame);

    if(waitKey(10) == 27)
        break;
}
```

<br>

`MAT -> UMAT` 로만 바꿔줌으로서 최소한의 소스 코드 변경을 통해 gpu를 사용하여 HW 가속이 가능하다. `cap >> frame;` 이 UMat에서 사용이 가능하게 되어 있다. 선언으로 이동해보면 UMat에 대한 동작이 되도록 선언되어 있다. 처음에는 Mat으로 받아서 알아서 UMat으로 변환해준다. 또한 `cvtColor`도 확인해보면 UMat인지에 대한 판단이 들어가 있다. 

그러나 `imshow`를 하거나, `cap>>frame`과 같이 frame을 받아오는 과정에서 gpu형태에서 cpu형태로 바꿔지기 때문에 이 것들에 대한 시간을 고려해야 한다. 그래서 UMat 사용이 항상 좋은 것은 아니나, 영상 관련해서는 분명히 빨라진다고 한다.

- gpu 사용

<img src="/assets/img/dev/week6/day5/gpu.png">

<br>

- cpu 사용

<img src="/assets/img/dev/week6/day5/cpu.png">


- Mat to UMat

```cpp
Mat mat = imread("lenna.bmp")

UMat umat1;

mat.copyTo(umat1);
UMat umat2 = mat.getUMat(ACCESS_READ); // transform to UMat
```

ACCESS_READ는 opencv에 설명이 부실하다. ACCESS_WRITE와 두개가 있는데 대체로 ACCESS_READ를 하면 잘 작동한다.

<br>

- UMat to Mat

```cpp
UMat umat;
videoCap >> umat;

Mat mat1;
umat.copyTo(mat1);

Mat mat2 = umat.getMat(ACCESS_READ);
```

UMat이 연산 속도가 빠르긴 하나 변환하는데도 속도가 조금 걸리기 때문에, 영상을 표현하는 Mat에 대해서만 UMat을 사용하는 것이 좋다. border 처리 연산시에도 `BORDER_REPLICATE`옵션을 사용하는 것이 조금 더 좋다고 한다. 

getUMat, getMat을 사용할 때 주의해야 할 것이 있다. getUMat 함수를 통해 UMat객체를 생성할 경우, 원본과 UMat, 이 두 개는 얕은 복사가 된 것이므로, 나중에 원본을 사용하기 전에 새로 생성한 UMat 객체가 완전히 소멸한 후 원본 Mat 객체를 사용해야 한다.

```cpp
cv::Mat mat1(height, width, CV_32FC1);
mat1.setTo(0);
{
	cv::UMat umat1 = mat1.getMat(cv::ACCESS_READ);
}
```



<br>

<br>

> 딥러닝 학습은 파이토치나 텐서플로우로 하고, 결과를 저장한 후 그 파일을 opencv에서 불러와 사용할 수 있다. 파이토치의 경우 onnx로 변환해서 불러오는 방법을 사용해야 한다고 한다.

<br>

> 실제로 차선 검출에서 hough변환을 사용하지는 않는다. 속도나 정확도가 뛰어나지 않기 때문이다. 그러나 컴퓨터비전을 처음 배우는 사람들에게는 적용해보기 쉬운 알고리즘이라서 필수적으로 배우는 내용이다.

- cv::LineSegmentDetector

[https://docs.opencv.org/4.5.5/db/d73/classcv_1_1LineSegmentDetector.html](https://docs.opencv.org/4.5.5/db/d73/classcv_1_1LineSegmentDetector.html)

<img src="/assets/img/dev/week6/day5/lsd.png">

<br>

cv 클래스에 라인을 따주는 클래스가 있다. 그러나 허프변환과 속도가 비슷하기도 하고, 정확도가 더 뛰어나다고 할 수도 없다. 어떤 경우든 튜닝을 잘 해야 한다.

