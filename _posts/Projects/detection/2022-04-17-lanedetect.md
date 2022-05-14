---
title:    "[lane detection] sliding window를 c++로 구현하기 "
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-04-17 18:02:00 +0800
categories: [Projects, detection]
tags: [projects, lane detection]
toc: True
comments: True
math: true
mermaid: true
image:
  src: /assets/img/lanedetect/slidingwindow/slidingwindow.png
  width: 600
  height: 500
---


github : [https://github.com/dkssud8150/LaneDetectpjt](https://github.com/dkssud8150/LaneDetectpjt)

개발 언어는 c++이고, lane detection 기법 중 sliding window를 구현해보았다. 

<br>

# 영상 처리

## 이진화

1. 
  1. grayscale
    - 원본인 컬러 이미지에서 Gray 영상으로 변환
  2. Gaussian Blur
    - 노이즈를 처리
  3. canny edge
    - 외곽선 추출
    - lowwer threshold는 upper threshold의 2~3배가 적당

2. 
  1. Gaussian -> HSV image
  2. inRange를 통한 이진화 (Threshold)
    - 이 때, 명도에 대한 V만 사용하여 차선의 색이나 주변 밝기를 지정해줘야 한다.
    - e.g. cv2.inRange(hsv, (0,0,50), (255,255,255)) or cv2.inRange(hsv, (0,0,150), (255,255,255))
  3. canny edge

3. 
  1. Gaussian -> LAB image
  2. inRange를 통한 이진화 (Threshold)
  3. Canny edge

<br>

# 이진화 후 처리

1. ROI 영역 설정
  - 차선이 존재하는 위치에 지정해야 함, 또 필요없는 부분들을 잘 처리해야 한다.

2. houghlineP로 라인 추출
  - threshold, maxval 등의 파라미터를 잘 설정해야 함
  - 기울기의 절대값이 너무 작은 건 다른 물체들에 해당할 확률이 크므로 처리 x

3. 오른쪽, 왼쪽 분리

4. 라인들의 대표 직선 찾기
  - 선분의 기울기 평균값, 양끝점 좌표의 평균값을 사용하여 대표직선을 찾는다.
  - 노이즈를 제거해야 멀리 떨어져 있는 이상한 값도 함께 처리하지 않게 된다.
  - 모든 데이터를 반영하여 계산하는 것이 아닌 노이즈 데이터를 찾아 계산에서 제외시켜야 한다.

5. offset을 설정하여 차선 인식하고, 차선의 중간값과 화면의 중앙과 비교하여 핸들링

RANSAC 알고리즘을 통해 노이즈를 제거한다.

<br>

# 예외 상황 처리

- 카메라 영상에서 차선이 잡히지 않는 경우 영상처리를 위한 작업공간인 스크린 사이즈를 확대시킨다.
  - 실제 카메라 영상 크기보다 옆으로 넓어진 가상 스크린을 사용하여 작업
  - e.g. 기존 크기 : 640 x 480, 좌우로 200픽셀씩 확장하여 1040x480
- 한쪽 차선만 보이는 경우(차선이 끊기거나 추출되지 않을 경우)에는 추출한 한쪽 차선을 활용하여 대칭을 맞춰서 예측한다.
- 또는 차선은 연속적인 선이므로 지난번 위치를 재사용
- 새로 찾은 차선의 위치가 너무 많이 차이가 날 경우 갑자기 위치가 크게 바뀔 수 없기 때문에 한계값을 정해 이를 넘어갈 경우 무시하고 지난번 위치를 재사용

<br>

<br>

# 사용한 차선 인식 알고리즘 - sliding window

<img src="https://www.mdpi.com/sensors/sensors-19-03166/article_deploy/html/images/sensors-19-03166-g012.png" width="50%">

<br>

- 알고리즘 순서
1. ROI 설정
2. perspective transform
3. hsv -\> split and using only `V`
4. inverse
5. brightness processing
6. gaussian
7. inRange
8. histogram -\> argmax abount left and right -\> sliding

<br>

<br>

# 차선인식 알고리즘 종류

## Hough transform + **RANSAC**

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0045790620305085-gr1.jpg" width="50%">
<img src="https://user-images.githubusercontent.com/33013780/162755205-554cf4b9-cc64-40a8-b084-8854fbfb184b.png" width="40%">

<br>

<br>

---

## [Deep learnging](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMCl79-Jxmus3idtZDypeyTOc4ss5H96VjsQ&usqp=CAU) + RANSAC 

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMCl79-Jxmus3idtZDypeyTOc4ss5H96VjsQ&usqp=CAU">

- [instance segmentation](https://www.google.com/url?sa=i&url=https%3A%2F%2Fpaperswithcode.com%2Fpaper%2Ftowards-end-to-end-lane-detection-an-instance&psig=AOvVaw3dgcm4vjtKvEwXFY-1ojXB&ust=1649769511140000&source=images&cd=vfe&ved=0CAsQjhxqFwoTCJjaxPGRjPcCFQAAAAAdAAAAABAy)

<br>

<br>

---

## [V-ROI](https://github.com/Yeowoolee/OpenCV-Lane-Detection)

<img src="https://user-images.githubusercontent.com/33013780/162750624-38287654-3b98-4132-a8ed-d54cf0672087.png" width="300px"> 
<img src="https://user-images.githubusercontent.com/33013780/162751237-760413eb-4d25-44b7-8c8e-f6c69a116dac.png" width="300px">


- 참고자료 : https://yeowool0217.tistory.com/558?category=803755

<br>

<br>

# sliding window 코드

```cpp
#include <iostream>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <algorithm >

using namespace std;
using namespace cv;

vector<Point> matrix_oper(Mat frame, Mat per_mat_tosrc, int lx1, int ly1, int lx2, int ly2, int rx1, int ry1, int rx2, int ry2) {
	vector<Point> warp_left_line, warp_right_line;

	int new_lx1, new_ly1, new_lx2, new_ly2;
	new_lx1 = (per_mat_tosrc.at<double>(0, 0) * lx1 + per_mat_tosrc.at<double>(0, 1) * ly1 + per_mat_tosrc.at<double>(0, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * lx1 + per_mat_tosrc.at<double>(2, 1) * ly1 + per_mat_tosrc.at<double>(2, 2));

	new_ly1 = (per_mat_tosrc.at<double>(1, 0) * lx1 + per_mat_tosrc.at<double>(1, 1) * ly1 + per_mat_tosrc.at<double>(1, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * lx1 + per_mat_tosrc.at<double>(2, 1) * ly1 + per_mat_tosrc.at<double>(2, 2));

	new_lx2 = (per_mat_tosrc.at<double>(0, 0) * lx2 + per_mat_tosrc.at<double>(0, 1) * ly2 + per_mat_tosrc.at<double>(0, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * lx2 + per_mat_tosrc.at<double>(2, 1) * ly2 + per_mat_tosrc.at<double>(2, 2));

	new_ly2 = (per_mat_tosrc.at<double>(1, 0) * lx2 + per_mat_tosrc.at<double>(1, 1) * ly2 + per_mat_tosrc.at<double>(1, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * lx2 + per_mat_tosrc.at<double>(2, 1) * ly2 + per_mat_tosrc.at<double>(2, 2));

	int new_rx1, new_ry1, new_rx2, new_ry2;
	new_rx1 = (per_mat_tosrc.at<double>(0, 0) * rx1 + per_mat_tosrc.at<double>(0, 1) * ry1 + per_mat_tosrc.at<double>(0, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * rx1 + per_mat_tosrc.at<double>(2, 1) * ry1 + per_mat_tosrc.at<double>(2, 2));

	new_ry1 = (per_mat_tosrc.at<double>(1, 0) * rx1 + per_mat_tosrc.at<double>(1, 1) * ry1 + per_mat_tosrc.at<double>(1, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * rx1 + per_mat_tosrc.at<double>(2, 1) * ry1 + per_mat_tosrc.at<double>(2, 2));

	new_rx2 = (per_mat_tosrc.at<double>(0, 0) * rx2 + per_mat_tosrc.at<double>(0, 1) * ry2 + per_mat_tosrc.at<double>(0, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * rx2 + per_mat_tosrc.at<double>(2, 1) * ry2 + per_mat_tosrc.at<double>(2, 2));

	new_ry2 = (per_mat_tosrc.at<double>(1, 0) * rx2 + per_mat_tosrc.at<double>(1, 1) * ry2 + per_mat_tosrc.at<double>(1, 2)) /
		(per_mat_tosrc.at<double>(2, 0) * rx2 + per_mat_tosrc.at<double>(2, 1) * ry2 + per_mat_tosrc.at<double>(2, 2));

	warp_left_line.push_back(Point(new_lx1, new_ly1)); warp_left_line.push_back(Point(new_lx2, new_ly2));
	warp_right_line.push_back(Point(new_rx1, new_ry1)); warp_right_line.push_back(Point(new_rx2, new_ry2));


	line(frame, Point(new_lx1, new_ly1), Point(new_lx2, new_ly2), Scalar(0, 255, 255), 2);
	line(frame, Point(new_rx1, new_ry1), Point(new_rx2, new_ry2), Scalar(0, 255, 255), 2);

	int offset = 400;
	int lpos = int((offset - warp_left_line[0].y) * ((warp_left_line[1].x - warp_left_line[0].x) / (warp_left_line[1].y - warp_left_line[0].y)) + warp_left_line[0].x);
	int rpos = int((offset - warp_right_line[0].y) * ((warp_right_line[1].x - warp_right_line[0].x) / (warp_right_line[1].y - warp_right_line[0].y)) + warp_right_line[0].x);
	vector<Point> pos;
	pos.push_back(Point(lpos, rpos));

	return warp_left_line, warp_right_line, pos;
}

vector<Point> n_window_sliding(int left_start, int right_start, Mat roi, Mat v_thres, int w, int h,
	vector<Point>& lpoints, vector<Point>& rpoints, Mat per_mat_tosrc, Mat frame) {
	// define constant for sliding window
	int nwindows = 12;
	int window_height = (int)(h / nwindows);
	int window_width = (int)(w / nwindows * 1.5);

	int margin = window_width / 2;

	// 양쪽이 인식이 되었다면 초기화하고 다시 입력
	vector<Point> mpoints(nwindows);

	// init value setting
	int lane_mid = w / 2;

	int win_y_high = h - window_height;
	int win_y_low = h;

	int win_x_leftb_right = left_start + margin;
	int win_x_leftb_left = left_start - margin;

	int win_x_rightb_right = right_start + margin;
	int win_x_rightb_left = right_start - margin;

	lpoints[0] = Point(left_start, (int)((win_y_high + win_y_low) / 2));
	rpoints[0] = Point(right_start, (int)((win_y_high + win_y_low) / 2));
	mpoints[0] = Point((int)((left_start + right_start) / 2), (int)((win_y_high + win_y_low) / 2));

	// init box draw
	rectangle(roi, Rect(win_x_leftb_left, win_y_high, window_width, window_height), Scalar(0, 150, 0), 2);
	rectangle(roi, Rect(win_x_rightb_left, win_y_high, window_width, window_height), Scalar(150, 0, 0), 2);



	// window search start, i drew the init box at the bottom, so i start from 1 to nwindows
	for (int window = 1; window < nwindows; window++) {

		win_y_high = h - (window + 1) * window_height;
		win_y_low = h - window * window_height;

		win_x_leftb_right = left_start + margin;
		win_x_leftb_left = left_start - margin;

		win_x_rightb_right = right_start + margin;
		win_x_rightb_left = right_start - margin;

		int offset = (int)((win_y_high + win_y_low) / 2);

		int pixel_thres = window_width * 0.2;

		int ll = 0, lr = 0; int rl = 960, rr = 960;
		int li = 0; // nonzero가 몇개인지 파악하기 위한 벡터에 사용될 인자
		// window의 위치를 고려해서 벡터에 집어넣으면 불필요한 부분이 많아질 수 있다. 어차피 0의 개수를 구하기 위한 벡터이므로 0부터 window_width+1 개수만큼 생성
		vector<int> lhigh_vector(window_width + 1); // nonzero가 몇개 인지 파악할 때 사용할 벡터
		for (auto x = win_x_leftb_left; x < win_x_leftb_right; x++) {
			li++;
			lhigh_vector[li] = v_thres.at<uchar>(offset, x);

			// 차선의 중앙을 계산하기 위해 255 시작점과 255 끝점을 계산
			if (v_thres.at<uchar>(offset, x) == 255 && ll == 0) {
				ll = x;
				lr = x;
			}
			if (v_thres.at<uchar>(offset, x) == 255 && lr != 0) {
				lr = x;
			}
		}

		int ri = 0;
		vector<int> rhigh_vector(window_width + 1);
		for (auto x = win_x_rightb_left; x < win_x_rightb_right; x++) {
			ri++;
			rhigh_vector[ri] = v_thres.at<uchar>(offset, x);
			if (v_thres.at<uchar>(offset, x) == 255 && rl == 960) {
				rl = x;
				rr = x;
			}
			if (v_thres.at<uchar>(offset, x) == 255 && lr != 960) {
				rr = x;
			}
		}

		// window안에서 0이 아닌 픽셀의 개수를 구함
		int lnonzero = countNonZero(lhigh_vector);
		int rnonzero = countNonZero(rhigh_vector);


		// 방금 구했던 255 픽셀 시작 지점과 끝 지점의 중앙 값을 다음 window의 중앙으로 잡는다.
		if (lnonzero >= pixel_thres) {
			left_start = (ll + lr) / 2;
		}
		if (rnonzero >= pixel_thres) {
			right_start = (rl + rr) / 2;
		}


		// 차선 중앙과 탐지한 차선과의 거리 측정
		int lane_mid = (right_start + left_start) / 2;
		int left_diff = lane_mid - left_start;
		int right_diff = -(lane_mid - right_start);

#if 1
		// 한쪽 차선의 nonzero가 임계값을 넘지 못할 경우 중간을 기점으로 반대편 차선 위치를 기준으로 대칭
		if (lnonzero < pixel_thres && rnonzero > pixel_thres) {
			lane_mid = right_start - right_diff;
			left_start = lane_mid - right_diff;
		}
		else if (lnonzero > pixel_thres && rnonzero < pixel_thres) {
			lane_mid = left_start + left_diff;
			right_start = lane_mid + left_diff;
		}
#else
		// 지난 프레임에서의 픽셀값을 기억하고 nonzero가 임계값을 넘지 못할 경우 지난 프레임의 해당 윈도우 번호의 값을 불러옴
		if (lnonzero < pixel_thres && rnonzero > pixel_thres) {
			left_start = lpoints[window].x;
			lane_mid = (right_start + left_start) / 2;
		}
		else if (lnonzero > pixel_thres && rnonzero < pixel_thres && rpoints[window].x != 0) {
			right_start = rpoints[window].x;
			lane_mid = (right_start + left_start) / 2;
		}

#endif
		// draw window at v_thres
		rectangle(roi, Rect(win_x_leftb_left, win_y_high, window_width, window_height), Scalar(0, 150, 0), 2);
		rectangle(roi, Rect(win_x_rightb_left, win_y_high, window_width, window_height), Scalar(150, 0, 0), 2);


		mpoints[window] = Point(lane_mid, (int)((win_y_high + win_y_low) / 2));
		lpoints[window] = Point(left_start, (int)((win_y_high + win_y_low) / 2));
		rpoints[window] = Point(right_start, (int)((win_y_high + win_y_low) / 2));
	}

	Vec4f left_line, right_line, mid_line;
	fitLine(lpoints, left_line, DIST_L2, 0, 0.01, 0.01); // 출력의 0,1 번째 인자는 단위벡터, 3,4번째 인자는 선 위의 한 점
	fitLine(rpoints, right_line, DIST_L2, 0, 0.01, 0.01);
	fitLine(mpoints, mid_line, DIST_L2, 0, 0.01, 0.01);

	// 방향이 항상 아래를 향하도록 만들기 위해 단위 벡터의 방향을 바꿔준다.
	if (left_line[1] > 0) {
		left_line[1] = -left_line[1];
	}
	if (right_line[1] > 0) {
		right_line[1] = -right_line[1];
	}
	if (mid_line[1] > 0) {
		mid_line[1] = -mid_line[1];
	}

	int lx0 = left_line[2], ly0 = left_line[3]; // 선 위의 한 점
	int lx1 = lx0 + h / 2 * left_line[0], ly1 = ly0 + h / 2 * left_line[1]; // 단위 벡터 -> 그리고자 하는 길이를 빼주거나 더해줌
	int lx2 = 2 * lx0 - lx1, ly2 = 2 * ly0 - ly1;

	int rx0 = right_line[2], ry0 = right_line[3];
	int rx1 = rx0 + h / 2 * right_line[0], ry1 = ry0 + h / 2 * right_line[1];
	int rx2 = 2 * rx0 - rx1, ry2 = 2 * ry0 - ry1;

	int mx0 = mid_line[2], my0 = mid_line[3];
	int mx1 = mx0 + h / 2 * mid_line[0], my1 = my0 + h / 2 * mid_line[1];
	int mx2 = 2 * mx0 - mx1, my2 = 2 * my0 - my1;

	line(roi, Point(lx1, ly1), Point(lx2, ly2), Scalar(0, 100, 200), 3);
	line(roi, Point(rx1, ry1), Point(rx2, ry2), Scalar(0, 100, 200), 3);
	line(roi, Point(mx1, my1), Point(mx2, my2), Scalar(0, 0, 255), 3);

	vector<Point> warp_left_line(2), warp_right_line(2), pos;
	warp_left_line, warp_right_line, pos = matrix_oper(frame, per_mat_tosrc, lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2);

	return warp_left_line, warp_right_line, pos;
}


int main()
{
	VideoCapture cap("../data/subProject.avi");

	if (!cap.isOpened()) {
		cerr << "Camera open failed" << endl;
		return -1;
	}


	//csv 파일 생성
	ofstream CSVFILE("lane_pos.csv");
	CSVFILE << "index" << "," << "frame" << "," << "lpos" << "," << "rpos" << endl;
	int index = 0;

	// src image size
	int width = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int height = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));

	// warped image size
	int w = (int)width * 1.5, h = (int)height * 1.5;

	// point about warp transform
	vector<Point2f> src_pts(4);
	vector<Point2f> dst_pts(4);

	// 파란색 선 없는 roi
	src_pts[0] = Point2f(0, 395); src_pts[1] = Point2f(198, 280); src_pts[2] = Point2f(403, 280); src_pts[3] = Point2f(580, 395);
	dst_pts[0] = Point2f(0, h - 1); dst_pts[1] = Point2f(0, 0); dst_pts[2] = Point2f(w - 1, 0); dst_pts[3] = Point2f(w - 1, h - 1);

	// point about polylines
	vector<Point> pts(4);
	pts[0] = Point(src_pts[0]); pts[1] = Point(src_pts[1]); pts[2] = Point(src_pts[2]); pts[3] = Point(src_pts[3]);


	Mat per_mat_todst = getPerspectiveTransform(src_pts, dst_pts);
	Mat per_mat_tosrc = getPerspectiveTransform(dst_pts, src_pts);

	Mat frame, roi;
	vector<Point> warp_left_line(2), warp_right_line(2);
	vector<Point> lpoints(12), rpoints(12);

	while (true) {
		cap >> frame;

		if (frame.empty()) break;

		// perspective transform
		Mat roi;
		warpPerspective(frame, roi, per_mat_todst, Size(w, h), INTER_LINEAR);

		// roi box indicate
		polylines(frame, pts, true, Scalar(255, 255, 0), 2);

		// 2-1 hsv -> gaussian -> inRange -> canny
		Mat hsv;
		Mat v_thres = Mat::zeros(w, h, CV_8UC1);
		int lane_binary_thres = 125; // contrast : 155
		cvtColor(roi, hsv, COLOR_BGR2HSV);

		// split H/S/V
		vector<Mat> hsv_planes;
		split(roi, hsv_planes);
		Mat v_plane = hsv_planes[2];

		// inverse
		v_plane = 255 - v_plane;

		// brightness control
		int means = mean(v_plane)[0];
		v_plane = v_plane + (100 - means);

		GaussianBlur(v_plane, v_plane, Size(), 1.0);

		inRange(v_plane, lane_binary_thres, 255, v_thres);

		imshow("v_thres", v_thres);

		// 첫위치 지정
		int left_l_init = 0, left_r_init = 0;
		int right_l_init = 960, right_r_init = 960;
		for (auto x = 0; x < w; x++) {
			if (x < w / 2) {
				if (v_thres.at<uchar>(h - 1, x) == 255 && left_l_init == 0) {
					left_l_init = x;
					left_r_init = x;
				}
				if (v_thres.at<uchar>(h - 1, x) == 255 && left_r_init != 0) {
					left_r_init = x;
				}
			}
			else {
				if (v_thres.at<uchar>(h - 1, x) == 255 && right_l_init == 960) {
					right_l_init = x;
					right_r_init = x;
				}
				if (v_thres.at<uchar>(h - 1, x) == 255 && right_r_init != 960) {
					right_r_init = x;
				}
			}
		}

		int left_start = (left_l_init + left_r_init) / 2;
		int right_start = (right_l_init + right_r_init) / 2;


		vector<Point> pos;
		warp_left_line, warp_right_line, pos = n_window_sliding(left_start, right_start, roi, v_thres,
			w, h, lpoints, rpoints, per_mat_tosrc, frame);


		imshow("src", frame);
		imshow("roi", roi);



		//csv 파일 생성
		int frame_number = cap.get(CAP_PROP_POS_FRAMES) - 1;
		if (frame_number % 30 == 0)
		{
			int lpos = pos[0].x;
			int rpos = pos[1].x;

			CSVFILE << index << "," << frame_number << "," << lpos << "," << rpos << endl;
			index++;
		}

		if (waitKey(10) == 27) break;

	}
	cap.release();
	destroyAllWindows();
}
```

<br>

다양한 기능을 구현해보기 위해 github를 사용했고, 기능마다의 branch를 생성했다.

<img src="/assets/img/lanedetect/slidingwindow/branch.png">

<br>

<br>

# 요약

1. 슬라이딩 윈도우를 통한 ROI 설정
2. warpPerspective
3. HSV를 통한 이진화
    - V 평면만 사용
    - 반전을 통해 더 잘 탐지
    - 평균 밝기 유지
4. 슬라이딩 윈도우 사용
5. 첫 시작점을 잡아줄 때, max_element를 사용하면 동일한 크기의 픽셀 중 가장 앞의 위치를 추출하기에 다른 방법을 사용
    - 초기값 설정을 위해 왼쪽은 0, 오른쪽은 ROI 가로 사이즈인 960으로 지정
    - 255인 위치 중 위치 지정이 안되어 있다면 해당 인덱스를 지정
    - 해당 인덱스가 지정된 후부터 255가 끝날 때까지 끝 점을 지정해줌
    - 255로 되어있다가 0이 되면 if문이 종료된다.
6. 차선의 인덱스들을 통해 중앙값을 차선의 중앙으로 설정
7. slidiing window 진행
    - window개수는 최대한 줄여서 변화에 잘 적응할 수 있도록 12개 정도로 지정
    - window 가로와 세로는 유동적인 설정을 위해 직접 지정이 아닌 window 개수에 따라 변하도록 설정
    - margin은 중앙값을 기준으로 가로 길이/2이다. 즉, 중앙값을 입력으로 받았기 때문에 좌우 +- margin하여 window 좌측, 우측 좌표를 지정함
    - 기존의 sliding window는 윈도우를 만들고 나서 해당 윈도우에서 다음 윈도우의 시작 위치를 찾아주었다. 이렇게 하면 변화에 둔해지는 현상 발생
    - 따라서 탐색할 높이인 offset을 먼저 지정해준 후 탐지 하고 나서 그림을 그려준다. 이렇게 하면 변화에 적응을 더 잘 할 뿐더러 코너를 나름 잘 따라감
    - 이 때도, 차선을 찾기 위해 오른쪽 차선의 시작점, 끝점을 탐지하여 지정, 왼쪽 차선의 시작점, 끝점을 탐지하고, 해당 offset에서의 nonzero값을 구해서 임계값보다 높으면 탐지한 결과값으로 진행하고, 임계값보다 낮으면 지난 픽셀의 값을 사용하거나, 반대편 차선의 대칭으로 만들어줌
    - 이렇게 탐지한 왼쪽, 오른쪽 차선과 그로인해 구해진 중앙 차선을 변수에 저장
    - fitline을 통해 직선을 구한다. 이 때 출력되는 값의 0,1번째 값은 단위 벡터, 2,3번째 값은 선들의 중간 픽셀을 출력해준다.
    - 방향을 일정하게 맞추기 위해 무조건 방향벡터에서의 y를 -로 맞춰주었다.
    - 이를 통해 선 위의 점을 추출하고, 단위 벡터는 중앙값을 기준으로 방향이 맞춰져 있기 때문에, h로곱하는 것이 아닌 중앙값에서 h/2\*단위벡터를 구했다.
    - 중앙 픽셀과 끝 픽셀의 차를 통해 맨 아래 점을 찾아서 ROI화면의 맨 위 아래로 직선을 그렸다.
    - 출력값을 통해 ROI를 다시 frame 형태로변환하는 행렬을 통해 원본 frame에 점 좌표를 그리거나, 직선을 그리고자 햇지만, 아직 수행하지 못했다.
    - ROI를 파란색 선이 포함되도록 자르면 차가 조금만 움직여도 차선을 탐지하기 어려워지기 때문에 높이를 395에서 잘라서 ROI를 만들었다.
    - 아까 그린 좌우 차선의 직선 색을 통해 차선 중앙의 x좌표를 구하고 csv파일로 입력한다.
