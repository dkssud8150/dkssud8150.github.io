---
title:    "[데브코스] 18주차 - Camera-LiDAR Calibration "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-07-04 22:20:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, calibration]
toc: true
comments: true
math: true
---

&nbsp;

2D(카메라)와 3D(라이다) 좌표계를 서로 맞춰주는 것에 대해 설명하고자 한다.

&nbsp;

2D 이미지 좌표계와 3D world 좌표계 간의 변환을 하기 위해서는 변환식을 알아야 한다. 변환하는 방법에는 2가지가 있다. 하나는 **homography method**라고 하여, 3개 이상의 2d 이미지 좌표계와 그에 상응되는 3d 월드 좌표계 8개를 가지고 있다는 가정하에, homography matrix를 구할 수 있다. 

현재에는 이 방법이 아닌 다른 방법을 위주로 설명하고, 추후 더 추가하고자 하니, homography에 대해 더 자세히 알고자 한다면, openCV를 참고하길 바란다. 또는 지난 블로그를 참고해도 좋다.

- [OpenCV homography tutorial](https://docs.opencv.org/4.x/de/d45/samples_2cpp_2tutorial_code_2features2D_2Homography_2decompose_homography_8cpp-example.html)
- [homograghy 블로그](https://dkssud8150.github.io/posts/distestim/)

&nbsp;

# Projection Method

변환하는 또다른 방법으로는 projection method가 있다. Projection matrix라고도 하는 카메라 자체 특성인 intrinsic 정보들과 카메라에서 월드 좌표계로의 이동, 회전 변환인 extrinsic를 가지고, 변환할 수 있다. 

<img src="/assets/img/dev/projection_matrix.png">

- u, v : 이미지 좌표계
- U,V,W : 월드 좌표계

- K : camera calibration 정보 (intrinsic matrix)
- R, t  : 카메라와 LIDAR 간의 관계 (extrinsic matrix)

