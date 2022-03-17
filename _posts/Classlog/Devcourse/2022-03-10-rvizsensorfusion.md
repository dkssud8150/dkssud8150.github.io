---
title:    "[데브코스] 4주차 - RVIZ에서 모터와 센서 통합 "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-10 21:14:00 +0800
categories: [Classlog, devcourse]
tags: [ros, rviz, sensor, devcourse]
toc: True
comments: True
image:
  src: /assets/img/dev/week4/day4/sensorfusion.png
  width: 800
  height: 500
---

<br>

# RVIZ에서 모터와 센서 통합하기

RVIZ 가상공간에서 8자 주행하는 자이카에 라이다 센서와 IMU센서의 뷰어를 통합해본다. 3D 모델링된 차량이 8자 주행을 하면서 주변 장애물가지의 거리값을 Range로 표시하고 IMU 센싱값에 따라 차체가 기울어지도록 한다.

![](/assets/img/dev/week4/day4/2022-03-18-00-39-22.png)

1. driver노드 -\> /xycar_motor토픽 -\> converter노드 -\> /joint_states -\> RVIZ viewer, rviz_odom 노드 -\> /odom 토픽 -\> rviz
2. lidar_topic.bag -\> rosbag -\> /scan 토픽 -\> lidar_urdf.py 파이썬 -\> scan1~4 토픽 4개 -\> rviz
3. imu_data.txt -\> imu_data_generator 노드 -\> /imu 토픽 -\> rviz

odom노드는 imu토픽과 joint_states 토픽을 받아서 /odom 토픽을 발행해야 한다.

rviz_all 패키지

