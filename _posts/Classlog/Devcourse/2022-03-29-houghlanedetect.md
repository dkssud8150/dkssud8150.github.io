---
title:    "[데브코스] 7주차 - Lane Detection using Hough transformation "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-29 20:19:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, ros]
toc: True
comments: True
# image:
#   src: /assets/img/dev/week6/calra/main.jpg
#   width: 800
#   height: 500
---

<br>

# 허프변환 기반 차선인식 주행

카메라로 촬영된 차량 전방의 도로 영상을 OpenCV를 이용하여 허프 변환 기반으로 차선을 찾고 양쪽 차선의 위치를 따져서 핸들을 얼마나 꺾을지 조향각을 결정한다.

과정
1. 카메라 입력 데이터에서 프레임 추출
- 카메라 토픽 구독

2. 얻어낸 영상 데이터를 처리하여 차선 위치를 결정
- 색변환 : Grayscale
- 외곽선 추출 : canny 함수로 임계값 범위 주고 외곽선 추출
- ROI 잘라내기

3. 차선 검출 : 허프 변환으로 직선 찾기
- 양쪽 차선을 나타내는 평균 직선 구하기
- 수평선 긋고 양쪽 직선과의 교점 좌표 구하기

4. 차선위치를 기준으로 조향각 결정
- 차선의 중앙을 차량이 달리도록 만듬

5. 결정한 조향각에 따라 조향 모터를 제어
- 모터제어 토픽 발행

<br>

- 허프 변환을 이용한 차선 찾기 단계
1. image read
2. grayscale (흑백 이미지 전환)
3. gaussian blur (노이즈 제거)
4. canny (edge 검출)
5. roi (관심 영역 잘라내기)
6. houghlineP (선분 추출)
7. 차선 위치 파악
8. 핸들 조종

<br>

## 작업 환경 설정

- 기존에 생성했던 `hough_drive` 패키지에서 작업
- `hough_drive.py`
- `hough_drive.launch`
- `steering_arrow.png`

### 1. hough_drive.launch

```xml
<launch>
    <!-- 노드 실행 : 자이카 모터 제어기 구동 -->
    <!-- 노드 실행 : 자이카 카메라 구동 -->
    <!-- 노드 실행 : 허프변환 기반 차선인식 주행 프로그램인 hough_drive.py 실행 -->
</launch>
```

<br>

### 2. hough_drive.py

단계
1. 카메라 노드에서 토픽 구독해서 영상 프레임 획득
2. 영상 프레임을 OpenCV 함수로 변환
3. OpenCV 영상 처리
- Grayscale (흑백 이미지로 변환)
- gaussian blur (노이즈 제거)
- canny edge (외곽선 Edge 추출)
- roi (관심영역 잘라내기)
- houghlineP (선분 검출)
4. 차선의 위치 찾고 화면 중앙에서 어느쪽으로 치우쳤는지 파악
5. 핸들을 얼마나 꺽을지 결정 (조향각 설정 각도 계산)
6. 모터제어 토픽을 발행해서 차량의 움직임을 조종

<br>

