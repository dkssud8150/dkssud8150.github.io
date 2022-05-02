---
title:    "[데브코스] 3주차 - Self driving Process "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-04 12:00:00 +0800
categories: [Classlog, devcourse]
tags: [ros, devcourse]
toc: True
comments: True
---

<br>

# 자율주행 자동차 기술 개요

## 자율주행 자동차는 어떤 기능을 제공해야 하는가

4차 산업혁명이랑 인공지능, 빅데이터, 초연결 등으로 촉발되는 지능화 혁명 또는 그 이상

자율주행 = 인공지능 + 빅데이터 + 초연결

## 자율주행 기술을 구분하는 6단계 구분법

미국 자동차기술학회(SAE)에서 구분한 자율주행 기술 6단계

<img src="/assets/img/dev/week3/day5/autodriving6stage.png">
<img src="/assets/img/dev/week3/day5/autodriving6stage2.png">

<br>

<br>

# 자율주행 프로세스

## 자율주행 프로세스 단계별 요소 기술

<img src="/assets/img/dev/week3/day5/skilleachprocess.png">

1. 인지
- 주변상황을 파악한다는 것, 경로 탐색, 차량간 통신, ADAS 센서
    → 고정지물(차로,차선,횡단보도,터널) 인식 및 경로탐색, 변동지물(차량,보행자,신호등,사고차량) 및 이동물체 인식
    
2. 판단
- 주변상황 판단 및 주행전략 결정 → 차선변경, 추월, 좌/우 회전, 정차
- 주행 경로 생성 → 목표궤적, 목표속도, 전방타겟 등
    
3. 제어
- 차량 제어 → 목표 조향각, 토크, 목표 가감속
- 엔진 가감속, 조향

<br>

<br>

- 사람의 경우 
    - 네비게이션의 도움을 받아 지도 제공 받음, 목적지까지 경로 찾기, 좌회전 우회전 경로 변경 안내를 받는다.
    - 사람은 차선과 지형지물을 보고 주행, 주변 차량 살펴서 주행하고, 교통 신호등을 인식해서 주행하고, 돌발 상황에 대응한다.
- 그러나 자율주행의 경우 
    - 보다 정확한 지도가 필요할 것
    - 지도에서 자기의 현재 위치를 알아내야 함 → GPS, 영상 매칭, 라이다 매칭
    - 목적지까지 경로 찾기
    - 주변 살피기 → 각종 센서
    - 상황에 맞게 속도 조정, 핸들 꺽기
        - 경로 그대로 따라가기 (차선 준수)
        - 교통신호 따르기
        - 예외상황에 대처 (추월 ,정차)
        - 돌발상황에 대처 (급브레이크, 장애물 피하기)

<br>

<br>

아래 1~8은 인지, 9,10은 판단, 11은 제어에 해당한다.

## 전체 프로세스

1.지도 생성

먼저 지도는 HD map(High Definition Map)이어야 한다.

- 기존 네비게이션, 티맵보다 더 정밀한 10~20센티 정밀도를 제공해야 한다.
- 다양한 부가정보 포함
    - 차선 정보
    - 가드레일
    - 도로 곡률, 경사
    - 신호등 위치
    - 교통 표식

고정밀 지도의 종류로는 벡터맵, 포인트맵 등이 있다.아래 사진이 벡터맵에 대한 것이다. 벡터맵은 차선 정보, 인식 정보 등 모든 정보가 담긴 맵핑된 지도다. 

<img src="/assets/img/dev/week3/day5/vectormap.png">

<br>

포인트맵은 라이다를 이용해서 만든 지도이다. 벡터맵은 카메라로 한 것이기에 2차원으로 보이지만, 이는 라이다를 통해 만든 것이므로 3차원으로 보인다. 

<img src="/assets/img/dev/week3/day5/pointmap1.png">
<img src="/assets/img/dev/week3/day5/pointmap2.png">

3차원의 좌표축의 점들을 표시한 것이고, 사진으로는 거리정보는 잘 알기 힘들다. 그러나 라이다는 거리를 알 수 있다. 따라서 벡터맵은 2차원, 포인트맵은 3차원으로 보이는 것이다.

<br>

이 두개를 합쳐서 사용하게 되면 아래의 그림이 된다.

<img src="/assets/img/dev/week3/day5/fusionmap1.png">
<img src="/assets/img/dev/week3/day5/fusionmap2.png">
<img src="/assets/img/dev/week3/day5/fusionmap3.png">

<br>

<br>

맵을 제작할 때는 MMS(Mobile Mapping System)을 사용한다. 이는 데이터를 수집한 다음 후처리 작업을 통해 지도를 제작하는 것을 말한다.

<img src="/assets/img/dev/week3/day5/mms.png">

<br>

<br>

2.위치 조정

정밀 지도와 연동하여 차량의 현재 위치를 파악하는 기술을 **Localization**이라 한다. 이는 라이다와 카메라를 이용하여 지도의 어느 위치에 있는지 파악한다. 

카메라가 기후 악화 등으로 잘 인식하지 못할 때는 라이다와 가지고 있는 지도를 통해 인식한다. 라이다와 지도를 비교하여 표지판이나 고정지물의 위치를 대조해보고 일치하도록 위치를 조정한다.

<img src="/assets/img/dev/week3/day5/mapcontrol.png">

이렇게 라이다와 비교하는 이유는 GPS가 오차가 존재할 수 있기 때문이다.

<br>

<br>

3.목적지 루트 설정 (Route Planning)

목적지까지 경로를 찾는 것을 Route Planning이라 한다. 중간 목적지 또는 최종 목적지까지의 경로를 찾을 수 있고, 경로에 맞게 각 교차로에서의 행동도 결정해야 한다.

<img src="/assets/img/dev/week3/day5/routeplan.png">

<br>

3-1. local path planning (trajectory planning)

다음 이동할 곳으로의 경로를 찾는 것이다. 이는 충돌을 회피할 때 고려되는 것으로 여러 개의 후보 경로를 확보해놓고, 끊임없이 후보를 삭제, 신규 후보를 등록하는 작업을 반복한다.

<img src="/assets/img/dev/week3/day5/trajectoryplan.png">

이 주황색 영역으로 이동을 해야하는데, 수많은 경로 후보를 생성하고, 추가/삭제를 반복한다. 짧은 시간안에 수행되어야 하므로 다시 경로를 계산하는 것이 아닌 10개정도의 경로 후보를 생성해 놓는 것이다.

<br>

<br>

4.객체 검출 (object detection)

주행 중 주변 차량, 보행자, 오토바이, 자전거, 신호등 등을 인식해야 한다. 이 객체를 인식하는 기술을 object detection이라 한다.

<img src="/assets/img/dev/week3/day5/objectdetect.png">

이 그림은 Monocular 3D Object Detection for Autonomous Driving 논문에서 발췌한 사진이다.

<br>

<br>

5.객체 추적 (object tracking)

각 객체를 인식만 하는 것이 아니라 추적하고, 카메라 밖을 벗어나면 추적을 멈추어야 한다. 그럴 때 사용하는 기술을 object tracking이라 한다.

<img src="/assets/img/dev/week3/day5/tracking.gif">

이 때는 각 객체에 ID를 부여하고, 예상되는 주행경로도 예측해야 한다.

<br>

6.Prediction

<img src="/assets/img/dev/week12/day1/prediction.png">

perception중에서 prediction이라는 기술도 존재한다. 이는 tracking에 기반한 객체의 현재까지의 움직임과 객체의 고유한 특징 등 다양한 정보를 바탕으로 `객체의 미래 움직임을 추정`한다. `Multimodal Trajectory Prediction`이라는 키워드로 연구가 되고 있다. 예를 들어 특정 자동차가 좌회전 깜박이를 킨다면 왼쪽으로 갈 확률이 높으므로 왼쪽으로 움직인다고 예측할 수 있다. 사람의 움직임도 예측해야 하는데, 차에 비해 사람은 전후좌우 즉각적으로 다 가능하므로, 차인지 사람인지에 따라서 예측이 또 달라질 수 있다.

<img src="/assets/img/dev/week12/day1/prediction.gif">

<br>

7.3D Pose Estimation

<img src="/assets/img/dev/week12/day1/3dpose.png">

객체를 인식하는 것 뿐만 아니라 객체의 정확한 위치를 추정하는 것이 중요하다. 대체로 이미지 안에서는 객체를 잘 찾는데, 그 객체에 대해 실제적인 위치를 알아야 사고가 나지 않으므로 인식 뿐만 아니라 객체의 위치까지 추정해야 한다. 그를 위해 bird's eye view로 변환하여 추정할수도 있고, 이를 `Multiple View Geometry` 분야라 말하기도 하지만, lidar 데이터를 통해 위치 정보를 통해 객체 위치를 찾는 것이 훨씬 더 쉽다. 이 lidar 정보와 카메라 정보를 융합하는데, 카메라에서 찾은 객체 정보와 lidar에서의 점 분포를 관계를 만들어서 3차원 객체 탐지를 한다.

lidar는 객체를 구분할 수 없지만, 정확한 위치 정보를 얻을 수 있고, 카메라는 객체의 위치 정보를 파악하기 어렵지만, 대상이 무엇인지는 구별할 수 있다.

<br>

8.Sensor fusion

lidar와 카메라의 각각의 장단점을 융합하기 위해 sensor fusion을 수행한다. lidar 데이터를 이미지 데이터에 그대로 적용을 시키고, 객체를 탐지한 bbox안에 있는 값들만 골라서 lidar의 x,y,z를 추정해서 3D object detection을 수행한다.

이외에 GPS와 IMU를 함께 사용해도 sensor fusion에 해당한다.

<br>

**위의 모든 것들이 결합되어 하나의 맵을 만든다.**

<img src="/assets/img/dev/week3/day5/allmap1.png">
<img src="/assets/img/dev/week3/day5/allmap2.png">

<br>

<br>

9.행동 결정 (behavior selector) 

지도가 만들어지고 나면 다음 행동을 결정해야 한다. 이를 행동 결정이라 하고, 운전 방법 및 성향을 반영할수도 있다.

<img src="/assets/img/dev/week3/day5/behaviorselect.png">

<br>

10.경로 따라 운전 (trajectory following)

경로를 따라 차량을 운전하는 것을 말한다.

<img src="/assets/img/dev/week3/day5/trajectoryfollow1.png">
<img src="/assets/img/dev/week3/day5/trajectoryfollow2.png">

1. 차량의 현재 위치를 결정
2. 차량에서 가장 가까운 경로상의 점을 찾는다
3. 목표점을 찾는다
4. 곡률을 계산, 해당 곡률로 차량의 방향을 업데이트
5. 차량의 위치를 업데이트

추가로 경로를 따라가기 위한 알고리즘으로 **Pure pursuit 알고리즘**이 있다.

<br>

<br>

11.vehicle control

다 결정이 되었으니 주행을 제어해야 한다. 이 때는 차량 운동학, 관성, 마찰력 등을 고려해야 한다.

<img src="/assets/img/dev/week3/day5/vehiclecontrol.png">

<br>

12.Acceleration

이 외에도, 장비의 성능과 속도에 대해서도 중요하다. 최근에는 perception에 딥러닝이 완전히 자리잡았다. 그로 인해 데이터도 많아지고 처리해야 하는 양도 많아졌다. 그러나 자율주행은 빠른 속도로 움직이는 환경이므로 검출 성능 뿐만 아니라 검출 속도도 중요하다. 그래서 최적화 방법으로 `Model Quantization`, `Pruning`, `Hardware Optimizaion` 등이 있다. Model quantization은 모델 양자화로, 데이터 타입들을 속도를 빠르게 만들기 위해 변경한다는 의미이다. 즉, float32를 float16으로 변경하거나 int로 변경하여 연산하는 것을 말한다. Pruning은 가지치기로 네트워크에서 출력에 영향이 별로 없는 가지(노드)를 삭제해버리는 기술이다. hardware optimization은 자신의 하드웨어에 맞게 최대한 빠르게 끌어 올리는 방법이다.

<br>

<br>

## 전체 프로세스

<img src="/assets/img/dev/week3/day5/allprocess2.png">
<img src="/assets/img/dev/week3/day5/allprocess.png">

- global planner: 지도가 global planning으로 연결되어 있는데, 넓은 단위의 자동차 내비게이션 역할을 한다.
- sensors : 위치와 가속도값을 결합하고, encoder는 자동차의 회전 카운트를 의미한다.
- perception : od(object detection)이나 global panner과 결합하여 지도에서의 내 위치를 결정짓는 용도로 사용
- local panner : 차선 변경이나 물체 회피 등의 용도로 사용되는 것으로 어디로 이동을 할지를 결정

<br>

detection에서도 lane detection, traffic light detection, traffic sign detection, object detection & tracking으로 나뉠 수 있다. 카메라를 통해 lane detection ,traffic detection, object detection을 수행할 수 있고, OD의 경우 RADAR, LIDAR와 융합하여 센서 퓨전을 할 수 있다. OD를 하다가 가려지거나 추출되지 않을 경우 이전 프레임의 정보를 사용하는 tracking을 활용할 수도 있다.



<br>

<br>

# 오픈소스 자율주행 통합 플랫폼

## Autoware

오토웨어는 자율주행 통합 플랫폼이다. 주소는 [https://www.autoware.org/](https://www.autoware.org/)이고, 레벨2 정도에 해당하는 기술력을 가지고 있으며, 실차에 적용이 가능한 솔루션이다. 현재 100개 이상의 회사들에 의해 사용되고 있다.

아래는 전체 프로세스에 대한 그림이다.

<img src="/assets/img/dev/week3/day5/autowareprocess.png">

<br>

<img src="/assets/img/dev/week12/day1/autowareoverview.png">

이는 조금더 보기 편하게 만들어진 그림으로 sensing, computing, actuation으로 구성되어 있다. 

<br>

그리고 아래 그림은 오토웨어 소프트웨어의 스택(아키텍처)에 대한 사진이다.

<img src="/assets/img/dev/week3/day5/autowarestack.png">

하드웨어로는 여러 장치가 있고, OS는 Linux를 사용하고 있다. 런타임 라이브러리로는 ros, cuda, caff, opencv 등이 있다.

<br>

<br>

## Apollo

[apollo github](https://github.com/apolloauto)

<img src="/assets/img/dev/week12/day1/apollo.png">

- V2X adapter : vehicle to everything, 자동차가 현재 어떤 신호를 내보내고 있는지 어딘가로 보낼 수 있는데, 이렇게 서로 통신하는 것을 V2X라 한다.
- Simulation : 자동차를 실험하기 위해서는 비용이 너무 많이 들엇허 simulation을 많이 사용한다.
- HD Map : 정밀 지도
- OTA : over-the-air, 무선으로 소프트웨어를 관리하고, 업데이트하는 방법을 말한다.

<br>

- apollo software overview

<img src="/assets/img/dev/week12/day1/apollosoftware.png">

apollo는 전체 프레임워크를 제공하기 때문에, monitor와 HMI를 제공한다. 이는 자동차에 탑승한 사람에게 자동차가 어떻게 움직이는지 알려주기 위한 용도이다. 

- CANBus : 내부 네트워크를 말하는 용어
- Guardian : 보안과 관련된 것으로, 내부에서 내리던 명령이 외부에서 해킹을 해서 명령을 내리면 너무 위험하므로 보안을 추가


<br>

<br>

# LiDAR vs Vision

자율주행의 현재 상태는 크게 `lidar + 정밀 지도 기반의 자율주행`과 `camera + 비정밀 지도 기반의 자율주행`으로 나뉠 수 있다. 정밀 지도(HD map)은 x,y,z,d 의 정보를 점 형태로 다 저장하기 때문에 용량이 너무 크지만, 이름 그대로 정밀한 정보를 얻을 수 있다. 전자의 경우 waymo, 후자의 경우 tesla, mobileye가 있다. 

<img src="/assets/img/dev/week12/day1/waymo.jpg">
<img src="/assets/img/dev/week12/day1/tesla.png">

waymo의 경우 3개의 lidar과 전방 주시하는 1개의 카메라가 존재한다. waymo는 HD map을 사용하고 있다. 그에 반해 tesla는 매우 많은 카메라만을 사용하고 있다. 여기에 radar는 최근에 사용하지 않고도 잘 주행이 가능하다고 한다. 그리고 tesla는 정밀 지도가 아닌 다른 지도를 만들어서 사용한다.

<br>

## HD map vs Navigation map

<img src="/assets/img/dev/week12/day1/hdmap.jpg">
<img src="/assets/img/dev/week12/day1/navimap.png">

정밀 지도(HD map)은 자율주행에 필요한 많은 사전 정보를 가지고 있고, 이를 통해 지도로 만든 것이다. HD map은 차선 단위의 지도로 높은 정밀도를 가지지만, 지도 제작이 어렵고 비싸다. 그에 반해 navigation map을 일반적인 내비게이션에서 사용하는 맵이다.

정밀 지도를 사용하지 않는 경우라도 네비게이션 맵은 반드시 사용할 수밖에 없다. 최종 목표로 도착에 필요한 global planning을 하기 위한 맵이 필요하다. local planning은 센서가 탐지가능한 거리에서의 planning이다. 

<br>

## REM

<img src="/assets/img/dev/week12/day1/rem.png">

HD map이 많은 정보를 가지지만, 이에 대한 지도 제작과 유지 보수가 어려워서 다른 방법을 채택한 회사도 있다. 그 대표적인 회사가 Mobileye, Tesla이다. REM(road experiment management)라 하여 도로에서 발생하는 다양한 경험들을 관리하겠다는 의미이다. 이는 크게 3가지 단계로 나뉠 수 있다.

<img src="/assets/img/dev/week12/day1/3steprem.png">

- harvesting
    - ADAS기능이나 카메라를 이용한 다양한 기술이 들어가 있다. 차선에 대한 곡률과 위치를 통해 주행해야 하는 가이드 path도 뽑을 수 있다.
- alignment
    - 위의 harvesting은 1대에 대한 주행 정보이지만, 이를 n대의 차량을 주행하여 모든 데이터를 모아놓으면 매우 많은 정보를 가질 수 있다. 사람마다 빠르게 달리는 사람이 있고, 느리게 달리는 사람이 있기 때문에, 그에 대해 누적 데이터를 만들고, 그에 대한 알맞는 경계값들을 조정한다.
- modeling & semantic
    - 이 누적된 데이터를 통해 정밀지도와 유사한 선 형태의 지도를 만들 수 있게 된다.

이 방법은 사람이 차량을 구매한 후에도 업데이트가 지속적으로 가능하다. 운전자가 직접 데이터를 모아서 컴퓨터가 alignment를 수행하여 지도를 계속 조정하게 되면 사람마다 알맞는 차량의 기능이 만들어진다.

<br>

## Vector space

<img src="/assets/img/dev/week12/day1/vectorspace.png">

tesla는 5~6대의 카메라가 존재한다. 이 카메라들이 서로 다른 화면을 보고 있지만, 같은 객체임을 인지하는 것이 중요하다. 그러나 카메라를 활용해서 위치 정보를 특정하는 것은 매우 어렵다. 그래서 tesla는 5~6대를 가지고 가상의 카메라 하나를 만들어서 각각의 카메라마다의 이미지를 1개의 이미지로 융합한다. 이를 통해 `vector space`라고 하는 공간을 찾게 되었다.

<br>

<br>

<br>

# 자율주행 자동차의 구현에 필요한 기술

1. 자율주행 알고리즘
    - 센싱, 인지, 의사 결정
2. 자율주행 클라이언트 시스템
    - 소프트웨어, 하드웨어
3. 자율주행 클라우드 플랫폼
    - 분산 컴퓨팅, 분산 스포티지
    
<br>

## 자율주행 알고리즘

1. 센싱
- 주변 정보 획득
    - GPS
    - IMU
    - 라이다
    - 카메라
    - 초음파

2. 인지
- Localization 자기 위치 파악
    - GPS와 IMU를 조합해서 위치 측정 (위치 예측과 업데이트 반복)
    - Stereo 카메라 영상으로 위치 측정
    - 라이다, 포인트 클라우드, 파이클 필터로 위치 측정
    - 여러 센서를 융합하여 정확도 개선
- Object detection
    - 딥러닝 기반의 인식모델 사용
- Object tracking
    - 객체 이동 궤적 추적
    - 차량, 보행자와의 충돌 회피

3. 의사 결정
- 동작 예측
    - 다른 차량의 동작을 예측
    - 확률 모델을 만들어 확률 분포 구하기
- 경로 계획
    - cost function으로 최적 경로 탐색
    - 계산량 줄이기 위해 확률 기법 적용
- 장애물 회피
    - 1단계 능동형: 충돌까지의 시간과 최소거리 추정치 뽑아서 경로를 다시 계획
    - 2단계 반응형: 이동경로상에 장애물이 감지되면 주행제어 시스템에 개입

<br>

## 자율주행 클라이언트 시스템

1. 소프트웨어
- 실시간성과 신뢰성 확보
- ROS 문제점 해결 필요
    - master가 죽으면 전체시스템이 다운되며 복구용도의 모니터가 없음
    - 메시지를 브로드캐스팅하면서 성능이 저하됨. 멀티캐스팅 메커니즘 적용하면 좋다.
    - 노드가 해킹되기 쉬움, 리눅스 컨테이너, 샌드박스로 보안 기능 강화 가능
    - 로컬에서는 TCP/IP 통신 대신에 공유메모리 통신방식을 적용하면 좋다.

2. 하드웨어
- 성능 향상 필요
    - 파이프라인 병렬 프로세싱 기능 필요
    - HW 가성비 좋게
    - 차량의 배터리 문제가 있어 전력 소요량 최소화 노력이 필요
    - 차량이라는 환경에서는 발열문제가 심각하기에 발열을 최소화하거나 열을 쉽게 배출할 수 있는 방법이 필요

## 자율주행 클라우드 플랫폼

1. 분산 컴퓨팅
- 시뮬레이션
    - ROS Bag/Replay
    - 분산 환경으로 처리
- HD맵 생성
    - 원본데이터 처리
    - 포인트 클라우드 생성
    - 포인트 클라우드 정렬
    - 2D 반사맵 생성
    - HD맵 레이블링
    
2. 분산 스토리지
- 딥러닝 모델 학습
    - 학습데이터 준비
    - 학습 진행
    - 모델의 유효성과 효율성을 지속적으로 업데이트