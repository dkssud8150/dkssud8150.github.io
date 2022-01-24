---
title:    "Multi-View 3D Object Detection Network for Autonomous Driving"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-01-17 12:00:00 +0800
categories: [Review, Autonomous Driving]
tags: [Autonomous Driving, MV3D]
toc: True
comments: True
math: true
mermaid: true
pin: true
image:
  src: _site/_site/assets/img/autodriving/MV3D/pointcloud.png
  width: 800
  height: 500
---

1. this ordered seed list will be replaced by the toc
{:toc}


Multi-View 3D Object Detection Network For Autonomous Driving 논문에 대한 리뷰한 글입니다. 이 논문은 2017 CVPR에 투고된 논문이며 약 1594회 인용되었다고 합니다. 혼자 공부하기 위해 정리한 내용이니 이해가 안되는 부분이 많을 수 있습니다. 참고용으로만 봐주세요.

<br>

# Abstract

이 논문은 자율주행 시나리오에서 3D Object Detection의 높은 정확도를 목표로 하고 있다. 저자는 LIDAR point cloud와 RGB 이미지를 입력으로 하여 3D object 경계 상자를 예측하는 센서 퓨전 프레임워크인 MV3D(Multi-View 3D networks)를 제안했다. 또, 소량의 다중 뷰(multi-view)만으로 3D point cloud를 인코딩했다. 네트워크는 `3D object 제안 생성`을 위한 네트워크와, `multi-view 특징 퓨전`을 위한 네트워크로 구성되어 있다. 제안 네트워크(proposal network)는 3D point cloud를 조감도(bird`s eye view)적 표현을 통해 3D 후보 box(3D candidate box)를 생성한다. 저자는 여러 관점(view)에서 지역별 특징을 결합하고, 서로 다른 경로의 중간 레이어들 간의 상호 작용을 가능하게 하기 위해 깊은 결합 체계를 설계했다. KITTI benchmark에서의 실험은 저자의 접근 방식이 3D localization과 3D Detection 태스크에서 25~30% AP정도 능가했다고 한다. 게다가, 2D detection에서는 LIDAR 기반의 최신 기술보다 14.9% AP가 더 높았다.

<br>

# 1. Introduction

3D object detection는 자율주행차에 시각적 인지 시스템에서 중요한 역할을 한다. 최근 자율주행 차는 보통 LIDAR과 카메라와 같은 다중 센서를 탑재하고 있다. 레이저 스캐너는 정확한 깊이 정보를 제공하는 반면 카메라는 훨씬 더 디테일한 의미 정보를 보존한다. LIDAR point cloud와 RGB 이미지의 융합은 자율주행 차에서 더 높은 수행과 정확도를 성취하게 될 것이다.

이 논문의 중점은 LIDAR와 image data 모두 활용한 3D Object Detection이다. 그래서 저자는 도로 상에서 3D localization과 물체 인지의 높은 정확도를 목표로 한다. 최근 LIDAR 기반의 방법론들은 3D voxel grid에 3D window를 배치하여 point cloud에 점수를 매기거나, dense box prediction 체계에서 전면 뷰 point 맵에 convolution network를 적용한다. image-based 방법론들은 대체로 먼저 3D box proposal들을 생성한 후 Fast R-CNN 파이프라인을 사용한 region-based 인지를 수행한다. LIDAR point cloud 기반의 방법론들은 보통 더 정확한 3D 위치를 성취하는 반면 이미지 기반의 방법들은 2D box 평가에 관한 더 높은 정확도를 가진다. 초기 또는 후기 융합 방식을 채택함으로써 2D dection을 위해 LIDAR과 이미지를 융합한다. 그러나 3D object detection 태스크에서는 여러 양식의 강점을 갖도록 더 잘 설계된 모델이 필요하다.

이 논문에서, 다방면의 데이터를 입력으로 받고 3D 공간에서 3D 물체를 예측하는 MV3D(Multi-View 3D object detection network)을 제안한다. 다방면의 정보를 활용하는 중심 아이디어는 지역기반의 특징 융합을 수행하는 것이다. 먼저 3D point cloud의 작고 효과적인 표현을 얻기 위한 체계를 인코딩하는 multi-view를 제안한다. 

<img src="/_site/assets/img/autodriving/MV3D/fig1.png">

이 그림에서 볼 수 있듯이 multi-view 3D dectection 네트워크는 `3D 제안 네트워크`, `Region-based 융합 네트워크` 2개로 구성되어 있다. 3D 제안 네트워크는 더 정확한 3D 후보 상자들을 발생시키기 위한 새 관점에서의 표현을 활용한다. 3D object 제안의 이점은 3D 공간에서 모든 뷰들을 투영할 수 있다는 것이다. multi-view 융합 네트워크는 다중 뷰들로부터 특징맵에 대한 3D 제안들을 투영함으로써 지역별 특징을 추출한다. 다른 뷰들로부터 중간 레이어들 간의 상호 작용을 가능하게 하기 위해 깊은 융합 방식을 설계했다. drop-path training과 보조 손실이 결합된 방식이 초기/후기 융합 체계에 비해 우수한 성능을 보여준다. 다중 뷰 특징 표현을 고려할 때, 네트워크는 3D 공간에서 물체의 정확한 3D 위치, 크기 및 방향을 예측하는 oriented 3D box regression을 수행한다. 

저자는 까다로운 KITTI object detection benchmark에서 3D 제안 생성, 3D localization, 3D detection, 3D detection 수행에 대한 접근 방식을 평가했다. 실험은 우리의 3D 제안이 최근 3D 제안 방법론인 3DOP과 Mono3D를 엄청 능가했다는 것을 보여준다. 특히, 300개의 제안에서는 IoU threshold가 0.25일때는 99.1%, 0.5일 때는 91% 3d recall을 얻는다. 이 논문의 LIDAR 기반의 방식은 3D localization task에서는 25% 더 높은 정확도를, 3D object detection task에서의 3D AP(average precision)에서는 30% 더 높은 정확도를 성취했다. 이것은 까다로운 KITTI test 데이터셋에서의 2D detection에 대한 14.9% AP를 가지는 다른 LIDAR 기반의 방법론를을 능가했다. 이미지와 융합한다면, LIDAR 기반의 결과에 대한 추가적인 개선이 가능하다.

<br>

# 2. Related Work

point cloud나 이미지에서 3D object detection, 다양한 융합 방법 및 3D object proposals에 대한 기존 연구를 간략히 검토하고자 한다.

## 3D object detection in point cloud

대부분의 존재하는 방법들은 voxel grid 표현에서의 3D point cloud를 인코딩한다. sliding Shapes와 Vote3D는 SVM 분류기를 사용하여 지역 특징을 인코딩한 3D grid들을 계산한다. 최근 제안된 방법들은 3D convolution을 통한 특징 표현을 개선한다. 하지만 비싼 계산을 요구한다. 3D voxel 표현 외에도, VeloFCN은 FV(Front view)에 대한 point cloud를 투영하여 2D point map을 얻는다. 여기서는 2D point map에 FCN(fully convlutional network)를 적용하고, convolutional feature map으로부터 3D box들을 예측한다. 3D 객체 분류를 위한 point cloud의 크기와 다중 뷰 표현을 조사한다. 본 연구에서는 멀티 뷰 feature map으로 3D point cloud를 인코딩하여 다양한 융합을 위한 지역 기반 표현을 가능하게 한다. 

<br>

## 3D Object Detection in Images

3DVP는 3D voxel 패턴을 소개하고, 2D detection과 3D pose estimation을 할 수 있는 ACF detector를 생산한다. 3DOP는 스테레오 이미지들로부터 Depth를 재구성하고, 에너지 최소화 접근 방식을 사용하여 3D box proposal을 생성하고, 이는 객체 인식을 위해 R-CNN 파이프라인에 공급된다. Mono3D는 3DOP와 같은 파이프라인을 구성하고 있고, 이것은 monocular images(단안상)으로부터 3D 제안들을 발생시킨다. 저자의 실험을 통해 3D localization을 개선하기 위한 LIDAR point cloud를 통합시키는 방법을 볼 수 있을 것이다.

<br>

## Multimodal Fusion

자율주행에서 오직 소수의 모델들만이 다양한 양식의 데이터를 사용한다. 저자는 FractalNet과 Deeply-Fused Net을 기반으로 한 깊은 융합 방식을 설계한다. FractalNet에서, 기본 방식은 기하급수적으로 증가하는 경로를 통해 네트워크를 구성하기 위해 반복한다. 간단하게, 얕은 서브 네트워크들과 깊은 서브 네트워크들을 융합함으로써 deeply-fused network를 구성하는 것이다. 이 논문의 네트워크는 각 열마다 동일한 베이스 네트워크를 사용하지만 정규화를 위한 보조 경로와 손실을 추가한다는 점에서 두 네트워크와 다르다.

<br>

## 3D Object proposals

2D object proposals와 유사하게, 3D object proposal 방법들은 3D 공간에서 객체 대부분을 복사하기 위해 작은 3D 후보 상자들을 생성한다. 이것을 위해 3DOP는 스테레오 point cloud의 일부 depth 특징들을 설계하여 큰 3D 후보 상자들을 점수매긴다. Mono3D는 ground 평면을 먼저 활용하고, 일부 분할 특징을 사용하여 단일 이미지에서 3D proposal을 생성한다. Deep Sliding Shapes는 더 강력한 딥러닝 특징들을 사용한다. 그러나, 이는 계산적으로 비싼 3D convolution을 사용하고 3D voxel grid안에서 작동된다. 그래서 bird`s eye view 표현을 생산하고, 더 정확한 3D proposal 발생시키기 위한 2D convolution을 생산함으로써 더 효과적인 접근 방식을 제안한다. 

<br>

# 3. MV3D Network

MV3D 네트워크는 이미지와 3D point cloud를 입력으로 하여 다중 뷰를 표현한다. 이는 먼저 bird`s eye view map와 region 기반의 표현을 통해 깊게 융합한 다중 뷰 특징들로부터 3D object proposal을 생성한다. 이 융합된 특징들은 classification과 3D box regression에 사용된다. 

## 3.1 3D point cloud Representation

현존하는 모델들은 보통 3D LIDAR point cloud를 3D grid나 FV(front view) map로 인코딩한다. 대부분의 point cloud의 기초적인 정보들은 3D grid 표현들을 보존하는데, 이는 그 다음의 특징 추출을 위해 더 복잡한 계산을 요구하게 된다. 그래서 저자는 아래 그림에서 볼 수 있듯이 bird`s eye view와 FV에 대한 3D point cloud를 투영함으로써 더 간단한 표현을 제안한다. 

<img src="/_site/assets/img/autodriving/MV3D/fig2.png">

* Bird's eye view representation

bird's eye view, 즉 조감도는 **height**(높이), **intensity**(강도), **density**(밀도)로 인코딩되어 있다. 저자는 투영된 point cloud를 0.1m의 해상도의 2D grid로 나눈다. 

각 셀의 **height** 특징들은 셀 안의 point들의 `최대 높이`로 계산되어 있다. 좀 더 자세한 높이 정보를 인코딩하기 위해, point cloud는 M개의 slices로 균등하게 나누어진다. height map은 각각의 slice로 계산되어 있기 때문에 M height maps를 얻을 수 있다. **intensity** 특징들은 각 셀의 최대 높이를 가지는 점의 반사도이다. **density**는 각 셀의 점들의 갯수를 나타낸다. 

특징들을 정규화시키기 위해 *min(1.0, log(N+1)/log(64))* 로 계산되는데 이때 N은 셀 안의 점들의 갯수를 나타낸다. 이 때, 중요한 것은 height feature은 M개의 slice에 의해, 즉 각 셀을 계산하는 반면, intensity와 density feature은 전체 point cloud에 대해 계산된다. 그래서 전체 조감도 map은 *(M+2) - channel*로 계산된다. 

* Front View Representation

전방 뷰는 조감도 표현에 대한 상호 보완적인 정보를 제공한다. LIDAR point cloud가 매우 밀도가 희박함에 따라, 이미지 평면에 이를 투영하는 것은 밀도가 희박한 2D point map를 얻는다. 따라서 밀도있는 전방 뷰 map을 발생시키기 위해 평면 대신 실린더 면(원통 면)에 투영한다. 3D point, *p = (x,y,z)*라고 할 때, 전방 뷰에 대한 point 표현은 *pfv = (r,c)*이고, r과 c에 대한 계산은 다음과 같다.

<img src="/_site/assets/img/autodriving/MV3D/fvpoint.png">

∆ϴ와 ∆ø는 각각 레이저의 수평, 수직에 대한 표현이다. fig2애서 볼 수 있듯이 저자는 height, distance, intensity의 3채널 특징을 통해 FV map을 인코딩한다.

<img src="/_site/assets/img/autodriving/MV3D/fig2.png">

<br>

## 3.2 3D Proposal Network

최첨단 2D object detector의 핵심 구성요소인 RPN(Region Proposals Network)에서 영감을 받아, 먼저 3D object proposal을 발생시키는 네트워크를 설계했다. 일단 조감도 맵을 입력으로 사용한다. 3D object detection에서 조감도 맵은 전방 뷰나 이미지 평면에 비해 몇 가지 장점이 있다. 먼저, 객체는 사이즈 변화가 작은 조감도 맵으로 투영했을 때 `물리적 사이즈가 보존`된다. 두번째로, 조감도 맵에서의 물체는 서로 다른 공간을 차지하기 때문에 `occlusion 문제`를 피할 수 있다. 세번째로 로드 뷰에서 물체는 대부분 지면에 놓여있거나 수직적 위치의 변화가 작기 때문에, 조감도 맵은 더 `정확한 3D bounding box`를 얻을 수 있다. 따라서 조감도 맵을 사용하면 3D 위치 예측이 더욱 실현 가능해진다.

네트워크는 3D prior boxes로부터 3D box proposals를 발생시킨다. 각 3D box는 *(x,y,z,l,w,h)*로 구성되어 있고, 각각 LIDAR 좌표계에서의 3D box의 중심 좌표(x,y,z)와 사이즈(meters)(l,w,h)를 나타낸다. 각 사전 box(prior box)에 해당하는 조감도 anchor*(xbv, ybv, lbv, wbv)*는 (x,y,l,w)를 통해 얻을 수 있다. 훈련 셋에서 GT(Ground Truth) object size를 군집 분류함으로써 N개의 3D prior boxes를 설계한다. 자동차 감지의 경우, (l,w)의 prior box는 {(3.9, 1.6), (1.0, 0.6)}의 값을 가지며, 높이 h는 1.56m를 가진다. 조감도 앵커를 90도 회전함으로써, N = 4의 prior boxes를 얻는다. (x,y)는 조감도 특징 맵에서의 위치이고, z는 카메라 높이와 물체 높이에 의해 계산된다. 저자는 제안 생성(proposal generation)에서 방향 회귀(orientation regression)를 하지 않고, 다음 예측 단계로 넘어간다. 3D box의 방향은 대부분의 로드 뷰에서의 물체의 실제 방향에 가까운 0~90도로 제한되어 있다. 이 간단함은 제안 회귀의 훈련을 쉽게 만들 것이다.

0.1m 해상도에 따라, 조감도의 물체 박스들은 오직 5~40 픽셀만 차지한다. 초소형 물체를 감지하는 것은 아직 어려운 문제지만, 하나의 해결책이 있다면 입력을 높은 해상도로 하는 것이다. 그러나 이는 더더욱 큰 계산을 요구한다. 그래서 *A unified multi-scale deep convolutional neural network for fast object detection* 논문의 업샘플링 방법을 채택하여, proposal network안의 마지막 convolution layer이후에, 2배 이선형(bilinear) 업샘플링을 사용한다. front-end convolution은 3번의 pooling작업을 하여 1/8배 다운샘플링을 진행한다. 이를 2배 업샘플링과 결합하여 proposal network에 제공된 특징 맵은 조감도 입력과 관련하여 1/4배 다운샘플링된다.

RPN과 유사하게 *t = (∆x,∆y,∆z,∆l,∆w,∆h)*에 대해 3D box regression한다. (∆x,∆y,∆z)는 anchor 크기에 의해 정규화된 중심 좌표이고, (∆l,∆w,∆h)는 다음과 같이 계산된 값이다.

<img src="/_site/assets/img/autodriving/MV3D/sizecompute.png">

물체/배경을 동시에 분류하고, 3D box regression을 수행하기 위해 multi-task loss를 사용한다. 특히, objectness loss와 smooth *l1*에 대한 cross-entropy를 사용하여 3D box regression loss를 구한다. box regresson loss를 구할 때는 배경 anchor는 무시한다. 

훈련하는 동안에, anchors과 GT 조감도 boxes들 사이의 IoU를 계산한다. IoU가 0.7이상이면 anchor가 positive이고, 0.5미만이면 negative가 된다. 그 사이의 값은 무시한다. LIDAR point cloud data가 많이 빈 anchor가 많기 때문에, 훈련과 테스트에서 계산을 줄이기 위해 모든 빈 anchor들을 제거한다. 

마지막 convolution 특징 맵의 각 위치에서의 비지 않은 anchor에서 네트워크가 3D box를 발생시킨다. 불필요한 반복을 줄이기 위해, NMS(Non-Maximum Suppression)을 조감도 박스들에 적용시킨다. 물체가 지면에서 각각 다른 공간을 차지해야 하기 때문에 3D NMS는 적용하지 않았다.

저자는 NMS를 위해 0.7 threshold의 IoU를 사용한다. 훈련에는 2000개의 top boxes를, 테스트에서는 300개의 top boxes를 유지시킨다.

<br>

## 3.3 Region-based Fusion Network

저자는 다중 뷰로부터 특징들을 효과적으로 결합하고, 객체 제안들을 분류하고, oriented 3D box regression을 하기 위해 region-based fusion network를 사용한다.

* Multi-View ROI Pooling

다른 뷰/방식들에 대한 특징들은 다른 해상도들을 가지고 있기 때문에, 같은 길이에 대한 특징 벡터를 얻기 위해 각각의 뷰에 ROI pooling을 적용한다. 생성된 3D proposals를 고려할 때, 3D 공간에서 모든 뷰에 대해 투영할 수 있게 된다. 이 논문에서는 BV(bird's eye view), FV(front view), RGB(image plane) 총 3개의 뷰에 대해 투영하였다.

<img src="/_site/assets/img/autodriving/MV3D/ROIproposal.png">

3D proposal, P*3D*를 고려할 때, T*3D -> v*는 각각 LIDAR 좌표계에서 조감도, 정면 뷰, 이미지 평면으로의 변환 기능을 나타내고, 위의 식에 따라 각 뷰에 대해 ROI를 얻을 수 있다. 각 뷰의 front-end network로부터 입력 특징 맵 x를 고려하여 ROI pooling에 의해 고정된 길이 특징,*fv*를 얻는다. 

<img src="/_site/assets/img/autodriving/MV3D/roipooling.png">

<br>

* Deep Fusion

<img src="/_site/assets/img/autodriving/MV3D/fig3.png">

다른 특징들과 정보를 결합하기 위해, 사전 작업은 보통 *초기 결합*이나 *후기 결합*을 사용한다. 하지만 저자는 다중 뷰 특징을 계층적으로 융합하는 *deep fusion* 방식을 적용한다. 초기/후기 결합 네트워크와 deep fusion 네트워크 아키텍쳐의 비교는 위의 그림에서 볼 수 있다.

<img src="/_site/assets/img/autodriving/MV3D/earlyfusion.png">

L개의 layers를 가진 네트워크의 경우, **초기 결합**은 입력 단계에서 다중 뷰들의 특징, *fv*들을 결합한다. 이 떄, {Hl,l = 1,...L} 은 특징 변환 함수이고, ⊕는 concatenation, summation과 같은 결합 연산이다. 

<img src="/_site/assets/img/autodriving/MV3D/latefusion.png">

대조적으로, **후기 결합**은 별도의 서브 네트워크를 사용하여 특징 변환을 독립적으로 수행하고 예측 단계에서 출력을 결합한다.

<img src="/_site/assets/img/autodriving/MV3D/intermediate.png">

그러나, 이 논문에서는 다른 뷰들에서 **중간 레이어**의 특징들 사이에 더 많은 상호작용을 가능하게 하기 위해 다음과 같은 **deep fusion** 프로세스를 설계했다. drop-path 훈련과 결합할 때 더 유연하기 때문에 이 논문에서는 deep fusion을 위한 결합 작업에 요소별 평균을 사용한다.

<br>

* Oriented 3D box Regression

다중 뷰 네트워크의 특징 결합을 고려할 때, 3D 제안들로부터 oriented 3D boxes를 회귀한다. 특히, regression(회귀) 타겟들은 3D boxes에 대한 8개의 모서리들이다. (t = (∆x0,...,∆x7,∆y0,...,∆y7,∆z0,...,∆z7)). 제안 박스의 대각선의 길이에 의해 정규호된 모서리 좌표들로 인코딩되어 있다. 총 24D 벡터 표현은 oriented 3D box를 나타내는 데 중복되지만, 저자는 이 인코딩 접근 방식이 축 정렬 3D box로 회귀하는 중심 및 크기 인코딩 접근 방식보다 더 잘 작동한다는 것을 발견했다고 한다. 이 논문의 모델에서, 객체 방향은 예측된 3D box 모서리로부터 계산될 수 있다. 우리는 객체 카테고리 및 oriented 3D boxes를 예측하는 multi-task loss를 사용한다. 

제안 네트워크에서 category loss는 cross-entropy를 사용하고, 3D box loss는 smooth *l1*을 사용했다. 훈련할 때 positive/negative ROI들은 조감도 boxes의 IOU 중첩을 기준으로 결정된다. 조감도 IOU 중첩이 0.5 이상일 때는 3D 제안이 positive가 되고, 나머지는 negative가 된다. 추론에서는 3D bounding box regression이후에 3D boxes에 NMS를 적용한다. 조감도에 대한 3D box들을 투영함으로써 IOU 중첩을 계산한다. 조감도에서 같은 공간을 차지할 수 없는 객체들을 가리키는 불필요한 박스를 제거하기 위해 0.05 IOU threshold를 사용한다. 

<br>

* Network Regularization

저자는 지역 기반의 결합 네트워크를 정규화하기 위해 `drop-path training`과 `auxiliary losses`, 두 가지 접근 방법을 사용했다. 각 반복마다 무작위로 50% 확률로 **global drop-path**와 **local drop-path**를 무작위로 선택한다. 만약 global drop path가 선택되면 동일한 확률로 3개의 뷰안에서 1개의 단일 뷰를 선택한다. 반대로 local drop-path가 선택되면, 각 결합 노드의 입력 경로는 50% 확률로 무작위로 drop된다. 적어도 1개의 입력 경로는 유지되게 한다. 그리고, 각 뷰의 표현 능력을 더욱 강화하기 위해 네트워크에 보조 경로와 손실을 추가한다. 

<br>

<img src="/_site/assets/img/autodriving/MV3D/fig4.png">

위의 그림에서 볼 수 있듯이, 보조 경로는 메인 네트워크와 같은 수의 layer를 가지고 있다. 보조 경로의 각 layer는 메인 네트워크의 연관된 층의 가중치를 공유한다. 그리고 동일하게 각 보조경로를 역전파하기 위해 classification loss와 3D box regression loss를 더한 multi-task loss를 사용한다. 보조 loss을 포함한 모든 loss는 동일하게 가중치를 부여한다. 보조 경로들은 추론에서는 삭제된다.

<br>

## 3.4 Implementation

* Network Architecture

다중 뷰 네트워크에서 각 뷰는 같은 아키텍처를 가진다. backbone 네트워크는 16-layer VGGNet으로 만들었고, 몇가지를 모델에 맞게 수정했다. 수정한 내용은 다음과 같다.

1. 채널을 원래 네트워크의 반으로 줄였다.
2. 초소형 객체를 다루기 위해 특징 근사를 사용하여 고해상도 특징 맵을 얻었다. 특히, 우리는 마지막 convolution 특징 맵과 3D porposal 네트워크 사이에 2배 이선형 업샘플링 layer를 삽입했다. 또한, BV/FV/RGB의 ROI pooling layer 앞에 4x/4x/2x 업샘플링 layer도 삽입했다.
3. VGG 네트워크 안에 있던 4번째 pooling 작업을 제거하여, convolution 부분에서 8배 다운샘플링을 진행한다.
4. 다중 뷰 결합 네트워크에서 원래는 FC 6 layer이나 FC 7 layer 였던 FC에 1개 FC layer를 추가했다.

imageNet으로 pretrained된 VGG16 네트워크의 가중치를 샘플링하여 파라미터를 초기화했다. 우리의 네트워크는 3개의 브런치를 가짐에도 불구하고 파라미터의 수는 VGG16 네트워크의 약 75%이다. 또, 1개의 이미지에 대한 네트워크의 추론 시간은 Titan X GPU에서 약 0.36s정도였다.

<br>

* Input Representation

FV(front view)안에서의 객체 정보(annotations)가 제공되는 KITTI의 경우, (0, 70.4) X (-40, 40) meters의 범위 안의 point cloud를 사용했고, 이미지 평면에 투영했을 때 이미지 경계를 넘어가는 point들은 제거했다. 조감도에서 각 해상도는 0.1m로 나누었기 때문에, 조감도 입력은 704x800의 입력사이즈를 가진다. KITTI는 64-beam Velodyne 레이저 스캐너를 사용하기 때문에 우리는 64x512 map의 FV point를 얻는다. RGB 이미지는 가장 짧은 크기가 500 사이즈가 되도록 업스케일링한다. 

<br>

* Training

네트워크는 end-to-end로 훈련되고, 각 mini-batch마다 1개의 이미지와 128개의 ROI 샘플링하여 ROI의 25%가 positive로 유지되도록 한다. 그리고, iteration = 100K, learning rate = 0.001의 SGD를 사용하여 훈련한다. 그리고 나서 다른 훈련에서는 iteration = 20K, learning rate = 0.0001로 줄인다.

<br>

# 4. Experiments

KITTI object detection benchmark에서 MV3D 네트워크를 평가했다. 그 데이터셋은 트레이닝을 위한 7481개의 이미지와 테스트에 사용될 7581개 이미지로 구성되어 있다. 테스트에서는 2D detection만 평가하기에 training data를 1:1 비율로 training set과 validation set으로 나눈다. 그리고, validation set에서 3D box 평가를 구성했다. KITTI가 충분한 자동차 인스턴스를 제공하기 때문에, 우리는 자동차에만 초점을 맞춰 실험을 진행했다. KITTI 셋팅에 따라 easy, moderate, hard에 대한 어려움 정도를 세가지로 나눠 평가를 진행할 것이다.

* Metrics

metrics에 따라 3D box recall을 사용한 3D object proposals를 평가한다. 두 직사각형의 IOU 중첩을 계산하는 2D box recall와 달리, 여기서는 두 직육면체의 IOU 중첩을 계산한다. 직육면체는 축과 평행하는 것이 아니라 3D boxes의 방향을 나타낸다. 평가에서는 3D IOU threshold를 0.25와 0.5로 설정한다. 최종적인 3D detection 결과를 위해, `3D localization`과 `3D bounding box detection`의 정확성을 측정하기 위해 두 방법을 사용한다. 3D localization에서는 방향성을 가진 조감도 box를 얻기 위해 지면에 3D boxes를 투영한다. 그런 다음 조감도 boxes에 대한 APloc(Average Precision)을 계산한다. 3D bounding box detection을 위해 저자는 full 3D bounding boxes를 평가하기 위한 방법인 AP3D(Average Precision)을 사용한다. 조감도 boxes와 3D boxes 둘 다 방향성이 있기 때문에, 객체 방향은 이 두 가지 지표에서 고려된다. 이미지 평면에 대한 3D boxes를 투영함으로써 2D detection의 수행을 평가한다. AP2D(Average Precision)은 측정 기준으로도 사용된다. KITTI 규칙에 따라 2D boxes는 IOU threshold를 0.7로 설정한다.

<br>

* Baselines

이 작업이 3D Object Detection을 목표로 하고 있기 때문에, LIDAR 기반의 방법들인 VeloFCN, Vote3Deep, Vote3D뿐만 아니라 이미지 기반의 방법들인 3DOP와 Mono3D들과도 비교했다. 공정한 비교를 위해 조감도와 전면도(BV+FV)를 입력으로 사용하는 순수 LIDAR 기반의 모델과 LIDAR과 RGB 데이터(BV+FV+RGB)를 결합한 멀티모달에만 초점을 맞췄다. 3D box 평가에서는 validation set이 제공되는 VeloFCN, 3DOP, Mono3D와 비교했다. 공개적으로 사용할 수 있는 결과가 없는 Vote3Deep과 Voto3D에서는 테스트 셋에서 2D detection만을 비교했다.

<br>

* 3D Proposal Recall

<img src="/_site/assets/img/autodriving/MV3D/fig5.png">

위 그림은 3D box recall을 나타낸 것이다. 300개의 제안을 사용하여 IOU 임계값의 함수로서 recall을 구성한다. 또한, 위 그림에서 볼 수 있듯이, 이 논문의 접근 방식은 모든 threshold에서 3DOP와 Mono3D를 엄청나게 능가했다. 그리고 IOU threshold가 0.25와 0.5아래인 제안들의 갯수에 대한 함수로 3D recall을 볼 수 있다. 이 논문의 방식은 0.25 IOU threshold에서 99.1% recall을, 0.5 IOU threshold에서 91% recall을 얻어냈다. 대조적으로 3DOP에서의 0.5 IOU threshold에 대한 최대 recall은 73.9%를 얻는다고 한다. recall이 크다는 것은 이미지 기반 방법에 비해 LIDAR 기반 접근 방식이 더 좋다는 것을 의미한다. 

<br>

* 3D Localization

3D localization 평가를 위해 0.5와 0.7 IOU threshold을 사용한다. 

<img src="/_site/assets/img/autodriving/MV3D/table1.png">

KITTI validation set에서 APloc를 table 1에서 할 수 있다. 모든 LIDAR 기반 접근 방식은 스테레오 기반 방식인 3DOP 및 monocular 방식인 Mono3D보다 우수했다. LIDAR 기반 접근 방식중에서 BV+FV 방법은 0.5 IOU threshold에서 최대 25% APloc인 VeloFCN을 능가했다. IOU를 0.7로 기준을 잡으면, 이 모델의 성능은 더 높아지고, moderate와 hard 환경에서 최대 45% 더 높은 APloc를 달성할 수 있었다. RGB images와 결합하면, 성능은 더 개선된다. 아래 그림에서 몇 가지 사례의 localization result를 시각화했다.

<img src="/_site/assets/img/autodriving/MV3D/fig6.png">

<br>

* 3D Object Detection

3D 중첩에 대해 설명하자면 0.5 3D IOU threshold와 LIDAR 기반의 방법들에 대해 0.7 IOU threshold에 초점을 맞췄다. 이러한 threshold들은 이미지 기반 방법에는 다소 엄격하기 때문에, 0.25 IOU threshold도 평가에 사용했다. 

<img src="/_site/assets/img/autodriving/MV3D/table2.png">

Table2에서 볼 수 있듯이, 우리의 BV+FV 방법은 moderate setting에서 86.75%를 달성하는 0.5 IOU threshold를 적용할 때 VeloFCN보다 AP3D를 30% 더 높게 얻을 수 있다. 0.7 IOU threshold를 사용할 때, multimodal 접근 방법은 easy data에서 71.29%의 AP3D를 달성했다. moderate 셋팅에서는 0.25 IOU threshold를 사용한 3DOP에 의해 달성될 수 있는 최대 AP3D는 68.82%인 반면, 이 논문의 접근 방식은 0.5 IOU threshold를 사용할 때 89.05% AP3D를 달성했다.

<br>

* Ablation Studies

<img src="/_site/assets/img/autodriving/MV3D/table3.png">

먼저 초기/후기 결합 방식과 이 논문의 deep fusion 방식을 비교해보자. 결합 연산은 초기/후기 융합 방식에서 연결과 함께 인스턴스화된다. table 3에서 볼 수 있듯이 초기/후기 융합 접근은 매우 단순하다. 근사 loss의 사용없이, deep fusion 방법은 초기/후기 결합 방식보다 0.5% 더 향상된 결과를 얻을 수 있다. 근사 loss를 추가하면 1% 더 향상된다.

<img src="/_site/assets/img/autodriving/MV3D/table4.png">

table4에서 볼 수 있듯이, 입력으로 단일 뷰만 사용한다면 조감도, 즉 bird's eye view 특징이 가장 높았고, front view가 가장 낮았다. 두개의 뷰를 섞는다면 단일 뷰보다 모든 뷰들이 더 높게 측정되었다. 이것은 서로 다른 뷰들의 특징들이 성호 보완적이라는 가정이 정당화된다. 세가지 뷰를 모두 결합하면 최상의 결과를 얻을 수 있다. 

<br>

* 2D Object Detection

우리는 마지막으로 KITTI 테스트 셋에서 2D detection 수행을 평가했다. 결과는 다음과 같다.

<img src="/_site/assets/img/autodriving/MV3D/table5.png">

LIDAR 기반의 방법들중에서 BV+FV 방식이 hard 셋팅에서 최근 제안된 Vote3Deep에 비해 AP2D가 14.93% 더 높게 결과가 나왔다. 전체적으로 이미지 기반의 방법들은 보통 2D detection에서 LIDAR 기반의 방법들보다 더 좋다. 이는 이미지 기반의 방법들이 3D boxes를 최적화하는 LIDAR 기반의 방법들과 달리 2D boxes를 직접적으로 최적화하기 때문이다. 하지만 이 논문의 방법이 3D boxes를 최적화함에도 불구하고 최첨단 2D detection 방법들보다 더 경쟁력있는 결과를 얻었다. 

<br>

* Qualitative Results

fig6에서 볼 수 있듯이, 접근 방식은 스테레오 기반의 방법인 3DOP과 LIDAR 기반의 방법인 VeloFCN에 비해 더 정확한 3D 위치, 사이즈, 객체의 방향성을 얻는다. 

<br>

# 5. Conclusion

그래서 최종적으로 로드 뷰에서 3D object detection에 대한 멀티뷰 센서퓨전 모델을 제안한다. 우리의 모델은 LIDAR point cloud와 images 둘 다 강점을 보인다. 3D 제안을 생성하고 특징 추출을 위해 여러 뷰에 투영하여 서로 다른 양식을 정렬한다. region-based fusion network는 방향성을 가진 3D box regression을 하고, 다중 뷰 정보를 깊게 융합한다. 우리의 접근 방식은 KITTI benchmark에서 3D localization과 3D detection의 task에 대한 현존하는 LIDAR 기반과 image 기반 방법들을 엄청나게 능가한다. 3D detection으로부터 얻어진 2D box 결과들은 최첨단 2D detection 방법들과 비교해도 경쟁력이 있다.

<br>

# Reference

* [https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Multi-View_3D_Object_CVPR_2017_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Multi-View_3D_Object_CVPR_2017_paper.pdf)