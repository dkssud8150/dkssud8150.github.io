---
title:    "MMDetection: Open MMLab Detection Toolbox and Benchmark"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2021-12-22 12:00:00 +0800
categories: [Review, Object Detection]
tags: [Object Detection, mmdetection]
toc: True
comments: True
math: true
mermaid: true
---

MMDetection에 대한 논문을 리뷰한 내용입니다. 혼자 공부하기 위해 정리한 내용으로 이해가 안되는 부분들이 많을 거라 생각됩니다. 참고용으로만 봐주세요.

# Abstract
mmdetection은 object detection과 instance segmentation을 다루는 유명하고 다양한 모델을 하나의 toolbox로 구현한 일종의 플랫폼이다. MMDetection은 training and inference 코드뿐만 아니라 200개 이상의 network 모델 weight를 제공한다. 코드는 pytorch를 사용하였다고 한다.

# 1. Introduction
주요 특징으로는
1. Modular design: 모델이 모듈화되어 있어 사용자화를 쉽게 할 수 있다.
2. Support of multiple frameworks out of box: 인기있는 여러 프레임워크를 지원한다.
3. High efficiency: 모든 BBox와 mask 연산은 GPU에서 동작한다. 그리고, training 속도가 다른 detectron, maskrcnn-benchmark, simpleDet 보다 더 빠르다.
4. State of the art: 2018 COCO detection challenge에서 우승한 MMDet team의 코드를 토대로 만들었다.

# 2. Supported Frameworks
MMDetection 안에 구성된 모델로는
- 2.1 Single-stage Methods
    - SSD(2015)
    - RetinaNet(2017): focal loss를 통한 고성능 단일 detector
    - GHM(2019): gradient harmonizing 메커니즘을 통한 단일 detector
    - FCOS(2019): fully convolutional anchor-free detector
    - FSAF(2019): feature selective anchor-free module detector

- 2.2 Two-stage Methods
    - Fast R-CNN(2015)
    - Faster R-CNN(2015)
    - R-FCN(2016): fully convolutional detector
    - Mask R-CNN(2017): instance segmentation method
    - Grid R-CNN(2018) grid를 통한 위치화 메커니즘을 통한 segmentation
    - Mask Scoring R-CNN(2019): IOU를 예측함으로써 Mask R-CNN 개선
    - Double-Head R-CNN(2019)

- 2.3 Multi-stage Methods
    - Cascade R-CNN(2017)
    - Hybrid Task Cascade(2019): multi branch를 통한 가장 높은 성능을 보인 모델

- 2.4 General Modules and Methods
    - Mixed Precision Training(2018): 반정밀 floating point 숫자를 이용해 훈련
    - Soft NMS(2017)
    - OHEM(2016): online hard sampling
    - DCN(2017): 변형 가능한 convolution 과 ROI Pooling
    - DCNv2(2018): 
    - Train from Scratch(2018): imageNet 대신 random initialization
    - ScratchDet(2018): 
    - M2Det(2018): 더 효과적인 feature pyramid를 위한 새로운 feature pyramid
    - GCNet(2019): 
    - Generalized Attention(2019)
    - SyncBN
    - Group Normalization(2018)
    - Weight Standardization(2019)
    - HRNet(2019): 고해상도 이미지를 학습하는 데 집중한 새로운 backbone
    - Guided Anchoring(2019)
    - Libra R-CNN(2019)

# 3. Architecture

## 3.1 Model Representation

**backbone**: 입력 이미지를 특징맵으로 변형시켜주는 부분, ResNet50과 같이 FC Layer(fully connected layer)가 없는 형태를 가짐

**Neck**: backbone과 head를 연결해주는 부분으로, backbone을 통해 생성된 특징맵을 정제하고 재구성해준다. 예를 들면 FPN(Feature Pyramid Network)와 같다.

**DenseHead(AnchorHead/AnchorFreeHead)**: anchorHead와 anchorFreeHead를 포함한 특징맵의 밀도 높은 위치에서 작동하는 요소

**RoIExtractor**: 단일 또는 다중 특징맵으로부터 RoIPooling과 같은 연산을 통해 RoI-wise 특징들을 추출, 예를 들어 특정 수준의 특징 피라미드에 대한 RoI 특징은 singleRoIExtractor이 된다.

**RoIHead(BBoxHead/MaskHead)**: RoI 특징을 입력으로 받고, RoI-wise한 task-specific 예측을 하는 부분이다. bounding box 분류/regression, mask 예측과 같다.

위의 과정들을 그림으로 나타내면 아래와 같다.

<img src="/assets/img/mmdetection/MMDetection_figure1.png" width = "100%">

<br>

## 3.2 Training Pipeline
최소한의 파이프라인만 정의하고 나머지는 hooking 메커니즘을 통해 정의되도록 했다. 후킹 중에서 forward hook은 모델의 파라미터(weights)나 특징맵(feature map)을 신호 전달 중간에 가로채는 기법이라 할 수 있다. backward hook은 gradient exploding을 방지하기 위해 활용한다. 

또한, before_run, before_train_epoch, after_train_epoch 등의 다양한 시점을 정의하고 관찰했다. 

<img src="/assets/img/mmdetection/MMDetection_figure2.png" width = "100%">

저자는 figure 2와 같이 다양한 시점으로 나누고, 사용자가 원하는대로 hook 설정을 변경할 수 있도록 했다. 이 figure2는 학습 과정이고, validation 과정은 평가 후크를 사용해 각 epoch 이후의 성능을 테스트 했기 때문에 보여주지 않았다. 아마 학습 과정과 비슷할 것이다.

<br>

# 4. Benchmarks

## 4.1 Experimental Setting

**dataset**: MS COCO 2017 데이터셋을 사용했다.

**Implementation Details**: 명시하지 않을 경우 default 값을 사용한다.
1. image는 비율 변화 없이 최대 1333x800으로 resize한다.
2. 학습에서는 총 16개의 batch size의 8개의 V100 GPU와 추론에서는 단일 V100 GPU를 사용했다.
3. 학습 과정은 detectron과 같다. 1x, 2x는 12epochs, 24epochs이고, 20e는 cascade 모델에서의 20 epochs를 의미한다.

**Evaluation metrics**: 0.5~0.95의 여러 IoU의 multiple IoU threshold 를 적용할 수 있는 COCO dataset에 대한 표준 평가 metrics를 선택한다. region proposal network(RPN)의 결과는 AR(Average Recall)로 평가했고, detection 결과는 mAP로 평가했다.


## 4.2 Benchmarking Results
**Main results**: SSD, RetinaNet, Faster R-CNN, Mask R-CNN의 method를 사용했고, ResNet-50, ResNet-101, ResNet-101-32x4d와 같이 다양한 backbone을 사용했다. method간의 bbox/mask AP에 대한 추론 속도를 figure 3를 통해 확인할 수 있다. 

**Comparison with other codebases**: Detectron, maskrcnn-benchmark, SimpleDet 등을 비교했다. 결과는 table 2에서 확인가능하다.

<img src="/assets/img/mmdetection/MMDetection_figure3.png" width="50%"><img src="/assets/img/mmdetection/MMDetection_figure4.png" width="50%">

위의 결과처럼 hybrid task cascade 모델이 가장 성능이 좋았다.

<img src="/assets/img/mmdetection/MMDetection_table2.png" width="50%">

<br>

# 5. Extensive Studies

## 5.1 Regression Losses

<img src="/assets/img/mmdetection/MMDetection_table5.png" width = "100%">

여러 가지의 loss를 적용해보았으며, loss weight를 증가시켜가며 성능을 비교했다. 간단하게 Smooth L1 Loss의 loss weight를 증가시켜봄으로써 0.5% 정도 향상되었다. 

<br>

## 5.2 Normalization Layers
GPU 메모리를 작게 하기 위해 detection 학습에서는 batch size가 비교적 작게 만든다. BN은 대체로 CNN에 적용되는데, 이 때, 통계량을 정확하게 추정하려면 큰 batch size를 요구한다. 하지만 object detection의 경우 batch size가 다소 작고, 보통 pretrained backbone을 사용하기에 학습할 때 weight나 BN이 업데이트 되지 않는다. 이 predtrained backbone을 Frozen BN이라고 부른다. 

최근 개발된 SyncBN(Synchronized BN)이나 GN(Group Normalization) 은 좋은 효과를 보여주었다. SyncBN은 다수의 GPU를 사용하여 평균과 분산을 계산하고, GN은 group 별로 특징들에 대한 channel을 나누어 각각의 group별로 평균과 분산을 계산한다.

<br>

그렇다면 각각의 normalization을 비교하고, 성능을 좋게 만들기 위해서는 어떻게 해야 할까?

<img src="/assets/img/mmdetection/MMDetection_table7.png" width = "100%">

같은 method와 같은 ResNet-50-FPN을 사용하고 여기서 BN layer만 교체하면서 비교를 진행했다. 그 결과 BN layer를 업데이트하여도 성능에 큰 변화는 없었고, FPN이나 bbox/mask head를 추가해도 별다른 이점이 없었다. 하지만, bbox head를 2fc를 4conv-1fc로 바꾸고, normalization을 추가하면 1.5% 정도의 성능 향상을 보였다. 또한, 더 많은 conv 층을 추가할 때 더 좋은 성능을 보였다.

<br>

## 5.3 Training scales
이전 연구에서는 대체로 1000x600 이나 1333x800 의 이미지 사이즈를 선호했다. 우리도 마찬가지로 1333x800을 default scale로 사용했다. 하지만 모델의 강인함(robust)을 위해 multi scale을 사용하는 것이 좋을 것이다. 이전에도 이에 대한 방법을 아직 논의하지 않았다. multi scale을 훈련하기 위해 우리는 각 반복마다 무작위로 scale을 정하고, 그것에 대해 입력 이미지를 resize했다.

resize 하는 방법으로는
1. value mode: 스케일셋을 미리 정해놓고 임의로 스케일을 선택한다.
2. range mode: 스케일 범위를 미리 정해놓고 최솟값과 최댓값 사이의 스케일을 임의로 만든다.

<img src="/assets/img/mmdetection/MMDetection_table8.png" width = "100%">

표를 보게 되면 mask rcnn에서 다양한 scale을 적용했다. 1333x[640:800:32]의 표기는 긴 방향은 1333으로 고정하고 짧은 방향을 {640,672,704,736,768,800} 중 무작위로 1개 선택하는데, 위의 경우 640~800 중 32 단위로 1개 정하는 것이다. 이는 value mode에 해당된다. 여기서 만약 1333x[640:800]이라면 range mode가 된다.

표에서 볼 수 있듯 range mode가 value mode보다 아주 조금 더 좋은 성능을 보인다.

<br>

## 5.4 Other Hyperparameter

<img src="/assets/img/mmdetection/MMDetection_table9.png" width = "100%">

간단하게 smoothL1_beta와 allowed_border을 볼 수 있다.

**smoothL1_beta**: 대부분의 detection method는 regression loss로서 smoothL1 loss를 사용한다.

<img src="/assets/img/mmdetection/mmdetection/smoothl1_beta.png" width = "100%">

beta는 L1과 MSELoss에서 threshold를 뜻한다. 1/9를 기본으로 사용한다.

**allowed_border**: RPN에서 특징맵의 각 위치에 미리 정의된 anchor가 생성된다. 이 때 이미지 경계를 넘어가는 anchor은 무시된다. 기본적으로 이를 0으로 잡는다. 하지만 우리는 이것을 무시하지 않을 때 더 좋은 성능을 보인다는 것을 알아냈다. 이를 무한대라 정의한다면, allowed_border을 무한대로 설정했을 때 0.6% 이상의 성능 향상을 보인다.

**neg_pos_ub**: 우리는 positive와 negative anchor라는 새롭게 하이퍼파라미터를 정의했다. 훈련 중 positive anchor이 충분하지 않을 경우 고정된 수의 훈련 샘플을 보장하기 위해 더 많은 음성 표본을 추출한다. negative sample 대비 positive sample의 비율이 더 많아지지 않게 하기 위해 neg_pos_ub를 설정한다. 이를 무한대로 설정하면 앞서 언급한 과정을 거친다. 3 또는 5로 설정한다는 것은 positive sample의 최대 3배 또는 5배까지는 negative sample을 허용하겠다는 뜻이다.





# Reference
- thesis: MMDetection:Open MMLab Detection Toolbox and Benchmark, https://arxiv.org/pdf/1906.07155.pdf
- github URL: https://github.com/open-mmlab/mmdetection
- Tutorial: <a href="https://colab.research.google.com/github/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- 참고 블로그:
    - https://wordbe.tistory.com/entry/MMDetection-%EB%85%BC%EB%AC%B8-%EC%A0%95%EB%A6%AC-%EB%B0%8F-%EB%AA%A8%EB%8D%B8-%EA%B5%AC%ED%98%84