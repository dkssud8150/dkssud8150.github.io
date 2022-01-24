---
title:    "병변 검출 AI 경진대회"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2021-12-20 12:00:00 +0800
categories: [Review, DACON]
tags: [DACON,Object Detection]
toc: True
comments: True
math: true
mermaid: true
---

## 결과
114/250 (상위 46%)


## 코드
[https://colab.research.google.com/drive/10vwjU62GcwKWN9ehU-01lORoJAP8WOj8#scrollTo=OGkiaInaZSLc](https://colab.research.google.com/drive/10vwjU62GcwKWN9ehU-01lORoJAP8WOj8#scrollTo=OGkiaInaZSLc)


# 복기
custom dataset을 사용하고, 나머지는 baseline을 참고하여 구축했다. 앙상블을 하지 않고, fold, TTA를 하지 않은 것, model을 yolov5와 같은 성능이 좋은 것이 있음에도 다른 것을 고집한 것, baseline을 딴 후, yolov5에 대한 코드 공유가 있었는데, 따라했어야 했다. 데이터가 너무 많아 오래걸린다면 model을 일단 학습시키기 위해 학습 데이터를 줄이거나 모델을 바꾸거나 하는 것이 좋지 않았나 싶다.


## 학습 시간
데이터가 너무 많아서 학습을 하고 파일로 출력하는데 너무 오랜 시간이 걸렸다. 이를 해결하는 방법을 공부해봐야 할 것 같다.

방법으로 parameter이나 weight 등의 시간이 오래걸리는 요소들을 파일로 저장/불러오기하면 시간이 단축되지 않을까
여러 번 돌려야 할 필요도 없고, 런타임이 끊기더라도 다시 불러오기만 하면 되니까


## GPU 공간
gpu가 부족할 경우 해결 방법으로는 batch size를 줄이면 된다.


## CUSTOM

- custom dataset
- custom transform

<br>

### Custom Dataset


### Custom Transform 


### Custom Model
대회때는 pretrained model을 사용하더라도 자기만의 model을 만들 줄 알면서 사용하는 것과 몰라서 사용하는 것에는 큰 차이가 있다. 따라서 **custom model**을 공부한다.


### Shell
코드가 많을 경우 파일을 나눠서 사용해야 할 것이다. detect.py model.py 등과 같은 **기능별 파일을 만들어 수행하는 방법을 공부한다.**


<br>

<br>

## Model
상위 랭커분들이 yolov5 + MMdetection을 사용했기에 이를 사용하여 detection 하는 방법을 공부한다.


## 앙상블
약 10개의 모델을 앙상블하는 모습을 볼 수 있으므로 앙상블하는 방법을 공부한다. **BBox를 앙상블**하는 방법은 [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)를 사용한다.
이 때, 모델 학습 + TTA(test time augmentation) 도 함께 앙상블한다.


## COCO dataset Tranfer
대회 데이터 format에서 COCO 데이터 format으로 변형해야 한다. COCO라 함은 id, image_id, area, bbox, iscrowd 이와 같은 것들이다.


## fold split
Kfold를 사용하여 적용해봐야 한다. 따라서 Kfold를 사용하여 모델을 구축하는 방법을 공부한다.



## MMDetection / yolov5
MMdetection 와 yolov5를 실제로 적용하는 방법을 공부해야 한다. yolov5를 공부했지만 다시해야 할 것이고, MMDetection은 이제 공부를 시작하자.