---
title:    "2021 Ego-Vision 손동작 인식 경진대회"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2021-10-07 12:00:00 +0800
categories: [Review, DACON]
tags: [DACON,keypoint, classification]
toc: True
comments: True
math: true
mermaid: true
---

일단 제출은 하지 못했다.

최소한의 기능만 가지고 제출을 목표로 코드를 구현했다.

아직 customdataset을 만드는 방법을 잘 몰라 고생을 했다. 정답 label과 폴더의 개수가 맞지 않고, 내가 원하는 image와 label을 불러와 돌릴 수 없으니 제약되는 것이 많았다. keypoint를 추출하여 ±100 하여 image를 crop하는 방법도 있었으나, customdataset을 만들지 못하여 사용하지 않았다.

그래서 train 폴더를 retrain폴더로 answer label 별로 재분류하여 model에 집어넣었다. 

많은 augmentation이 있지만, 일주일밖에 시간이 없었고, 제출만을 목표로 하였기 때문에 적용하지 않았다.

valid도 추가하여 loss나 accuracy를 보며 수정해야 했지만, 이 또한 수행하지 못했다. 

epoch 자체가 train을 통해 loss를 줄여나가는 것이다. 따라서 test에 epoch을 사용하는 것이 아니라, 모든 test image를 다 돌려보기 위해서는 다른 함수가 필요하다.

pytorch책에서 배운 evaluate 함수는 val을 위한 함수다. 따라서 test image에 대한 모델과 이미지는 따로 만들어야 한다. 이미지는 모든 이미지를 불러오는 코드를 구현해야 할 것이다.

train은 val을 평가하는 것이기도 하다. 따라서 model parameter을 저장해야 한다.

<br>

사실 train,val,test 함수는 별 차이 없다. 하지만 preprocessing을 어떻게 하냐를 많이 구현하고, train 후 plot하여 loss를 크게 변화시키는 데이터가 어떤 것인지 알아보는게 중요한 듯하다.

아니면,, 여러가지 augmentation이나 kfold, ensemble 등의 방법을 통해 성능을 끌어올리는듯하다.

<br>

5등껄보니 꽤 비슷했다. customdataset만 만들줄 알았다면 꽤 등수가 올랐을 것 같다.

# 공부해야 할 것들

따라서 이 대회를 통해 깨달은 나의 부족한 점이나 보완할 점은
1. Custom Dataset 만드는 방법 공부
2. validation 추가하고, model을 save하고 plot까지
3. augmentation 공부
  
  - 일반적인 transfrom 말고도 많으니 다양한 aumentation을 적용하는 방법 공부 
    - Rule-based Approach : Public-0.00670, Private score-0.00578 (아래에서 자세히 분석하겠지만 큰 효과를 봤습니다.)
    - Flip Augmentation
    - Random Margin Crop Image
    - Random Affine & Random Perspective Augmentation
    - OneCycleLR Scheduler
    - WarmUpLR Scheduler

4.  Kfold 적용해보기
  
  - validation Kfold 등등

5. Ensembles 적용하는 방법 공부
6. seed 설정 방법
7. batch normalization 적용하는 방법
8. overfitting 막는 방법
- weight_decay 는 regularization 하는 매개변수이다. 따라서 overfitting을 막기 위한 방법 중 하나로 weight_decay를 사용

<br>

<br>

먼저 egovision 코드로 한 후, 교통 수신호 동작 인식 ai경진대회도 참고하여 코드 공부하기

<br>

<br>

# URL
- [https://github.com/dkssud8150/dacon-egovision/blob/main/egovision.ipynb](https://github.com/dkssud8150/dacon-egovision/blob/main/egovision.ipynb)
- [https://dacon.io/competitions/official/235805/codeshare/3362?page=1&dtype=recent](https://dacon.io/competitions/official/235805/codeshare/3362?page=1&dtype=recent)
- [다른 대회 참고 코드](https://dacon.io/competitions/official/235842/codeshare/3587?page=1&dtype=recent)