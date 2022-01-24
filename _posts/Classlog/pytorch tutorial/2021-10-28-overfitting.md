---
title:    "overfitting 방지(prevent)하는 방법"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2021-10-28 12:00:00 +0800
categories: [Classlog,Pytorch Tutorial]
tags: [pytorch tutorial,overfitting]
toc: True
comments: True
math: true
mermaid: true
#image:
#  src: /commons/devices-mockup.png
#  width: 800
#  height: 500
---

# overfitting

모델이 과적합되면 훈련 데이터에 대한 정확도는 높을지라도, 새로운 데이터 즉 검증 데이터나 테스트 데이터에 대해서는 제대로 동작하지 않는다. 이는 모델이 학습 데이터를 불필요할 정도로 과하게 암기하여 훈련 데이터에 포함된 노이즈까지 학습한 상태라고 해석할 수 있다. 

overfitting을 막는 방법에는
1. 데이터 양 늘리기

  데이터 양이 적을 경우 해당 데이터의 특정 패턴이나 노이즈까지 쉽게 암기하므로 과적합이 잘 발생한다. 데이터가 적을 경우 data augmentation을 통해 양을 늘릴 수 있다.

2. 모델 복잡도 줄이기

  은닉층의 수나 매개변수의 수 등을 줄이는 것이다.

3. 가중치 규제(regularization) 적용하기

  복잡한 모델을 좀 더 간단하게 하는 방법으로는 가중치 규제(regularization)이 있다.

<br>

## Regularization

weight_decay를 설정하는 것이 regularization을 설정하는 것이다.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

<br>

## weightedrandomsampler

```python
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
```






# Reference
- [https://wikidocs.net/61374](https://wikidocs.net/61374)
- [https://ichi.pro/ko/pytorch-basics-saempeulling-saempeulleo-46244616519466](https://ichi.pro/ko/pytorch-basics-saempeulling-saempeulleo-46244616519466)