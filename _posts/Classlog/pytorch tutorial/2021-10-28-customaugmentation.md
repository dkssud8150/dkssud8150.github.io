---
title:    "Custom augmentation class 정의해보기"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2021-10-28 12:00:00 +0800
categories: [Classlog,Pytorch Tutorial]
tags: [pytorch tutorial,custom augmentation]
toc: True
comments: True
math: true
mermaid: true
#image:
#  src: /commons/devices-mockup.png
#  width: 800
#  height: 500
---

# Data Augmentation

data augmentation 방법은 크게 "기하학 변환"과 "태스크 기반" 확장으로 나뉜다.

<br>

기하적 변환은 크기조절, 반전, 자르기, 회전, 이동, 윈도우 분할 등이 있다.

태스크 기반 확장은 데이터를 합성 시 분류에 대한 척도를 고려하여 학습을 수행한다. 물체 인식에 대해 자르기(crop)와 반전(filp), 자르기와 색 조절은 학습 데이터 확장에, 크기 변환과 다각도관점 변환 등이 있다.

<br>

이것들의 함수를 직접 정의해보고자 한다.

## 기본 image와 label 지정하기

```python
import wget
import torch
from PIL import Image
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

!wget https://i.stack.imgur.com/LW5Fv.png -O dog.jpg
image = Image.open("dog.jpg")
image = np.array(image)

xmin, xmax = (1,2)

red_color = (255,0,0)

label = (1000,450)

drow = cv2.rectangle(image, (label[0]-550,label[1]-400),(label[0],label[1]),red_color,3)

image = Image.from_array(image)

plt.imshow(drow)
plt.show()
```

<br>

## 1. Resize

이미지를 resize 해줌과 동시에 label도 갱신해줘야 한다.

```python
def resize_img_label(image, label=(0.,0.), target_size = (256,256)):
  w_orig, h_orig = image.size
  w_target, h_target = target_size
  cx, cy = label
  image_new = TF.resize(image, target_size)
  label_new = cx/w_orig*w_target, cy/h_orig*h_target
  return image_new, label_new

image_new, label_new = resize_img_label(image,label)

plt.imshow(image_new)
plt.show()
```

<br>

## 2. horizontally Flip

```python
def random_hflip(image, label):
  w, h = image.size
  x, y = label

  image = TF.hflip(image)
  label = w-x, y
  return image, label

image_new, label_new = random_hflip(image,label)

plt.imshow(image_new)
plt.show()
```

<br>

## 3. Vertically Flip

```python
def random_vflip(image, label):
  w, h = image.size
  x, y = label

  image = TF.vflip(image)
  label = x, w-y
  return image, label

image_new, label_new = random_vflip(image,label)

plt.imshow(image_new)
plt.show()
```

<br>

## 4. Shift or translate image

```python
def random_shift(image, label, max_translate=(0.2,0.2)):
  w,h = image.size
  max_t_w, max_t_h = max_translate
  cx, cy = label
  trans_coef = np.random.rand() * 2 - 1
  w_t = int(trans_coef * max_t_w * w)
  h_t = int(trans_coef * max_t_h * h)
  image = TF.affine(image, translate=(w_t, h_t), shear=0, angle=0, scale=1)
  label = cx + w_t, cy + h_t
  return image, label

image_new, label_new = random_shift(image,label)

plt.imshow(image_new)
plt.show() 
```

# Reference
- [https://deep-learning-study.tistory.com/494](https://deep-learning-study.tistory.com/494)