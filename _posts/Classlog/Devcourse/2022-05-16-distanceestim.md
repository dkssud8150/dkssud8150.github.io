---
title:    "[데브코스] 14주차 - DeepLearning Geometrical Distance Estimation"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-05-16 14:20:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, deeplearning]
toc: true
comments: true
math: true
---

<br>

[이전 글](https://dkssud8150.github.io/posts/perception/)에서 computer vision에서 활용하는 기하학적 방법으로 객체의 위치를 추정하는 방법을 배웠다.

<br>

# Geometrical Distance Estimation

이처럼 기하학적 방법으로 객체의 위치를 추정하는 방법들은 많이 존재한다. 그 첫번째 방법이 카메라의 투영(calibration)을 활용한 방법이다.

<br>

## Geometrical Method

카메라와 대상 객체의 기하학적 조건을 활용한 방법으로 카메라의 설치 노이와 대상 객체가 3차원 공간에 존재할 때 Extrinsic calibration 정보를 활용한 방법이다.

<br>

<img src="/assets/img/dev/week12/day5/figure2.png">

이는 camera와 차량의 위치 관계에 대한 그림이다. 카메라를 A에 장착하고, B차량을 촬영하면 B차량의 타이어 지점은 이미지에서 y1에 표현된다.

그러나 이 방법은 바닥이 평면을 이루어야 하고, 바닥과 광학축이 평행해야 한다는 조건이 존재했다.

삼각 비례식을 활용하여 객체의 실제 거리를 추정할 수 있다.

$ object\:distance = \cfrac{focal\:length * real\:world's\:height\:of\:object}{image's\:height\:of\:object} $

<br>

이를 3차원 좌표로 표현하면 다음과 같다.

<img src="/assets/img/dev/week14/projection.png">

전방을 z, 오른쪽을 y, 아래를 x인 카메라 좌표계로 되어 있고, 3차원 공간에 존재하는 객체 P에 대해 이미지에 투영한다. 식은 위의 식과 거의 동일하다.

$ y : f = h : Z , Z = \cfrac{f * h}{y} $

<br>

이를 구하기 위해서는 ZX plane 뿐만이 아닌, ZY plane을 사용해야 한다. 또한 객체가 지면에 붙어 있지 않는다면 추정을 하기 어렵다. 예를 들어 신호등을 인식했지만, 기둥은 추론이 되지 않았고, 신호등만 인식이 되어 있다면 지면에서 일정 높이 떨어져 있는 객체의 bbox만을 사용하여 거리를 추정해야 한다.

<br>

## Field of View

카메라의 대상 객체의 특성을 활용한 방법으로 카메라의 고유한 특성(intrinsic , sensor, lens의 속성)과 대상 객체의 실제 크기 정보를 활용하는 방법이다. 이는 extrinsic calibration 정보에 대해 상대적으로 의존도가 낮다.

FOV(Field of View)를 사용하면 ZY plane에 대한 거리 정보와 지면에서 일정 높이 떨어져 있는 객체에 대한 거리를 추정할 수 있다.

<img src="/assets/img/dev/week14/fov.png">

카메라가 좌표계 원점에 존재하고, 사각형이 이미지일 때, 가로를 width, 세로를 height라 할 수 있다. 이 때, width와 height에 대해 $ FOV_v $ 와 $ FOV_H $ 가 존재한다. v는 vertical, h는 horizontal이고, FOV는 $ ^{\circ} $ 단위로 표기한다. 이미지와 원점간의 거리를 `f`라 할 수 있다. 

FOV가 커질수록 실제 공간을 더 넓게 투영하여 이미지로 표현할 수 있을 것이고, FOV가 작을수록 실제 공간을 더 좁게 투영하여 이미지를 표현할 수 있다.

<br>

<img src="/assets/img/dev/week14/fov2d.png">

간단하게 2차원으로 보자. 

실제 객체는 `H`의 높이를 가지고, 객체와 렌즈 사이의 거리를 $ D_o $, 이미지 안에서의 객체 높이를 `h`, 이미지와 렌즈 사이의 거리를 $ D_f $ 라 한다. F는 focal length이다. 그리고 $ \alpha $는 이미지 안에서 객체의 bbox에 대한 FOV_H값이다. 즉 bbox 위와 아래가 이루는 각도이다.

이 때, 수식 $ \cfrac{1}{D_o} + \cfrac{1}{D_f} = \cfrac{1}{F} $ 에 의해 $ D_o = \cfrac{D_f * F}{D_f - F} $ 로 표현된다. 그리고, $ tan(\cfrac{\alpha}{2}) = \cfrac{h}{2 D_f} $ 이므로, D_o 는 다음과 같다.

$$ D_o = \cfrac{F * \cfrac{h}{2 tan\cfrac{\alpha}{2}}}{\cfrac{h}{2 tan\cfrac{\alpha}{2}} - F} $$

<br>

이 때, a는 FOV로 계산이 가능하고, 다른 변수도 모두 알고 있는 값이므로 이미지 내에서의 object의 크기와 위치를 알면 D_o를 구할 수 있다.

<br>

더 자세한 설명을 위해 가상의 카메라의 intrinsic, FOV를 다음과 같이 정의하자.

<img src="/assets/img/dev/week14/fov_ex.png">

- image size : 1280px, 720 px
- fx = 1000, fy = 1000
- cx = 640, cy = 360
- $ FOV_H = 80 ^{\circ} $, $ FOV_V = 40 ^{\circ} $

이미지 내 표지판의 위치에 대한 값
- bbox size : 40px, 40px
- center about bbox bottom to bbox bottom : 940px
- image center to bbox left bottom : 640px

바운딩 박스의 좌하단과 height 방향으로의 직선 사이의 각도를 $ \theta $ 라 한다. 그리고 실제 표지판의 크기는 반지름이 0.5m이다. 어차피 대부분의 경우 사각형을 기준으로 측정하므로 w = 1m, h = 1m 로 할 수 있다.

$ FOV_{H(640)} = 0 ^{\circ} $ 인 이유는 이미지를 기준으로 중앙이 0, 왼쪽 방향이 (-), 오른쪽 방향이 (+)이다. 따라서 $ FOV_{H} $ 의 범위는 -40 ~ 40 이다.

<br>

이미지의 중점을 기준으로 객체의 중점과의 이루는 방위각이 $ \theta $ 이고, 비례식 $ 640 : 300 = 40 : \theta $ 을 세워서 계산한다. 

$ \theta = \cfrac{\triangle x}{640} * 40.0 ^{\circ} = \cfrac{300}{640} * 40.0 = 18.75 ^{\circ} $ 

$ \theta $ 를 구했다면, 표지판과의 거리를 구할 수 있다.

$ D_o = \cfrac{F * \cfrac{h}{2 tan\cfrac{\alpha}{2}}}{\cfrac{h}{2 tan\cfrac{\alpha}{2}} - F} = \cfrac{1000 * \cfrac{40}{2 tan(18.75 ^{\circ})}}{\cfrac{40}{2 tan(18.75 ^{\circ} )} - 1000} $

그리고 종방향과 횡방향의 분리를 위해 dx, dy로 분리한다.

$$ d_x = d * cos(\theta) , d_y = d * sin(\theta) $$

<br>

이때까지는 FOV_H 를 사용하여 구했지만, FOV_V를 사용하여 구할 수도 있다.

<img src="/assets/img/dev/week14/distance.png">

FOV_H에 대한 좌표와 값들은 빨간색에 해당되고, FOV_V는 파란색에 해당한다.

<br>

<br>

> - **평면의 특성을 활용한 방법**
>
><img src="/assets/img/dev/week14/projection2.png">
><img src="/assets/img/dev/week14/fov_ex.png">

>여기서 bbox 각각의 좌표를 a,b,c,d 라 표현했을 때, 각각의 좌표는 다른 값을 가지고 있겠지만, 다른 평면으로 바라봤을 때는 직선으로 표현될 것이고, 그것이 첫번째 이미지가 된다.

>그렇다면 bbox 각각의 좌표는 다 동일한 distance, d과 높이 R을 가지고 있으므로 다음과 같은 비례식을 사용할 수 있다.

>$$ r_a : f = d : R_a $$

>$$ r_b : f = d : R_b $$

>이를 정리하여 `d`에 대해 정리하면 $ d = \cfrac{(r_b - r_a) * (R_b - R_a) }{f} = \cfrac{40px * 1m}{1000px} $ 로 정리될 수 있다.

>그러나 이 방법은 extrinsic에 의존적이라 잘 설정해줘야 한다.

<br>

<br>

위의 두 방법(geometric method, FOV) 는 복잡한 기하학적 변환이나 수식이 들어가 이해하기 어려운 단점이 있다. 

## Perspective Projection Method

또 다른 방법으로 평면 변환(plane transform)이 있다. 이 방법에는 객체가 지정하고자 하는 평면과 관련이 있어야 하는 단점이 존재한다.

<br>

### Homography

perspective projection method의 대표적인 방법으로 homography가 있다. 

<br>

이를 위해서는 평면과 평면 변환에 대해 알고 있어야 한다. 투영(projection)이란, 3차원 공간에 존재하는 어떤 대상을 2차원 이미지 공간(평면)에 투영하는 과정이다. 이미지를 다룬다면 언제나 1개 이상의 평면을 사용하고 있다. 이 평면을 image coordinate에서 정의되는 image plane이라고 한다.

<img src="/assets\img\dev\week14\image_plane.png">

<br>

3차원 공간에 존재하는 어떤 점이 2차원 평면에 투영한다고 할 때,  3차원 공간은 무한 개의 평면을 가지고 있다. 그래서 간단하게 보기 위해 하나의 평면으로 표시하게 되면, 2차원 투영 좌표는 (x,y)에서 (x,y,1)로 , 3차원 평면의 좌표는 (X,Y,Z)에서 (X,Y,1)이 된다. 

이처럼 3차원 공간 상에 놓여진 하나의 2차원 평면을 표현할 때 homogeneous 좌표계를 사용한다. homogeneous 좌표계에서는 2차원 점이 3차원으로 투영이 된다.

<br>

<img src="/assets\img\dev\week12\day4\projectionmatrix.png">

그렇다면 카메라 좌표계를 기준으로 normalized image coordinate 좌표계에서의 점과 image coordinate에서의 점을 3차원으로 변환할 수 있다.

카메라 좌표계에서 Z는 전방을 가리킨다. 이 때 normalized image coordinate는 원점으로부터 `1`만큼 떨어져 있고, image coordinate는 `f`, 실제 좌표계에서의 점은 `Z`만큼 떨어져 있다. 그래서 각각의 점들을 변환한다.

- normalized image coordinate : $ (u_n, v_n, 1) $
- image coordinate : $ (f_x, f_y, f) $
- world coordinate : $ (X_{w_o}, Y_{w_o}, Z_{w_o}) $

homogeneous 좌표계에서는 같은 투영선(projection ray) 상에 있는 좌표들은 다 동일한 점이라고 판단한다. 결과적으로 homogeneous 좌표계는 3차원 공간을 2차원 공간으로 투영하면 무한개의 점으로 표현이 가능해진다. 그러나 어떤 이미지 공간에 투영하는가에 따라 달라지는 것이지만, 그 대상은 동일하다. 2차원을 3차원으로 변환하는 것을 `Inverse-Projection`이라 부른다.

<br>

이미지 평면에서 정규 이미지 평면으로 변환하는 것을 평면 변환(projective transform)의 한 종류로 볼 수 있다. 평면의 좌표계 기준은 달라지지 않았지만, 차선과 같이 실제 공간에서는 평행하는 두 선이 이미지 상에서는 만나지도록 변환이 된다.

<br>

projective geometry의 성질을 이용한 이미지 평면을 다른 평면으로 변환하는 것도 가능하다. 이를 활용한 대표적인 예가 BEV(bird eye view) 변환이다. 같은 높이에서 바라보는 방향과 새처럼 위에서 바라보는 방향은 다르다. 위에서 아래로 바라보는 시점을 BEV라 한다. 

원래의 이미지 상에서는 차선들이 소실점에서 만나게 된다. 그러나 이를 BEV로 변환하게 되면 차선들이 다시 평행하게 보인다. 원래 카메라 이미지 상에서 객체의 bbox, class를 추출하고 이를 bird`s eye view에 투영시켜서 회피를 할지 말지를 결정한다.

<br>

<img src="/assets/img/dev/week14/figure1.png">

여기서 ground plane은 실제 지면을 의미하고 이는 `m`단위를 사용하고 있다. 카메라로 보는 이미지 a를 c로 바꾸는 작업을 수행하는 것을 perspective transformation이라 하고, 원래의 이미지 a에서는 pixel단위이고, BEV인 c에서는 `m`단위를 사용한다. BEV도 이미지이므로 pixel 단위라고 생각할 수 있지만 이는 틀렸다.

2차원 좌표계에서 데이터를 부르는 단위를 `Pixel`이라 하고, 3차원 좌표계에서 데이터를 부르는 단위를 `Voxel`이라 하는데, 픽셀이 이미지 해상도를 의미하지는 않는다. 즉 이미지를 부르는 방법이 픽셀이지 모든 픽셀이 이미지를 의미하는 것은 아니다.

<br>

<br>

## Geometrical Distance Estimation code

본질적으로 원근 변환(perspective transform)과 투영 변환(projective transform, homography)은 동일하다. 원근 변환은 투영 변환이지만, 투영 변환은 원근 변환일 필요는 없다. 개념과 변환의 결과는 동일하지만 투영 변환이 조금 더 상위 개념이다. 대체로 같은 것으로 판단한다.

openCV에서 제공하는 homography와 perspective transformation에 대해 공부하고자 한다. 이는 geometrical distance estimation을 수행할 때 반드시 필요한 intrinsic matrix와 extrinsic matrix에 대한 개념과 공부를 하기에 적절하다.

### Homography

tutorial : https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html

먼저 homography를 공부하고자 한다. openCV에서는 평면 변환을 위한 함수를 제공한다. 

- getPerspectiveTransform()
- findHomography()

<img src="/assets/img/dev/week14/getperspectivetransform.png" width="55%">
<img src="/assets/img/dev/week14/findhomography.png" width="40%">

이 두 함수는 동일하지만, 입력 인자가 조금 다르다. 전자의 경우 4개의 점들을 입력으로 받지만, 후자의 경우 4개 이상의 점들을 입력으로 받는다. 4개의 점을 가지고 있다면 전자를 사용하고, 4개 이상의 점들을 가지고 있어서 좀 더 정확한 변환을 하고자 하거나, 점들 사이에 존재하는 오차를 알아서 제거해주길 원하는 경우에는 후자를 사용하는 것이 좋다. 그리고 findhomography의 경우 입력 포인트들을 통해 변환 행렬을 계산할 때 오차를 발생시키는 값을 제거하는 방법(RANSAC)이 포함되어 있다.

<br>

Geometrical Method에서는 카메라와 지면이 서로 평행하다는 것을 가정했지만, 실제로는 완벽하게 평행할 수 없다. 그래서 카메라의 자세를 추정하는 별도의 알고리즘을 또 사용해야 한다. 즉 이미지를 취득한 후 소실점/소실선을 이용한 카메라 자세를 추정한 후 객체와의 거리를 추정한다. 그러나 이 평면 변환의 경우 이미지와 그 타겟이 되는 가상의 평면에 대한 변환 과정을 다루기 때문에 **카메라와 지면이 서로 평행하지 않아도 된다**. 즉 어떤 카메라의 자세든 변환이 가능하다.

<br>

<br>

이전에 배웠던 camera calibration 코드를 먼저 보자.

이 코드에서는 캘리브래이션 패턴에 존재하는 특징(코너)를 검출한다. 가장 중요한 것은 imgpoints와 objpoints의 pair 데이터를 만드는 것이 중요하다. 그 이유는 추후에 distance estimation을 할 때는 image가 아닌 pair 데이터를 사용하기 때문이다.

```python
import cv2
import glob
import numpy as np
import time
import sys

DISPLAY_IMAGE = False

# Get Image Path List
image_path_list = glob.glob("images/*.jpg")

# Chessboard Config
BOARD_WIDTH = 9
BOARD_HEIGHT = 6
SQUARE_SIZE = 0.025

# Calibration Config
flags = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
    + cv2.CALIB_CB_FAST_CHECK
)

pattern_size = (BOARD_WIDTH, BOARD_HEIGHT)
counter = 0

image_points = list()

''' =========== calibration pattern's corner detection =========='''
for image_path in image_path_list:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # OpneCV Color Space -> BGR
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(image_gray, pattern_size, flags)
    if ret == True:
        if DISPLAY_IMAGE:
            image_draw = cv2.drawChessboardCorners(image, pattern_size, corners, ret)
            for corner in corners:
                counter_text = str(counter)
                point = (int(corner[0][0]), int(corner[0][1]))
                cv2.putText(image_draw, counter_text, point, 2, 0.5, (0, 0, 255), 1)
                counter += 1

            counter = 0
            cv2.imshow("img", image_draw)
            cv2.waitKey(0)

        image_points.append(corners)

object_points = list()
object_points = np.asarray(object_points, dtype=np.float32) # change numpy array
```

<br>

<br>

- **camera calibration**

1개의 이미지를 통해 intrinsic, extrinsic 정보를 얻어오는 코드이다.

```python
''' =========== camera calibration =========='''
tmp_image = cv2.imread("images/left01.jpg", cv2.IMREAD_ANYCOLOR)
image_shape = np.shape(tmp_image)

image_height = image_shape[0]
image_width = image_shape[1]
image_size = (image_width, image_height)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
```

- intrinsic
  - ret : RMS(ROOT Mean Square) 오차 =\> extrinsic 정보와 objpoints를 가지고 이미지에 재투영하여 원래의 점과 비교해서 오차를 계산한다.
    - 계산했을 때 많이 튀는 노이즈 값을 필터링하는 것도 중요
  - mtx : intrinsic matrix
  - dist : distortion cofficients
- extrinsic
  - 각 이미지에서 설정한 objpoints의 원점에 대한 실제 카메라의 extrinsic calibration 정보이다.
  - rvecs : 각 이미지에 대한 camera coordinate에서의 rotation
  - tvecs : 각 이미지에 대한 camera coordinate에서의 translation
  

<br>

<br>

- **get homography matrix**

```python
''' =========== homography matrix =========='''
# Part 2. Find homography matrix
# Step A. Select images
img1 = cv.imread(os.path.join("images", "left01.jpg"))
ret, corners1 = cv.findChessboardCorners(img1, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT))
corners1 = corners1.reshape(-1, 2)

img2 = cv.imread(os.path.join("images", "right13.jpg"))
ret, corners2 = cv.findChessboardCorners(img2, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT))
corners2 = corners2.reshape(-1, 2)


## Homography
# src -> dst point transformation -> find matrix
# Step B. Find homography
homography, status = cv.findHomography(corners1, corners2, cv.RANSAC)

# Step C. Display Result
img_draw_matches = cv.hconcat([img1, img2])
for i in range(len(corners1)):
    pt1 = np.array([corners1[i][0], corners1[i][1], 1])
    pt1 = pt1.reshape(3, 1)
    pt2 = np.dot(homography, pt1)
    pt2 = pt2 / pt2[2]
    end = (int(img1.shape[1] + pt2[0]), int(pt2[1]))
    start = (int(pt1[0][0]), int(pt1[1][0]))

    color = list(np.random.choice(range(256), size=3))
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv.line(img_draw_matches, start, end, tuple(color), 5)

cv.namedWindow("Draw Matches", cv.WINDOW_NORMAL)
cv.imshow("Draw Matches", img_draw_matches)
cv.imwrite("draw.png", img_draw_matches)
cv.waitKey(0)
```

- step A
  - img1은 원본 이미지, img2는 대상이 되는 이미지
  - findChessboardCorners를 사용하여 transform의 pair 데이터를 결정
  - 추후 변환에 사용되는 데이터는 이미지가 아니라 pair 데이터이다.
  - pair 포인트들을 자기 마음대로 찍어도 상관은 없으나 편리함을 위해 함수를 사용한 것이다.
- step B
  - 이미지가 아닌 N개의 corners1, corners2 를 사용
  - cv.RANSAC은 여러 개의 pair 데이터를 사용하면서 오류가 있는 데이터는 제외하는 알고리즘
- step C
  - 결과 확인
  - cv.line를 사용하여 pair 데이터가 어떻게 변환이 되었는지 확인
  - cv.warpPerspective 를 사용하여 homography matrix로 이미지 자체를 변환


<br>

<br>

#### convert img plane to ground plane

이때까지는 ground plane를 image plane으로 변환했다. 이제는 반대로 image plane을 ground plane으로 변환하는 과정을 수행한다.
- homography matrix로 카메라의 extrinsic 정보를 추출할 수 있다.
- 카메라의 POSE를 그리는 것이 중요 : drawFrameAxes()
  - 이 좌표가 obj point의 0index와 동일한 좌표를 가져야 한다.

<br>

- **homography distance estimation**

image point와 homography를 통해 거리를 추정하는 코드

```python
# Step A. homography distance estimation
import cv2
import json
import os
import numpy as np

import matplotlib.pyplot as plt

import calibration_parser


if __name__ == "__main__":
    calibration_json_filepath = os.path.join("image", "cologne_000065_000019_camera.json")
    camera_matrix = calibration_parser.read_json_file(calibration_json_filepath)
    image = cv2.imread(os.path.join("image", "cologne_000065_000019_leftImg8bit.png"), cv2.IMREAD_ANYCOLOR)

    # extrinsic -> homography src, dst
    # prior dst -> image coordinate
    # present dst -> vehicle coordinate (=camera coordinate)

    # world's lane value
    # lane (inner) width -> 2.5m, lane width -> 0.25m
    # lane length -> 2.0m
    # lane interval -> 2.0m

    """
    Extrinsic Calibration for Ground Plane
    z가 전방 , x,y,z
    [0, 1]
    464, 833 -> 0.0, 0.0, 0.0
    1639, 833 -> 0.0, 3.0, 0.0      lane width + 2 * lane inner width

    [2, 3]
    638, 709 -> 0.0, 0.0, 2.0       lane length
    1467, 709 -> 0.0, 3.0, 2.0

    [4, 5]
    742, 643 -> 0.0, 0.0, 4.0       
    1361, 643 -> 0.0, 3.0, 4.0

    [6, 7]
    797, 605 -> 0.0, 0.0, 6.0
    1310, 605 -> 0.0, 3.0, 6.0
    """


    image_points = np.array([
        [464, 833],
        [1639, 833],
        [638, 709],
        [1467, 709],
        [742, 643],
        [1361, 643],
        [797, 605],
        [1310, 605]
    ], dtype=np.float32)

    # X Y Z, X -> down, Z -> forward, Y -> Right
    # 실측이 중요하다.
    object_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 3.7, 0.0],
        [0.0, 0.0, 6.0],
        [0.0, 3.7, 6.0],
        [0.0, 0.0, 12.0],
        [0.0, 3.6, 12.0],
        [0.0, 0.0, 18.0],
        [0.0, 3.7, 18.0]
    ], dtype=np.float32) # z,y,x 이고, 각각의 0.0 자리에 z는 extrinsic에 있는 z를 넣으면 실제 카메라의 위치에 대한 축을 생성할 수 있음

    DATA_SIZE = 8

    # object point
    # X: forward, Y: left, Z: 1
    # homography를 하기 위해서는 0index와 (1,2)index를 떼어내서 카메라 좌표계로 변환해줘야 한다. (2,-1,0)
    # homo에서 계산을 편하게 하기 위해 z를 사용하지 않으려고 1로 지정
    homo_object_point = np.append(object_points[:,2:3], -object_points[:,1:2], axis=1)
    homo_object_point = np.append(homo_object_point, np.ones([1, DATA_SIZE]).T, axis=1)

    print(homo_object_point)


    # 지면에 대해서 위치와 자세 추정이 가능하다면,
    # 임의의 포인트를 생성하여 이미지에 투영할수있다.
    # extrinsic은 차량의 중심축에 대한 값들이므로 , 객체와 차량의 중심축으로부터의 거리를 다 계산할 수 있고, 이를 통해 범퍼까지의 거리를 계산할 수 있다.
    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distCoeffs=None, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)

    # 잘 맞지 않는다.
    # 왜냐하면, 이미지 좌표와 실제 오브젝트와의 관계가 부정확하기 때문
    # 실제 측정을 통해 개선이 가능하다.
    # TODO: 축과 차선의 위치가 맞지 않는 이유는 실제 이미지 좌표에 대한 거리정보를 모르기 때문에 발생됨. 실측을 통해 정확한 거리값으로 맵핑하면 잘 맞게 된다.
    image = cv2.drawFrameAxes(image, camera_matrix, None, rvec, tvec, 1, 5)

    # proj_image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    # print(proj_image_points) # TODO: 정확한 obj point를 한다면 출력값이 image points와 같은 값을 가질 것

    homography, _ = cv2.findHomography(image_points, homo_object_point) # None 이 나온다면 두 shape이 맞지 않기 때문이다.
    # print(proj_image_points.shape)
    
    # (u, v) -> (u, v, 1)
    appned_image_points = np.append(image_points.reshape(8, 2), np.ones([1, DATA_SIZE]).T, axis=1)
    # print(homography.shape)

    for image_point in appned_image_points:
        # estimation point(object_point) -> homography * src(image_point)
        estimation_distance = np.dot(homography, image_point)

        x = estimation_distance[0]
        y = estimation_distance[1]
        z = estimation_distance[2]

        print(x/z, y/z, z/z) # homogeneous 좌표게에서는 마지막이 항상 1이어야 하므로 나눔
        # homo object point와 비교하여 얼마나 차이나는지 확인
```

- 순서
1. 실제 측량을 통해 obj point와 image point를 생성
2. random의 intrinsic 정보를 설정
3. opj point는 world 좌표계이므로 이를 카메라 좌표계로 변환
4. solvePnp를 통해 카메라 위치 추정 rotation, translation vector를 얻고 그를 통해 drawFrameAxes
5. 이미지 좌표와 실제 오브젝트의 관계를 수정
6. projectPoints를 통해 iamge point와 object point가 거의 동일한지 확인
7. findHomography를 통해 iamge point와 object point에 대한 변환행렬 생성
8. iamge point의 좌표계를 homogeneous 좌표계로 변환
9. 각 image point에 대해 homography를 곱하여 거리를 추정

여기서 중요한 점 : 호모그래피는 평면과 평면 사이의 변환을 의미한다. 따라서 바닥면이 평면이 아니라면 사용하기 어렵다.

<br>

<br>

- **geometrical distance estimation**

intrinsic을 통해 이미지 왜곡 보정 및 bbox에 대한 거리 추정한 값을 실제와 비교하는 코드

```python
# Step B. geometrical distance estimation

import json
import cv2
from cv2 import undistort
import matplotlib.pyplot as plt
import os
import numpy as np

# file parsing
json_file_path = os.path.join("data", "000076.json")
image_file_path = os.path.join("data", "1616343619200.jpg")

window_name = "Perception"

with open(json_file_path, "r") as json_file:
    labeling_info = json.load(json_file)

image = cv2.imread(image_file_path, cv2.IMREAD_ANYCOLOR)

camera_matrix = np.asarray(labeling_info["calib"]["cam01"]["cam_intrinsic"], dtype=np.float32)
dist_coeff = np.asarray(labeling_info["calib"]["cam01"]["distortion"], dtype=np.float32)
undist_image = cv2.undistort(image, camera_matrix, dist_coeff, None, None)

labeling = labeling_info["frames"][0]["annos"]
class_names = labeling["names"]
boxes_2d = labeling["boxes_2d"]["cam01"]

CAMERA_HEIGHT = 1.3 # TODO: 이 값에 따라 거리가 크게 바뀜. 지면과 카메라가 바라보는 방향이 이루는 각이 특정한 각을 이룬다면 pitch에 대한 보정이 필요하다.

# distance = f * height / img(y)
# 종/횡 방향으로 분리된 거리가 아닌, 직선거리
# FOV 정보를 알면 -> 종/횡 분리가 가능하다.

index = 0
for class_name, bbox in zip(class_names, boxes_2d):
    xmin, ymin, xmax, ymax = bbox
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0: # if -1, pass
        continue

    width = xmax - xmin
    height = ymax - ymin

    # Normalized Image Plane
    y_norm = (ymax - camera_matrix[1][2]) / camera_matrix[1][1]

    distance = 1 * CAMERA_HEIGHT / y_norm

    print(int(distance))
    cv2.rectangle(undist_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    cv2.putText(undist_image, f"{index}-{class_name}-{int(distance)}", (xmin, ymin+25), 1, 2, (255, 255, 0), 2)
    index += 1

display_image = cv2.cvtColor(undist_image, cv2.COLOR_BGR2RGB)
plt.imshow(display_image)
plt.show()
```

- 순서
1. 실제 객체와 카메라 사이의 거리를 측정
2. 카메라 intrinsic, distortion 획득
3. undistorting image
4. bbox 정보 저장
5. bbox 정보에 대한 fy, cy를 가져와서 normalized image plane 구함
6. distance 저장
7. 실제 거리와 distance가 맞는지 확인하고, 실제 camera 높이도 측정하여 camera height를 조정
8. 만약 높이가 맞다면 intrinsic이 틀렸거나 bbox가 틀리게 나온 것
9. bbox가 맞게 들어오고 있는지 확인을 위해 visualize
10. 맞다면 intrinsic 조정


<br>

<br>

- calibration_parser.py

```python
import json
import numpy as np

def read_json_file(path):
    print("=" * 50)
    print("Read JSON File: ", path)
    print("=" * 50)
    with open(path, "r",) as f:
        calibration_json = json.load(f)

    intrinsic = calibration_json["intrinsic"]
    print("Intrinsic Calibration\n", intrinsic)
    extrinsic = calibration_json["extrinsic"]
    print("Extrinsic Calibration\n", extrinsic)

    camera_matrix = parse_intrinsic_calibration(intrinsic)

    return camera_matrix

"""
    [
        [fx, 0, cx = u0],
        [0, fy, cy = v0],
        [0,  0,  1]
    ]
"""

def parse_intrinsic_calibration(intrinsic):
    fx = intrinsic["fx"]
    fy = intrinsic["fy"]
    cx = intrinsic["u0"]
    cy = intrinsic["v0"]
    camera_matrix = np.zeros([3, 3], dtype=np.float32)
    camera_matrix[0][0] = fx
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = fy
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 0.0

    return camera_matrix
```


# Summary

3차원 정보를 복원하는 방법
1. 카메라의 extrinsic을 이용하는 방법
  - 지면과 카메라 광학축을 활용한 삼각비
2. 카메라의 intrinsic과 대상 객체의 사전 정보를 이용하는 방법
  - FOV를 활용하여 이미지 bbox에 대한 dx, dy를 구하는 방법
  - geometrical distance estimation
3. 카메라의 image plane과 대상 plane의 변환 matrix를 사용하는 방법
  - warpPerspectiveTransform, findHomography
  - homography distance estimation

카메라는 본질적으로 3차원 공간에 대한 정보가 이미지를 취득하는 동시에 소실되기 때문에 3차원 공간에 대한 이해가 굉장히 어렵다.

<br>

이때까지는 1개의 카메라(Monocular camera)를 사용한 3D vision 방법을 배웠다. 그러나 카메라는 1개만 사용하는 것은 불가능하기에 multiple camera를 사용해야 하는데 이는 매우 복잡한 분야다. multiple camera에 대한 바이블인 눈깔책으로 불리는 `Multiple View Geometry` 책을 보길 추천한다.

<img src="/assets\img\dev\week14/book.jpg">