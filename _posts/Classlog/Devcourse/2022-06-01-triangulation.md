---
title:    "[데브코스] 15주차 - Visual-SLAM triangulation and perspective "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-06-01 14:20:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, Visual-SLAM]
toc: true
comments: true
math: true
---

<br>

# Triangulation

[지난 글](https://dkssud8150.github.io/posts/motion_estimation/)에서 F-matrix와 E-matrix를 구하는 방법을 배웠고, 이 matrix를 분해하면 두 이미지 간의 translation과 rotation matrix를 구할 수 있었다. 이 후의 작업으로 3D 구조를 복원해주는 mapping 단계가 수행되어야 한다. mapping의 첫단계가 **triangulation** 이다.

*Triangulation*이란 두 개의 이미지 사이에서 translation 값과 두 이미지들간의 correspendence를 알고 있을 때, feature들이 의미하는 실제 x,y,z값을 복원하는 과정이다. 이 과정이 결국 2D에서 3D point로의 변환이다.

<img src="/assets/img/dev/week16/day2/triangulation.png">

위의 그림은 2개의 3D point를 각각의 ray로 나타낸 것으로 두 쌍의 correspendence를 triangulation을 통해 3d Point를 구한 후, 2개의 point 간의 거리 값도 추정을 할 수 있다.

<br>

현실에서는 2개의 카메라가 1개의 point를 본다고 해도 노이즈가 껴있기 떄문에 두 이미지 간의 오차가 존재할 것이다. 그러면 두 카메라에서의 point 두 개를 평균내서 가상의 점으로 추정하도록 만들어야 한다.

<img src="/assets/img/dev/week16/day2/two_ray_trangulation.png">

이 때, F와 G를 구하는 식은 다음과 같다.

$ F = p + \lambda r $

$ G = q + \mu s$

$ \lambda $와 $ \mu $ 는 image point, `P`와 `Q`에 대한 3D point로 가는 방향 벡터이고, `r`과 `s`는 방향 벡터에 대한 스케일이다. 그리고 p와 q는 *camera center*를 의미한다.

즉, 시작 위치에서 방향벡터와 스케일을 곱한 것을 더하면 3d point를 구할 수 있다. 이 때, p와 q는 우리가 아는 값이고, r과 s는 다음과 같이 구할 수 있다.

$ r = R'^T x'^k \; ,\color{gray} with \;\; x'^k = (x',y',c)^T $ 

$ s = R''^T x''^k \; ,\color{gray} with \;\; x''^k = (x'',y'',c)^T $

$ x'^k $ 는 calibration된 P 카메라에 대한 camera coordinate에서의 좌표를 의미하고, $ x''^k $ 는 calibration된 Q 카메라에서의 2d point를 의미한다. 여기서 x',y'는 정확한 pixel 위치, c는 focal length이다. `R'`은 P에 대한 world coordinate에서 camera coordinate로의 옮기는 matrix를 의미한다. 그에 대한 T는 camera coordinate에서 world coordinate로 변환하는 matrix이다. 

<br>

그 후, 현재 2개의 3D ray는 교차하지 않는다고 가정했기 때문에, H를 구해줘야 한다. 기하학적으로 봤을 때, 점 F와 G를 잇는 선은 P ray와 Q ray에 대해 가장 가깝게 그은 선이므로 수직을 이룬다.

따라서, 수직을 이루는 두 직선에 대한 수식을 활용하여 H를 구할 수 있다.

$ (f - g) \cdot r = 0 \;\; (f - g) \cdot s = 0 $

<br>

$ (p + \lambda r - q - \mu s) \cdot r = 0 $

$ (p + \lambda r - q - \mu s) \cdot s = 0 $

이 때, 우리는 p,q,r,s 를 알고 있기 때문에 미지수는 $\lambda,\mu$ 2개 뿐이다. 따라서 2개의 식을 활용하여 미지수를 모두 구해낼 수 있다. 그러면 결국 F,G를 구해낼 수 있고, 이 두 점을 잇는 line을 그리고 그 중간값을 계산하면 **H**를 구할 수 있다. 그러나 이 때 굳이 중간점을 사용하지 않아도 F와 G의 불확실성을 구하여 weight를 지정해줄 수 있다. 예를 들어 확률적으로 GT값이 P와 더 가깝다면 F의 가중치를 0.5대신 0.7로 하여 H를 구할 수도 있다.

<br>

<br>

위의 식을 조금 더 풀어보도록 하자. p와 q는 각각 camera center 즉, 카메라 원점을 의미한다. 이를 X_O로 표현하고자 한다.

$ (X_{O'} + \lambda r - X_{O''} - \mu s)^T r = 0 $

$ (X_{O'} + \lambda r - X_{O''} - \mu s)^T s = 0 $

이고, 이를 정리하면 간단한 식이 만들어진다.

$ \begin{bmatrix} (r^Ts - s^Tr) \\ (r^Ts - s^Ts) \end{bmatrix} \begin{bmatrix} \lambda \\ \mu \end{bmatrix} = \begin{bmatrix} (X_{O''} - X_{O'})^T \\ (X_{O''} - X_{O'})^T \end{bmatrix} \begin{bmatrix} r \\ s \end{bmatrix} =\> Ax = b =\> x = A^{-1}b $

<br>

<br>

## Stereo Triangulation

이는 두 카메라가 함께 전방을 바라보는 상황을 의미한다.

<img src="/assets/img/dev/week16/day2/stereo_triangulation.png">

이 둘 간의 baseline을 `B`, 3D world에서의 좌표 P가 각각의 이미지에 `x'`, `x''`으로 매핑된다. 각각의 feature `x'`,`x''`의 거리를 **Parallax** 또는 **Disparity**라 한다.

위의 그림을 위에서 바라보면 다음과 같은 그림이 만들어진다.

<img src="/assets/img/dev/week16/day2/similar_triangle_z.png">

이 때, 3D point, P와 왼쪽 카메라의 카메라 원점 O'에 대한 ray를 O''으로 옮겨서 삼각형 O''PA을 만든다.

- c : focal length
- B : baseline
- (x' - x'') : disparity

삼각비를 활용하여 Z값을 구해줄 수 있다.

$ Z : c = B : (x' - x'') \; =\> \; \cfrac{Z}{c} = \cfrac{B}{x' - x''} \; =\> \; Z = c \cfrac{B}{-(x'' - x')}$ 

이 때, c와 B는 상수이므로 x''와 x만 알아도 Z값을 구할 수 있다. 동일하게 X방향도 추정이 가능하다.

<img src="/assets/img/dev/week16/day2/similar_triangle_x.png">

$ Z : c = X : x' \; =\> \; X = x' \cfrac{B}{-(x''-x')} $ 

이전에 Z를 구했으면 값을 그대로 집어넣으면 되고, 구하지 않았더라도 이전의 식을 그대로 사용하여 X를 구할 수 있다.

<br>

Y방향도 동일하게 구해줄 수 있을 수도 있지만, 이는 틀린 가정이다. 원래 두 카메라의 feature point가 F와 G로 완벽하게 동일하지 않았다. 그런 상황에서 X를 동일하게 맞추고 계산을 했으므로 Y는 당연히 동일하지 않는다. 반대로 Y를 먼저 구하기 위해 동일시해서 계산을 한다면 X방향의 값이 동일하지 않을 것이다.

그렇기에 Y방향으로의 계산은 조금 다른 방식으로 구해야 한다. 왼쪽 카메라에서의 Y값과 오른쪽 카메라에서의 Y값을 평균내서 Y를 추정한다.

<img src="/assets/img/dev/week16/day2/similar_triangle_y.png">

<br>

<br>

# Perspective n Points

PnP(perspective n Points)는 n개의 데이터를 통해 world to camera coordinate transformation을 수행하는 것을 목표로 한다. PnP는 이미지에서 feature를 뽑고, world coordinate와 매칭함으로써 world에서의 나의 위치를 찾을 수 있도록 해준다.

<img src="/assets/img/dev/week16/day2/pnp.png">

나의 위치를 찾게 해준다는 것은 localization을 수행해준다는 말이고, 실제로 visual localization 태스크에서 PnP는 많이 사용되는 기법이다. PnP에서는 3d point가 담고 있는 descriptor와 현재 이미지에서 보이는 descriptor를 매칭해서 2D-3D correspondence를 구하고, 충분한 양의 correspondence pair가 모이면 카메라 위치를 구할 수 있게 된다. 카메라 위치를 구하는 것이므로 6DoF를 가지고 있다. 즉 world coordinate system에서 camera coordinate system으로의 각각 Rotation matrix 3, translation 3을 가진다. 그래서 총 6개의 데이터이자 3개의 데이터 쌍을 사용하는 minimum solver에 해당한다.

그러나 p3p의 경우는 노이즈를 전혀 고려하지 않는 알고리즘이므로, outlier를 제거해줄 RANSAC 알고리즘이 필요하다.

PnP solver의 경우 OpenCV에서는 `solvePnP()`라는 함수로 구현되어 있다. [OpenCV docs](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)

<img src="/assets/img/dev/week16/day2/solvePnP.png">

PnP의 경우 픽셀값으로 데이터를 계산하기 때문에, Calibration matrix, 즉 intrinsic matrix를 가지고 있어야 연산이 가능하다.

<br>

<br>

## P3P - Grunert

P3P를 푸는 방법은 다양하다. 1842년도에 나온 Grunert가 있고, 그 이후에 나오는 E-P3P와 같이 다양한 알고리즘들이 있지만, 대부분의 경우 Grunert 기법을 기반으로 만들어졌다. 그래서 Grunert method를 배워보고자 한다.

<br>

<img src="/assets/img/dev/week16/day2/grunert.png">

기본적으로 P3P 문제를 풀기 위한 전제 조건으로는 실제 world coordinate에서의 세 점 좌표(A,B,C)를 모두 알고 있고, 이미지 위에서의 세 점 좌표(a,b,c)도 알고 있어야 한다.

그러면 optical center로부터의 픽셀 위치(focal length)를 알고 있기 때문에, 각 ray들마다의 각도($\theta$)를 계산할 수 있다. 그러나 optical center에서 각 점으로의 거리는 알지 못한다. 그래서 P3P에서는 2가지 단계로 진행된다.

1. projection ray들의 길이를 추정
2. 길이와 각 ray들의 각도를 통한 방향 추정

<br>

triangulation에서 배웠듯이 $ s_i x_i^k = R(X_i - E) $ 식을 사용하여 ray의 길이를 추정할 수 있다.

- i : i번째 ray
- s : ray의 길이
- x^k : object point를 향하고 있는 ray의 방향 벡터
- R : world to camera coordinate transformation의 rotation matrix
- X_i : A,B,C의 좌표
- E : optical center

R이 world to camera transformation이므로 X_i - E, 즉 ray의 길이가 camera coordinate에서의 표현으로 만든다. 그렇게 만들어진 길이와 방향 벡터, x_i와 거리, s를 곱한 값과 같다는 것을 알 수 있다.

<br>

ray의 방향벡터 x를 조금 더 자세히 보면

$ x_i^K = -sin(c)N(K^{-1}x_i) $

- c : focal length, -를 붙여준 이유는 대부분의 focal length의 방향은 image point에서 optical center로의 방향이다. 그래서 -를 붙여 center에서 image point로 가도록 만들어준다.
- N : x_i^k는 방향 벡터이므로 유닛 벡터로 만들기 위한 normalize
- K^-1 : x는 현재 픽셀 단위로 구성되어 있다. 그리고 K는 intrinsic matrix를 의미하는데, intrinsic matrix와 normalized image plane에 있는 값을 곱해주면 픽셀값으로 변환이 된다. 그래서 이를 역으로 normalized image plane에서 pixel값으로 변환해준다.

### 1. Length of projection rays

먼저 ray들간의 각도를 구한다. 각도를 구할 때는 내적을 사용하여 구한다. 두 벡터간의 내적은 길이를 서로 곱하고, 두 벡터가 이루는 각도의 cos값을 통해 구한다. $ \overrightarrow{a} \cdot \overrightarrow{b} = \| \overrightarrow{a} \| \| \overrightarrow{b} \| cos\theta$ 

따라서 ray간의 각도를 구할 때 내적 공식을 사용한다. optical center를 X_0, 각 포인트를 각각 X_1,X_2,X_3 라 할 때의 각각의 각도 alpha, beta, gamma를 구한다.

<img src="/assets/img/dev/week16/day2/length_ray.png">

$ cos\gamma = \cfrac{(X_1 - X_0) \cdot (X_2 - X_0)}{\| X_1 - X_0 \| \|X_2 - X_0 \|} $

<br>

어차피 우리는 방향벡터, x_i^k를 알고 있다면, 간단한 식으로 표현이 가능하다.

$ \alpha = arccos(x_2^k, x_3^k), \; \beta = arccos(x_3^k, x_1^k), \; \gamma = arccos(x_1^k, x_2^k) $

<br>

<br>

그 후 우리는 각각의 3D point 위치를 알고 있으므로 3D point들간의 거리도 알 수 있다. 

$ a = \| X_3 - X_2 \|, \; b = \| X_1 - X_3 \| ,\; c = \| X_2 - X_1 \| $

<br>

<img src="/assets/img/dev/week16/day2/calcut_length_ray.png">

이렇게 각 ray들간의 각도와 a,b,c를 알고 있다면 코사인 제2 법칙을 활용하여 각 ray의 길이 s1,s2,s3에 대해 정리해줄 수 있을 것이다. 그러나 이 3개의 식만으로는 s를 구할 수가 없다.

<br>

그래서 나온 방법으로는 다른 외부 센서를 활용하여 orientation을 판단해서 방정식을 풀어내거나 다른 1개의 점을 하나 더 가져와서 풀어낼 수 있을 것이다. 그러나 전자의 경우 다른 센서의 값을 요구하기 때문에 카메라만 사용하는 visual SLAM에는 맞지 않다. 후자의 경우도 1개의 점을 더 요구하기 때문에 P3P solution에는 맞지 않다.

<br>

최근 P3P 논문을 보면 위의 방법을 해결하기 위한 다른 기하학적 방식을 사용하기도 한다.

<br>

<br>






