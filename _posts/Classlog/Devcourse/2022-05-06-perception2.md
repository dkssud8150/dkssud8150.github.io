---
title:    "[데브코스] 12주차 - DeepLearning Perception Applications"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-05-06 14:20:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, deeplearning]
toc: true
comments: true
math: true
---

<br>

자율주행에서 Perception, 인지란 인식 + 이해하는 과정으로, 유의미한 정보를 생성하는 단계가 필요하다. 유의미한 정보라 함은 객체를 인식하는 것도 중요하지만, 객체를 인식하는 것만으로는 거리 정보를 얻을 수 없다. 그래서 그 객체와의 거리를 추정하는 `3D POSE estimation`도 함께 수행해야 한다. 

<br>

3D POSE estimation을 적용하기 위해서는 **vision geometry**의 기본적인 지식이 수반되어야 한다. 그를 위해 카메라의 투영 과정을 이해해야 한다.

<img src="/assets/img/dev/week12/day4/projection.png">

<br>

3차원 위치를 가진 객체가 (Xw, Yw, Zw) 좌표를 가진 P로 표현되었을 때, 이 점이 카메라에 투영되면 (u,v) 좌표 지점에 상이 존재하는 이미지 평면을 얻게 된다. 그렇다면 이 이미지 평면을 통해 객체 위치(X,Y,Z)를 추정하는 과정도 수행이 가능해진다.

사람은 2D 이미지만 봐도 입체감이 느껴지는데, 이를 공간감(illusion of space)라 한다. 사람은 이 공간감을 가지고 있지만, 컴퓨터 비전(기계)는 이 공간감을 가지고 있지 않기 때문에 다양한 수학적 모델을 통해 공간감을 대체해야만 한다.

<br>

<img src="/assets/img/dev/week12/day4/illusion.png">
<img src="/assets/img/dev/week12/day4/illusion2.jpg">

실제에서는 직선이지만, 카메라에서 원근법으로 인해 기울어지게 그려진 직선들이 모이는 점을 **소실점**(vanishing point)이라 하고, 찍는 각도에 따라서 소실점의 위치는 달라진다. 소실점으로 모이고 있는 여러 직선들을 소실선(vanishing line)이라 한다.

<br>

사람이 사물을 인식한다는 것은 물체에 반사된 빛들을 인식하는 것이고, 카메라는 그 빛을 기록하는 장치이다. 힌홀 카메라(pinhole camera)는 빛이 들어오는 구멍을 아주 작게 만들어서 맨 부터 맨 아래까지의 반사된 빛들이 매우 작은 구멍으로 들어와 뒤집힌 채로 상이 맺힌다.

<img src="/assets/img/dev/week12/day4/pinhole.jpg">

이 카메라는 동일한 크기의 물체여도 가까이서 촬영하면 크게 보이고, 멀리서 촬영하면 작게 보인다. 그 이유는 멀리 있으면 물체가 이루는 각도가 작아서 상이 작게 맺히고, 가까우면 물체가 이루는 각도가 커서 상이 크게 맺힌다.

<img src="/assets/img/dev/week12/day4/pinhole2.jpg">

이를 수학적으로 판단해봤을 때, 투과되는 구멍과 맺힌 상과의 거리를 `f`라 하고 맺힌 상의 크기를 `h`, 물체와 구멍사이의 거리를 `D`, 실제 물체의 크기를 `H`라 할 때, h는 f와 D의 크기에 따라 달라진다.

<br>

<img src="/assets/img/dev/week12/day4/projection.png">

다시 이 그림으로 돌아와서, 이미지 평면에서의 점,s 와 실제 3차원에서의 점 좌표, P에 대해 `projection matrix`를 세워보면 다음과 같다.

$$ s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x \ 0 \ c_x \\ 0 \ f_y \ c_y \\ 0 \ 0 \ 1 \end{bmatrix} \begin{bmatrix} r_{11} \ r_{12} \ r_{13} \ t_1 \\ r_{21} \ r_{22} \ r_{23} \ t_2 \\ r_{31} \ r_{32} \ r_{33} \ t_3 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

u,v,1은 이미지 평면에서의 점 좌표를, X,Y,Z,1은 실제 좌표계에서의 점 좌표를 나타낸다. 그리고 우항에서 1번째 항을 `Intrinsic항`, 2번째 항을 `extrinsic항`이라 한다. intrinsic은 카메라 자체의 특성을 의미하고, extrinsic은 카메라와 대상과의 관계를 의미한다. extrinsic에서 r은 회전(rotation), t는 평행이동(translation)을 의미히므로, 따라서 기준 좌표계가 이미지 좌표계로 변환되기 위한 좌표 변환을 나타낸다.

<br>

pinhole camera model이외에도 fisheye camera model(초광각), weak perspective model 등이 있다. fisheye 는 369도를 다 관찰할 수 있는 카메라이고, weak perspective는 약한 변환, 즉 가상의 depth plane을 생성해서 투과하여 변환이 덜 되도록 만드는 방법이다.

사용하려는 카메라에 따라 matrix가 다 달라지므로 확실하게 알아보고 연산을 수행해야 한다.

<br>

# Camera Intrinsic calibration

camera calibration이란 카메라가 가지고 있는 고유한 특성을 파악하는 과정이다. 카메라 렌즈와 이미지 센서와의 관계로부터 파생되는 초점거리가 intrinsic에 해당되고, 카메라의 위치와 자세에 대한 특성은 extrinsic에 해당한다.

같은 거리에서 서로 다른 카메라로 동일한 피사체를 촬영하면 결과가 다르다. 이는 카메라의 intrinsic 특성이 달라서 생기는 문제이다. 같은 카메라로 다른 위치에서 동일한 피사체를 촬영해고 결과는 다르게 나온다. 이는 extrinsic 특성이 다르기 때문에 생기는 문제이다.

projection matrix에서 intrinsic 특성을 보면 f(focus distance)와 c(center point)가 있다. 이는 각각 초점거리와 주점을 의미한다.

<br>

<br>

- 초점거리

초점거리란 렌즈 또는 구멍으로부터 상이 맺힌 평면 사이의 거리를 말한다. 동일한 x에 대해 초점거리를 측정한다면 f_y라고 표기할 수 있고, 동일한 y에 대해 상이 맺힌 초점거리를 측정한다면 f_x라고도 표현할 수 있다. 이상적인 카메라의 특성으로는 `f_x = f_y`가 될 것이다. 초점거리에 따라 피사체의 크기가 달라진다. 

<br>

컴퓨터 비전에서 초점거리는 pixel 단위로 표현된다. 카메라에서는 초점거리를 mm단위로 표현하지만 픽셀 단위로 표현하게 되면, `f = 500(pixel)` 가 되고, 만약 cell 하나의 크기를 0.1mm이라 정의하고, f=500px, 해상도가 100x100, cell size도 100x100 라면 카메라에서의 초점거리는 `f = 0.1mm x 500px = 50mm`라 할 수 있다. 그러나 만약 동일한 조건에서 해상도가 50x50이라면 `0.2mm x 500px = 100mm`이 된다. 또는 f(mm)가 50으로 고정되어 있다면 `f = 0.2 x 250px = 50mm`이라는 식이 된다.

이를 일반화하면

- n x n cell size, p mm each pixel => r x r resolution
- (n/r) x (n/r) cell size = p mm
- p mm * focal length(pixel) = focal length(mm)

<br>

- 주점

주점이란 pinhole(구멍)의 중심이 이미지 센서에 직교하는 위치(Cx,Cy)를 의미한다. 대체로 이미지가 (height, width)의 크기를 가진다고 할 때,  $ Cx = \frac{width}{2}, Cy = \frac{height}{2} $ 라 할 수 있으나 모든 경우가 이렇지는 않다. 이미지의 중심점과 주점(렌즈 중심점)은 다른 의미인데, 이상적인 경우에는 이 두개가 일치하지만, 카메라 제조 공정에서 발생하는 다양한 이슈로 인해 일치하지 않는 경우도 있다. 따라서 이 주점을 구하기 위해 calibration이 정의하는 것이 필요하다.

<br>

<br>

원래의 instrinsic의 행렬은

$$ \begin{bmatrix} f_x \ 0 \ c_x \\ 0 \ f_y \ c_y \\ 0 \ 0 \ 1 \end{bmatrix} $$

이지만, 공정에 의한 오류로 인해 발생하는 계수를 표기하기 위해 추가 파라미터가 존재한다.

$$ \begin{bmatrix} f_x \ skew_{cf_x} \ c_x \\ 0 \ f_y \ c_y \\ 0 \ 0 \ 1 \end{bmatrix} $$

이 때, skew\_cfx는 이미지의 비대칭 계수(skew coefficient)를 의미하는데, 이는 이미지 센서가 기울어진 정도를 의미한다. 그러나 최근에는 이러한 상황이 발생하는 경우가 거의 없기 때문에 0으로 계산을 한다. 

<br>

## 카메라 좌표계

<img src="/assets/img/dev/week12/day4/projectionmatrix.png">

3차원 공간에 존재하는 한 물체를 2차원 이미지 공간에 투영되는 과정을 설명하기 위해서는 좌표계가 정의되어야 한다. computer vision에서는 4개의 좌표계를 사용한다.

- World(realworld) coordinate : Xw, Yw, Zw
- camera coordinate : Xc, Yc, Zc
- image coordinate : u, v
- Normalized image coordinate : u_n, v_n

P =\> Pn =\> Pimg =\> Pw 

<br>

- **world coordinate**

월드 좌표계는 우리가 살고 있는 3차원 공간에 존재하는 좌표계를 말한다. 어떤 물체의 위치를 3차원 좌표로 표현하면 (Xw, Yw, Zw) 이다. 이 좌표는 어떤 곳을 원점으로 고정하냐에 따라 값이 달라질 수 있는데, 컴퓨터 비전에서는 주로 카메라 좌표계를 원점으로 잡는다.

<br>

- **camera coordinate**

카메라 좌표계는 카메라를 기준으로 표현한 좌표계이다. 카메라를 기준으로 설정하였기 때문에 각각의 요소들은 다음과 같이 정의할 수 있다.

- Zc : 카메라 렌즈가 바라보는 방향
- Xc : 카메라 아래쪽 방향
- Yc : 카메라 오른쪽 방향

<img src="/assets/img/dev/week12/day4/cameradirection.png">

기본적인 월드 좌표계의 방향과는 조금 다르게 되어 있으므로 이를 하나로 통일하는 것이 중요하다.

<br>

- **image coordinate**

이미지 좌표계는 실제 이미지에 대한 데이터를 표현하는 좌표계이다. 이미지 좌표계의 기준은 다음과 같다.

- 이미지 왼쪽 상단을 원점으로 한다
- 이미지의 오른쪽 방향을 x 또는 u로 표현
- 이미지 아래쪽 방향을 y 또는 v로 표현
- 이미지 크기는 (height, width)로 표현

<br>

- **normalized image coordinate**

정규 이미지 좌표계는 실제로 존재하지 않는 좌표계로 컴퓨터 비전에서 해석을 위해 정의한 가상의 좌표계이다. 렌즈로부터 이미지 평면까지의 거리를 초점거리라 정의하고, 같은 물체를 동일한 위치에서 서로 다른 카메라로 촬영할 경우 이미지는 다르게 표현된다. 따라서 초점거리를 1로 정규화한 가상의 이미지 좌표계를 사용하여 초점거리에 대한 영향을 제거시킨다. 이를 위한 좌표계이다.

카메라의 intrinsic calibration 정보를 알면 이미지를 정규 이미지 좌표계로 변환이 가능해진다. 

$$ \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}= \begin{bmatrix} f_x \ 0 \ c_x \\ 0 \ f_y \ c_y \\ 0 \ 0 \ 1 \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} $$

좌항의 x,y,1이 이미지 좌표계, u,v,1이 정규 이미지 좌표계이다. 이를 변환하게 위해서는 초점거리와 주점에 대한 정보를 알아야 할 것이고, 이를 통해 행렬을 풀면 아래와 같다.

$$ u = \frac{(x - c_x)}{f_x} $$

$$ v = \frac{(y - c_y)}{f_y} $$ 

<br>

<br>

## Distortion

실제 카메라는 구멍이 아닌 볼록렌즈를 통해 빛을 모은다. 렌즈에 의해 빛이 굴절되는데, 빛의 굴절은 이미지 센서에 보이는 이미지를 왜곡한다. 직선이 곡선으로 보이는 상황이 왜곡이라 할 수 있다.

<img src="/assets/img/dev/week12/day4/distortion.png">

볼록 렌즈의 형상이 곡률을 가지는 구면 형태이므로 이미지 중심부(주점)에서 멀어질수록 표현의 비율이 달라지기 때문에 왜곡이 발생한다. 어느 위치에 있을 것이라 예측한 값인 PD(Predicted Distance)와 실제 거리 값인 AD(Actual Distance)가 존재할 때 렌즈의 왜곡 정도 D(%)는 다음과 같은 식을 가진다.

$$ D = \frac{AD - PD}{PD} * 100% $$ 

<br>

왜곡에서도 다양한 종류의 왜곡이 존재한다.

- 방사 왜곡

렌즈 왜곡의 대표적인 예로, 대표적으로 두 가지 형태로 표현된다.

<img src="/assets/img/dev/week12/day4/distortion2.png">

1. Barrel Distortion : 중심부가 외각부보다 원래의 크기에 비해 큰 형태로 발생
  - 가운데가 볼록한 형태
2. Pincushion Distortion : 중심부가 외각부보다 원래의 크기에 비해 작은 형태로 발생
  - 가운데가 오목한 형태

<br>

- 접선 왜곡(Tangential Distortion)

접선 왜곡은 렌즈와 이미지 센서의 관계에 의해 생겨나는 왜곡이다. 즉 이미지의 평면과 렌즈의 평면이 평행하지 않는다는 의미이다. 

<img src="/assets/img/dev/week12/day4/tangential.png">

접선 왜곡의 경우 가운데가 원형으로 볼록한 것이 아닌 타원으로 볼록/오목한 형태를 가진다.

<br>

- 원근 왜곡(Perspective Distortion)

방사/접선 왜곡이 대표적인 왜곡이지만, 다른 형태의 왜곡도 존재한다. 원근 왜곡은 3차원 공간이 2차원 공간으로 투영되면서 발생하는 왜곡이다. 이미지는 공간의 깊이 정보가 하나의 평면으로 투영되는 데이터이므로 촬영하는 환경에 따라 원근감 손실과 같은 다양한 왜곡이 생길 수 있다.

<br>

사람의 경우 이미지를 통해 원근감을 추정하기 위해서 세 가지 정보를 통해 추정한다.

1. 사물의 실제 크기에 대한 정보
2. 사물과 주변의 관계에 대한 정보
3. 추정이 가능한 기하학적 구조를 가지는지

이를 통해 원근감을 추정하는데, 이를 컴퓨터비전에서도 적용을 할 수는 있다. 그러나 이 원근 왜곡에 의해 손실된 객체에 대해서는 추정이 거의 불가능하다.

<br>

이 원근감 손실을 해결하기 위해 해결하기 위한 다양한 방법이 존재한다.

1. 다수의 카메라를 사용

2개 이상의 카메라로 동일한 시점에 촬영한 각 1장의 이미지만으로 3차원 위치 정보를 추정한다. 각 카메라의 Extrinsic 파라미터를 알아야 정확한 정보를 추정할 수 있다.

2. 2장 이상의 이미지를 사용

같은 카메라로 카메라가 움직이는 환경에서 연속된 이미지 정보를 활용하여 3차원 위치 정보를 추정한다. 카메라의 움직임 정보를 정밀하게 측정/추정해야 정확한 정보를 추정할 수 있다.

<br>

<br>

카메라 좌표계를 기준으로 3차원 공간상에 존재하는 객체를 투영하는 모델을 계산해본다. 이 때는 distortion이 있을 때와 없을 때 두 가지를 각각 계산하여 비교해볼 것이다.

extrinsic calibration에서 translation이 없다고 가정을 하면 rotation만 남게 된다.

$$ extrinsic = rotation \| translation = \begin{bmatrix} r_{11} \ r_{12} \ r_{13} \ t_1 \\ r_{21} \ r_{22} \ r_{23} \ t_2 \\ r_{31} \ r_{32} \ r_{33} t_3 \end{bmatrix} $$

이 때, translation이 없다면 rotation matrix는 identity matrix가 된다.

$$ extrinsic = \begin{bmatrix} r_{11} \ r_{12} \ r_{13} \\ r_{21} \ r_{22} \ r_{23} \\ r_{31} \ r_{32} \ r_{33} \end{bmatrix} = \begin{bmatrix} 1 \ 0 \ 0 \\ 0 \ 1 \ 0 \\ 0 \ 0 \ 1 \end{bmatrix} = I $$ 

그리고 normalized image plane에 투영을 한다면 instrinsic matrix도 생략이 가능하다. 따라서 투영하는 projection matrix의 식은 다음과 같이 간편화된다.

$$ \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix} $$

이를 다시 정리하면 

$$ \begin{bmatrix} u_{n_u} \ v_{n_u} \end{bmatrix} = \begin{bmatrix} \frac{X_c}{Z_c} \\ \frac{Y_c}{Z_c} \end{bmatrix} $$

여기서 u는 undistortion, 왜곡이 되지않음을, n은 normalize를 의미한다.

<br>

왜곡항은 또 radial(방사)와 tangential(접선) 왜곡, 2가지로 나뉠 수 있다. 그에 대한 식으로는 다음과 같다. d는 distortion, 왜곡이 됨을 의미한다.

$$ \begin{bmatrix} u_{n_d} \\ v_{n_d} \end{bmatrix} = (1 + k_1r_u^2 + k_2r_u^4 + k_3r_u^6) \begin{bmatrix} u_{n_u} \\ v_{n_u} \end{bmatrix} \begin{bmatrix} 2p_1u_{n_u}v_{n_u} + p_2(r_u^2 + 2u_{n_u}^2) \\ p_1(r_u^2 + 2v_{n_u}^2) + 2p_2u_{n_u}v_{n_u} \end{bmatrix} $$

그리고, r_u에 대해서는

$$ r_u^2 = u_{n_u}^2 + v_{n_u}^2 $$

의 식을 가진다.

<br>

normalize 이미지 평면을 image plane으로 변환을 하려면 intrinsic matrix를 추가하면 된다.

$$ \begin{bmatrix} x_{p_d} \\ y_{p_d} \\ 1 \end{bmatrix} = \begin{bmatrix} f_x \ skew\_cfx \ c_x \\ 0 \ f_y \ c_y \\ 0 \ 0 \ 1 \end{bmatrix} \begin{bmatrix} u_{n_d} \\ v_{n_d} \\ 1 \end{bmatrix} $$ 

p는 plane을 의미한다. 그래서 최종적인 좌표 x,y,1에 대한 왜곡이 있는 이미지 projection matrix를 완성한다. 위의 식을 정리하면 다음과 같이 간략해진다. skew\_cfx 는 대체로 0으로 간주하므로 0을 넣어준다.

$$ x_{p_d} = f_x(u_{n_d} + skew\_cfxv_{n_d}) + c_x = f_xu_{n_d} + c_x , y_{p_d} = f_yv_{n_d} + c_y $$ 

<br>

지금까지는 왜곡이 없는 이미지에서 왜곡이 있는 이미지를 계산했다. 그러나 우리가 실제로 궁금한 것은 왜곡이 있는 이미지에서 왜곡이 없는 이미지 좌표를 알고 싶기에 이를 계산해보자. 

앞서 계산한 image plane을 역으로 계산하면 된다.

$$ \begin{bmatrix} x_{p_u} \\ y_{p_u} \\ 1 \end{bmatrix} = \begin{bmatrix} f_x \ skew\_cfx \ c_x \\ 0 \ f_y \ c_y \\ 0 \ 0 \ 1 \end{bmatrix} \begin{bmatrix} u_{n_d} \\ v_{n_d} \\ 1 \end{bmatrix} -> \begin{bmatrix} u_{n_d} \\ v_{n_d} \\ 1 \end{bmatrix} = \begin{bmatrix} f_x \ skew\_cfx \ c_x \\ 0 \ f_y \ c_y \\ 0 \ 0 \ 1 \end{bmatrix}^{-1} \begin{bmatrix} x_{p_u} \\ y_{p_u} \\ 1 \end{bmatrix} $$ 

$$ u_{n_u} = \frac{x_{p_u} - c_x}{f_x} - skew\_cv_{n_u} , v_{n_u} = \frac{(y_{p_u} - c_y)}{f_y} $$

<br>

여기서 왜곡 모델을 적용하면 다음과 같다. 왜곡x -\> 왜곡o 변환 과정과 다른 점은 역행렬(-1)이 들어간 것뿐이다.

$$ \begin{bmatrix} u_{n_d} \\ v_{n_d} \end{bmatrix} = (1 + k_1r_u^2 + k_2r_u^4 + k_3r_u^6) \begin{bmatrix} u_{n_u} \\ v_{n_u} \end{bmatrix} \begin{bmatrix} 2p_1u_{n_u}v_{n_u} + p_2(r_u^2 + 2u_{n_u}^2) \\ p_1(r_u^2 + 2v_{n_u}^2) + 2p_2u_{n_u}v_{n_u} \end{bmatrix} $$

$$ \begin{bmatrix} x_{p_d} \\ y_{p_d} \\ 1 \end{bmatrix} = \begin{bmatrix} f_x \ skew\_cfx \ c_x \\ 0 \ f_y \ c_y \\ 0 \ 0 \ 1 \end{bmatrix} \begin{bmatrix} u_{n_d} \\ v_{n_d} \\ 1 \end{bmatrix} $$

<br>

이 모든 과정은 코드로 직접 옮길 필요없이 openCV의 함수로 `undistort`나 `initUndistortRectifyMap`&`remap` 이다. 이 함수의 의미, 즉 왜곡 보정은 왜곡된 이미지로부터 왜곡이 제거된 이미지로의 mapping을 의미한다.

<br>

<br>

## Intrinsic Calibration code

intrinsic matrix와 왜곡 계수(distortion coefficients)를 계산하는 코드를 작성해보도록 하자.

- 참고: [https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

<br>

3차원 위치 과정
- input distorted image
- undistort
- detection
- pose estimation

여기서 전체 이미지가 아닌 detection 후 4개의 좌표에 대해서만 undistort를 사용하여 위치 추정을 하기도 한다.

<br>

### 캘리브레이션 보드를 활용한 intrinsic matrix 출력

위의 참고자료를 통해 툴박스를 사용하여 calibration을 할 수 있지만 직접 코드를 구현해서 사용할수도 있다.

intrinsic calibration을 수행하기 위해서는 준비물이 필요하다. calibration pattern이라 부르는 특별한 보드인데, 체스보드와 같이 직각으로 이루어진 사진이 좋다. 이 체스보드 모양을 프린트해서 화면에 비춰볼 예정인데, calibration 보드를 통해 3차원 공간을 이미지라는 2차원 공간에 투영하는 과정에서 필요한 카메라의 고유 특성을 파악할 수 있다.

현재 캘리브레이션 보드에서의 3차원 좌표 (x,y,z)를 (u,v)로 투영을 할 건데, 투영하는 식은

$$ \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} intrinsic \end{bmatrix} \begin{bmatrix} extrinsic \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix} $$ 

이고, 3차원 좌표 (x,y,z)를 알고, extrinsic은 원점을 기준으로 하고 있으므로 identity matrix가 되어서 intrinsic의 파라미터들을 최적화시킬 수 있게 되는 것이다. 이때, z는 일정하게 되어야 하므로 캘리브레이션 보드는 반드시 평면에 만들어져 있어야 한다. 또한, 각 grid cell의 정확한 크기와 grid 사이즈를 알고 있어야 한다. grid 사이즈란 맨 끝의 테두리를 제외하고 안쪽에 존재하는 직각좌표를 의미한다. 대체로 4x7과 같은 짝수 x 홀수로 이루어진 보드를 많이 사용한다.

<br>

```python
import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # constant 

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32) # chess board size , 3(x,y,z)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) * 0.025 # 0.025 == 1 grid size (m unit)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg') # image file list

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None) # gray image, grid size ->  bool value about finding corners, corner points of image 
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # refines corner locations using gradient direction
        # image,corners, window size for search,zero zone(no size of -1,-1),end point after max count or value under epsilon to move
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) 

        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret) 
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()
```

3차원 object 위치 정보인 objpoints와 2차원 이미지 공간에 존재하는 object 이미지 픽셀 위치 정보인 imgpoints를 통해 intrinsic 파라미터를 조정한다.

<br>

openCV의 calibrateCamera라는 함수를 사용하여 matrix를 구할 수 있다. 이 함수는 입력, 출력인자가 꽤 많다.

```python
ret, mat, dist, revecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```

입력
- object points for 3 dimension
- image points for 2 dimension in 2d image
- image size for gray scale
- intrinsic matrix (필수 인자이므로 가지고 있지 않으면 None)
- distortion coefficients (필수 인자이므로 가지고 있지 않으면 None)

출력
- True or False about calibration
- intrinsic calibration
  - intrinsic matrix
  - distortion coefficients
- extrinsic calibration
  - rvecs, tvecs : extrinsic matrix

여기서 rvecs, tvecs는 카메라의 위치와 calibration을 수행할 때의 가상의 카메라 위치(캘리브레이션 보드의 좌상단 좌표를 (0,0,0)으로 가정)에 대한 값이다.

<img src="/assets/img/dev/week12/day4/extrinsic.png">

objpoints를 카메라 좌표계인 (0,0,0)으로 시작하면 첫번째 인덱스를 기준으로 카메라가 해당 위치에 존재한다고 가정이 된다. 즉 원래의 위치에서 캘리브레이션 보드의 좌상단의 위치로 카메라 위치를 가정한다. 그렇다면 calibration은 가상의 카메라 위치를 기준으로 계산하지만, 실제로는 실제 카메라 위치와의 차이가 존재한다. 좌표계 변환을 위한 rotation, translation matrix를 반환한다. 이 extrinsic matrix들은 이미지를 어떻게 찍냐에 따라, 즉 캘리브레이션 보드의 위치에 따라 값이 다 달라진다.

<br>

<br>

### intrinsic matrix를 활용하여 OpenCV 왜곡 보정 함수를 통해 보정

OpenCV에서 왜곡 보정(undistortion)을 하는 방법은 2가지가 있다.

1. undistort
2. initUndistortRectifyMap & remap

<br>

```python
# undistort 1.
dst = cv.undistort(img, mtx, dist, None, mtx)

# undistort 2.
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, None, (w,h), 5)
remap = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
```

uninstort는 내부적으로 initUndistortRectifyMap, remap을 호출하고 있다. 하지만 recitymap를 통해 camera matrix(intrinsic)과 distortion coefficients(왜곡 계수)를 얻는 것이므로 매 이미지마다 연산을 할 필요가 없다. 그래서 두 함수를 분리하여 적용하는 것이 효율적이다.

<br>

<br>

추가적으로 getOptimalNewCameraMatrix()라는 함수를 사용하여 camera matrix, 즉 intrinsic matrix를 변환해줄 수 있다. 즉, 예를 들어 방사 왜곡이 0.5의 정도로 적용되는 카메라 특성이 있다고 생각했을 때, 이 함수를 사용하여 방사 왜곡이 0.8의 정도로 왜곡이 되는 matrix를 만들어낼 수 있다. 따라서 이 함수는 입력으로 camera matrix를 받아 출력으로 새로운 camera matrix를 츨력한다.

원래의 camera matrix와 새로운 camera matrix를 비교해볼 수 있다.

<br>

<br>

### 예제 코드

- image point 저장

chessboard 함수를 통해 image point를 저장한다. 추후 calibration에 사용될 점들이다.

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
print(np.shape(image_points))

# (13, 54, 1, 2)
# (number of image, number of feature point, list, image_point(u, v))
# ojbect_points
# (13, 54, 1, 3)
# (number of image, number of feature point, list, object_point(x, y, z))
```

이미지는 opencv에서 제공하는 파일들로, chessboard에 대한 사진이다.

<img src="/assets/img/dev/week12/day5/left01.jpg">

<br>

이 사진들을 불러와 각 board에서의 코너 점들을 저장하는 과정으로, 필요한 상수들은 grid(pattern)개수와 scale size이다. 맨 마지막 모서리값은 제외하고, 중간에서의 코너들의 개수를 센다. 투영을 할 때 scale의 크기도 지정해준다. flags는 opencv에서 제공하는 것으로 지정되어 있는 값들이다. 

저장한 이미지 리스트를 하나씩 불러와 이미지를 열고, `findChessboardCorners`를 통해 코너점들을 찾는다. ret는 코너점들을 찾았는지에 대한 true or false 값으로, 코너점들을 찾았을 경우 `display image`라는 인자를 통해 시각화 여부를 결정하여 찾은 점들을 보고 싶다면 display image를 True로 설정한다. 

image points의 shape을 출력해보면 (이미지 개수, 각 이미지 마다의 점 개수, 1, object point의 크기)로 이루어져 있다. 제공된 이미지의 개수가 13개이므로 13, 코너점의 개수가 9x6=54이므로 54이고, iamge point의 경우 2차원 좌표(u,v)로 이루어지므로 2이다.

<br>

<br>

- calibration

카메라가 바라보는 방향은 전방이 z, 오른쪽이 y, 아래쪽이 x이다. 위에서 출력해본 점들의 순서는 width방향으로 간 후 height방향으로 진행된다. 

<img src="/assets/img/dev/week12/day5/draw.png">

<br>

```python
"""
    forward: Z
    right: Y
    down: X
"""

for i in range(len(image_path_list)):
    object_point = list()
    height = 0
    for _ in range(BOARD_HEIGHT):
        # Loop Height -> 9
        width = 0
        for _ in range(BOARD_WIDTH):
            # Loop Width -> 6
            point = [[height, width, 0]]
            print("point",point)
            object_point.append(point)
            width += SQUARE_SIZE
        height += SQUARE_SIZE
    object_points.append(object_point)

# ---------------------- #

point [[0, 0, 0]]
point [[0, 0.025, 0]]
point [[0, 0.05, 0]]
point [[0, 0.07500000000000001, 0]]
point [[0, 0.1, 0]]
point [[0, 0.125, 0]]
point [[0, 0.15, 0]]
point [[0, 0.175, 0]]
point [[0, 0.19999999999999998, 0]]
point [[0.025, 0, 0]]
point [[0.025, 0.025, 0]]
```

가상의 카메라 위치를 기반으로 하면 좌상단 기준 (x,y,z) = (0,0,0)이 원점이다. 그리고, 이미지 평면은 2차원이므로 항상 z는 일정이기에 0으로 지정한다. 출력값을 보면, width, x방향이 먼저 증가하고, 끝나면 height가 1단계 증가한다. 0.025씩 증가하는 이유는 grid의 크기가 0.025이기 때문이다. 캘리브레이션 보드는 가로 0.025, 세로 0.025의 grid로 이루어져 있다.

<br>

```python
object_points = np.asarray(object_points, dtype=np.float32) # change numpy array

tmp_image = cv2.imread("images/left01.jpg", cv2.IMREAD_ANYCOLOR)
image_shape = np.shape(tmp_image)

image_height = image_shape[0]
image_width = image_shape[1]
image_size = (image_width, image_height)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

print("=" * 20)
print(f"re-projection error\n {ret}\n")
print(f"camera matrix\n {camera_matrix}\n")
print(f"distortion coefficientes error\n {dist_coeffs}\n")
print(f"extrinsic for each image\n {len(rvecs)} {len(tvecs)}")

# ---------------------------- #
re-projection error
 0.3811715802258524

camera matrix
 [[531.14747903   0.         341.82113196]
 [  0.         531.43207499 235.03970371]
 [  0.           0.           1.        ]]

distortion coefficientes error
 [[-2.74391037e-01 -3.07998458e-02  8.95674654e-04 -2.21527245e-04
   2.76816016e-01]]

extrinsic for each image
 13 13
```

opencv에 입력으로 넣을 때는 Mat이나 numpy로 입력이 되어야 하므로 numpy로 변환해준다.

1장의 이미지를 미리 읽어오는 이유는 위에서 말한 것과 같이 intrinsic matrix를 미리 얻어놓고 반복문에서는 적용만 시키는 것이 효율적이기 때문이다. openCV에서 size를 입력할 때는 (width, height) 순으로 되어 있다. numpy의 shape를 사용해보면 반대로 (height, width)로 출력되기 때문에 위와 같이 입력해준다.

출력을 보게 되면 camera matrix(instrinsic matrix)와 distortion coefficients(왜곡 계수)를 얻은 것을 볼 수 있고, extrinsic 벡터들은 각 이미지마다의 값을 가지므로 이미지 개수인 13개가 있는 것을 확인할 수 있다.

<br>

```python

for rvec, tvec, op, ip in zip(rvecs, tvecs, object_points, image_points):
    imagePoints, jacobian = cv2.projectPoints(op, rvec, tvec, camera_matrix, dist_coeffs)

    for det, proj in zip(ip, imagePoints):
        print(det, proj)
        sub += sum((det - proj)[0])
print(sub)

# ------------------- #
[[244.45415  94.33141]] [[244.59253   94.102356]]
[[274.62177  92.24126]] [[274.3844  92.1637]]
[[305.49387   90.402885]] [[305.59717  90.43659]]
[[338.36407   88.836266]] [[338.12567  88.95949]]
[[371.59216  87.98364]] [[371.82565   87.773575]]
[[406.84354   86.916374]] [[406.50888  86.92322]]
[[441.63345  86.37207]] [[441.93878  86.45395]]
[[477.623   86.3797]] [[477.83344  86.40558]]
[[513.9866   86.79912]] [[513.8886  86.7965]]
...
-0.01483917236328125

```

projectpoint 함수는 object points 들을 통해 다시 image_points 형태로 투영시켜주는 함수이다. 그렇게 출력된 imagePoints와 원래의 image_points를 비교해보면 거의 일치하는 것을 확인할 수 있다. 이 두개를 각각 뺀 값을 다 더한 후 전체적인 값들을 고려하여 출력한 값이 `re-projection error`, ret이다.

<br>

```python
start_time = time.process_time()
for image_path in image_path_list:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_undist = cv2.undistort(image, camera_matrix, dist_coeffs, None)
end_time = time.process_time()
print("undistort time : ",end_time - start_time)

start_time = time.process_time()
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, None, image_size, cv2.CV_32FC1)
for image_path in image_path_list:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_undist = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
end_time = time.process_time()
print("remap time : ",end_time - start_time)

# -------------------- #

undistort time  :  0.53125
remap time      :  0.078125
```

약 7배의 시간이 차이가 나고 있는 것을 확인할 수 있다.

<br>

<br>

- projection points

가지고 있는 object point를 통해 image point로 투영한 후 xy,yz,zx 평면을 표시해본다.

```python
for rvec, tvec, image_path in zip(rvecs, tvecs, image_path_list):
    # read image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # draw frame coordinate
    image = cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.2, 3)

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
```

<img src="/assets/img/dev/week12/day5/axes.png">

이미지를 불러와 그에 맞는 좌표축을 그리는 코드이다.

<br>

```python
# B. projection frame points
## object points -> [X, Y, Z]
## camera coordinate
# X -> down
# Y -> right
# Z -> forward

# points for XY plane, YZ plane, XZ plane
xy_points, yz_points, zx_points = list(), list(), list()

# each 1cm point, total size = 1meter
point_size = 0.01

for i in range(len(image_path_list)):
    # xy plane
    xy_point = list()
    x,y,z = 0,0,0

    for _ in range(100):
        y = 0
        for _ in range(100):
            point = [[x, y, z]]
            y += point_size
            xy_point.append(point)
        x += point_size
    xy_points.append(xy_point)

    # yz plane
    yz_point = list()
    x,y,z = 0,0,0

    for _ in range(100):
        z = 0
        for _ in range(100):
            point = [[x, y, z]]
            z += point_size
            yz_point.append(point)
        y += point_size
    yz_points.append(yz_point)

    # zx plane
    zx_point = list()
    x,y,z = 0,0,0

    for _ in range(100):
        x = 0
        for _ in range(100):
            point = [[x, y, z]]
            x += point_size
            zx_point.append(point)
        z += point_size
    zx_points.append(zx_point)
```

object point를 image point로 투영하고자 하는데, object point는 단순히 3차원의 0.01단위로 구성된 각각의 평면을 의미한다. 


```python
xy_points = np.asarray(xy_points, dtype=np.float32)
yz_points = np.asarray(yz_points, dtype=np.float32)
zx_points = np.asarray(zx_points, dtype=np.float32)

# BGR
blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)
```

opencv에 집어넣기 위해서는 numpy로 변환해야 하기 때문에 numpy로 변환해주고, plane을 표시할 색깔을 지정해둔다.

<br>

```python
def projection_points(image, object_points, rvec, tvec, camera_matrix, dist_coeffs, color):
    print(object_points.shape, object_points[:5]) # (10000, 1, 3)
    image_points, jacobians = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    print(image_points.shape, image_points[:5]) # (10000, 1, 2)
    for image_point in image_points:
        image_point = image_point[0]

        x = image_point[0]
        y = image_point[1]

        if x > 0 and y > 0 and x < image_width and y < image_height:
            image = cv2.circle(image, (int(x), int(y)), 1, color)

    return image
    

for rvec, tvec, image_path, xy, yz, zx in zip(rvecs, tvecs, image_path_list, xy_points, yz_points, zx_points):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    image = projection_points(image, xy, rvec, tvec, camera_matrix, dist_coeffs, blue_color)
    image = projection_points(image, yz, rvec, tvec, camera_matrix, dist_coeffs, red_color)
    image = projection_points(image, zx, rvec, tvec, camera_matrix, dist_coeffs, green_color)


    image = cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.03, 3)
    
    cv2.imshow(window_name, image)
    if cv2.waitKey(0) == 27: break

# ------------------------ #
[[[0.   0.   0.  ]]
 [[0.01 0.   0.  ]]
 [[0.02 0.   0.  ]]
 [[0.03 0.   0.  ]]
 [[0.04 0.   0.  ]]]

[[[244.59253   94.102356]]
 [[244.73958  106.93609 ]]
 [[244.92354  119.808784]]
 [[245.14368  132.70506 ]]
 [[245.39908  145.6097  ]]]
```

projectPoints 함수를 통해 3차원 좌표를 2차원 image point로 변환한다. 변환된 이미지의 shape은 (number of points, list, point size(u,v))로 구성되어 있다. 이를 출력해보면 3차원 좌표에 맞는 캘리브레이션 보드 위의 좌표를 의미한다. object point는 실제 world에 대한 3차원 좌표, image point는 가상의 카메라가 캘리브레이션 보드 좌상단의 위치를 기준으로 한 2차원 좌표이기 때문이다.

<br>

2차원 좌표로 변환된 점들을 image에 표기하는데, 만약 0보다 작거나 이미지 크기보다 크다면 필터링을 걸어준다. 이에 대해 각각 xy, yz, zx 평면을 모두 표시해준 후 맞게 그려졌는지 확인을 위해 축도 함께 그려준다. 

<img src="/assets/img/dev/week12/day5/projectionpoints1.png">
<img src="/assets/img/dev/week12/day5/projectionpoints.png">

첫번째 사진은 `squared_size = 0.025`로 설정한 것이고, 두번째 사진은 `squared_size = 0.25`로 준 사진이다. 즉 squared_size를 어떻게 주냐에 따라 각 point들의 scale이 달라진다.

<br>

<br>

# Camera Extrinsic Calibration

extrinsic이란 외부, 즉 위치와 자세를 의미한다. extrinsic calibration은 카메라가 실제로 존재하는 3차원 공간에 대한 정보를 다룬다. 월드 좌표계(3차원)에서 이미지 좌표계(2차원)으로 투영된 후 다시 월드 좌표계로 변환된다. 이 extrinsic calibration은 두 가지 방법으로 활용될 수 있다. **sensor fusion**을 위한 정보로 활용하거나 **perception application**을 위한 정보로 활용할 수 있다. extrinsic calibration은 환경과 조건에 따라 목적과 방법이 달라진다. 그래서 다양한 방법론을 소개하고 자신에게 주어진 환경에 맞는 방법을 찾는 것이 중요하다.

<br>

일반적으로 자동차 좌표계라고 하는 정보를 통일하기 위한 좌표계가 존재한다. 이는 후륜 구동축의 중심 지면을 원점으로 하고, 회전 방향은 다음과 같다.

<img src="/assets/img/dev/week13/day1/rollpitchyaw.png">

차량에 장착되는 센서는 종류가 다양하다. 카메라, Lidar, gps, imu 등이 있다. 이들은 각각의 좌표계를 따른다.

- 카메라 : 카메라 좌표계를 따름
- LIDAR : LIDAR 좌표계를 따름
- GPS, IMU : 각 센서의 좌표계를 따름

각 센서의 장착 위치도 다르고, 사용하는 좌표계도 다르기 때문에 이를 통합하는 과정이 필요하다. 다양한 좌표계를 통합하기 위해 기준 자동차 좌표계를 사용하는 것이고, 통합하는 과정을 sensor fusion이라 한다. 

<br>

각 센서의 장착 위치와 자세를 파악해야 하고, 카메라의 경우 extrinsic matrix의 추출값인 rvecs, tvecs를 사용한다. 이 때의 rvecs, tvecs는 카메라의 위치를 기준으로 한 값들인데, 이를 자동차 좌표계이 기준이 되도록 변경을 해야 한다. 이는 calibration 코드에 적용을 할 때 자동차 좌표계에 대한 정보를 입력으로 넣으면 자돋차 좌표계에 대한 rvec, tvec를 얻을 수 있다.

<img src="/assets/img/dev/week13/day1/car.png">

자동차를 위에서 바라본 그림이다. 주황색 사각형이 카메라를 의미하고, 보라색은 캘리브레이션 보드, 초록색, 노란색이 후륜축과 객체와의 거리를 측정하기 위한 장치이다. 카메라의 높이가 h라고 가정하고, 캘리브레이션 보드의 패턴 시작이 A distance와의 직교하는 위치에서의 높이 H, 보드 중심에서 패턴 시작 위치까지의 거리가 n라 가정한다. 그렇다면 자동차 좌표계에서의 패턴 시작 좌표는 (A distance, n, H) 이 될 것이다. 그렇다면 패턴들의 object point들은 (A, n, H)를 기준으로 한 값들이 쭉 있을 것이고, 이 값을 카메라 calibration 연산에 넣으면 image point들과 자동차 좌표계 원점이 카메라 위치로 가는 rvecs, tvecs가 출력될 것이다.

쉽게 말해 object points들의 좌표의 기준을 (0,0,0)이 아닌 (A,n,H)로 설정하는 것이다. 이 때는 intrinsic matrix는 변화없이 연산을 해야 한다.

<br>

그래서 object point들을 통해 카메라의 위치와 자세를 계산하기 위해서는 opencv의 `solvePnP` 함수를 사용한다.

```python
retval, rvec, tvec = cv.solvePnP(objpoints, imgpoints, mtx, dist)
```

이 때, 중요한 것은 mtx, dist는 intrinsic matrix로, 이것들을 필요로 한다.

[참고](https://docs.opencv.org/3.4/d5/d1f/calib3d_solvePnP.html)

<br>

여러 개의 패턴을 사용한다고 하면 자동차 좌표계부터 패턴까지의 r,t를 미리 계산을 해놓는다면, 새로운 카메라를 사용하더라도, 또는 새로운 패턴을 사용하더라도 패턴들 사이의 관계만 알고 있다면 패턴을 찍지 않아도 r,t 추정이 가능해진다. 즉, 모든 패턴들을 자동차 좌표계에 대한 rotation, translation을 구하고, 각각의 패턴들 사이의 관계만 구해놓으면, 차량이 그 위치에 들어가서 패턴을 1장만 찍어도 모든 패턴의 r,t를 추정할 수 있다.

<br>

<br>

## camera - LIDAR calibration

<img src="/assets/img/dev/week13/day1/calicalib.png">

extrinsic calibration의 활용 사례로는 camera - LIDAR fusion calibration이다.

<br>

3차원 데이터를 출력하는 LIDAR 센서와 카메라를 data fusion하면 object의 depth 정보를 구할 수 있다. 카메라는 Z축, 깊이 정보가 없기 때문에 이 둘을 융합하기 위해서는 LIDAR의 정보를 카메라 좌표계로 이동시켜야 한다. 이동시키기 위한 rvec,tvec를 구하기 위해 3D object points와 이것에 대응되는 2D object points를 얻는 것이 중요하다. 얻는 방법은 매우 많지만 대표적으로 2가지가 있다. lidar의 데이터는 가로 줄 형태로 표시되기 때문에 객체의 높이 정보를 얻기가 어렵다. 그래서 구멍이 뚫려 있는 사각형이나, 마름모를 사용한다. 

원형을 뚫는 방법에서는 뚫린 곳은 라이다 data에서 구멍 형태로 차이가 있을 것이고, 이 형태를 통해 원형의 좌표를 얻을 수 있다. 이들을 평균내서 원의 중심점을 찾을 수 있다면 3d objpoint와 2d imgpoint를 대칭시킬 수 있다.

라이다로 얻어진 objpoint와 imgpoint로 solvePnP 함수를 사용하면 lidar에서 camera로 이동시키기 위한 rvec,tvec를 계산할 수 있다. 그리고 `projectpoints` 함수를 사용하여 입력으로 lidar 데이터에 대한 objpoint와 lidar-\>camera의 rvec,tvec를 넣게 되면 lidar 데이터와 함께 projection하면 이미지 위에 lidar 데이터를 올릴 수 있다. 중요한 점은 전방에 대한 정보를 투영할 수 있지만, 후방에 대한 정보도 투영이 될 수 있다. 그래서 라이다에 대한 정보를 알맞게 잘라야 정확한 정보를 얻을 수 있다. 또는 라이다는 카메라보다 위에 위치하므로 카메라에서는 가려지는 객체에 대한 정보가 라이다에는 같이 들어 있을 수 있다. 그래서 카메라가 인지하고 있는 객체인지 확인하여 제거하는 것이 중요하다.

<br>

주의할 것은 rvec, tvec를 통해 이미지 데이터로 3차원 공간 정보를 복원할 수는 없다. 무수히 많은 해가 존재하기 때문이다.

<br>

<br>

# Geometrical Distance Estimation Theory

[Vision-based ACC with a Single Camera : Bounds on Range and Range Rate Accuracy](https://ieeexplore.ieee.org/abstract/document/1212895), 28 July 2003

2D를 3D로 변환하는 접근 방법에 대해 설명하는 논문이다.

## Abstract

단안 카메라를 통한 범위(깊이 정보, 거리 등) 추정 방법에 대한 논문이다. 오래된 논문이지만, 지금도 다양한 응용으로 활용되는 방법이다. 이 논문은 단안 카메라(single camera)를 입력으로 하는 Vision-based Adaptive Cruise Control(ACC) 시스템을 설명한다. 특히 범위(깊이 정보, 거리)등에 대한 계산을 설명하고, 이 범위가 이미지의 기하학적 요소가 범위에 미치는 영향에 대해 설명한다. 

## Introduction

기존 시스템의 기본 범위 측정 기술은 LIDAR 및 스테레오 이미지를 포함한 직접 범위 센서를 사용한다. 그러나 본 논문에서는 원근 법칙을 이용하여 간접 범위(범위 정보를 추정하는 방법)만을 이용한 단안 카메라를 사용하여 직렬 생산 ACC 제품을 만들어 거리 제어를 수행하고자 한다. 사람의 시각 시스템은 사람의 손이 닿는 범위 및 더 먼 거리에서 대락적으로 거리를 측정한다. 반면 ACC 어플리케이션은 사람이 시각적으로 측정할 수 없는 100m 정도의 거리에 대해서도 거리 측정이 필요하다. 이 때 RADAR나 LIDAR에 의해 제공되는 거리의 정확도는 거리 제어에 충분하지만, 사람의 시각 시스템인 원근 법칙(law of perspective)만으로도 만족스러운 성능을 보일 수 있음을 보여주고자 한다.

단안 시각 시스템을 사용하기 위해서는 두 가지의 문제점이 존재한다. 첫번째는 대상에 대한 깊이 정보가 부족하다는 것이고, 두번째는 깊이 정보가 패턴 인식 기술(이미지의 특징을 인식)에 크게 의존적이라는 것이다. 이를 해결하기 위한 **원근 법칙**과 **망막 발산**(retinal divergence)만을 사용하여 ACC의 요구 정확도를 만족시킬 수 있을지가 중요하다.

### range

원근법을 사용한다는 것은 **차량의 크기**와 **차량 바닥의 위치**를 활용한다는 것이다. 도로면의 기하학적 정보를 활용하면 더 정확한 결과를 얻을 수 있다. 카메라를 평평한 바닥에 평행하게 맞추고, 카메라로부터 `Z`만큼 떨어진 노면은 이미지에서 높이 `y`에 투영되고, y는 다음과 같이 구할 수 있다.

$$ y = \frac{fH}{Z} $$

이 때, f는 초점거리, H는 카메라의 높이를 말한다.

<img src="/assets/img/dev/week12/day5/figure2.png">

A를 기준으로 하여 B와 C 차량을 탐지한다고 할 때, P는 카메라가 있는 위치를 말하고, f만큼의 거리를 가진 초점거리 뒤에 떨어져 있는 이미지 평면 I는 뒤집힌 상태이고, B의 하단 좌표를 통해 이미지 평면 상에서의 y2 높이를 구할 수 있다. C도 동일하게 y1을 구한다. 그림에서는 y1이 위에 위치하지만, 이는 뒤집혀있는 상태이므로 뒤집으면 y2가 위에 있는 상태로 출력될 것이다.

이 때, 바닥이 평평하고, 광학축이 바닥과 평행하다는 가정하에 `y:f = H:Z` 의 비례식이 가능하므로 간편한 연산이 가능하다. 

<br>

<img src="/assets/img/dev/week12/day5/figure3.png">

실제 주행 중 거리를 추정한 결과를 보여준다. 객체의 위치가 멀수록 이미지 평면 상에서 원점과 가깝다. $ y = \frac{fH}{Z} $ 식을 사용하여 $$ Z = \frac{fH}{y} $$에서 Z를 추정한다.

<br>

그러나 다음과 같은 상황에서는 오차가 발생할 수 있다.

- 카메라의 광학 축이 노면과 평행할 수 없다 -\> 수평선이 이미지 중앙에 위치할 수 없다.
- 카메라 장착 각도와 차량의 움직임(pitch)에 의해 변화가 생길 수 있다.
- 이러한 오차를 보정하더라도 차량이 노면과 닿는 지점의 이미지 좌표를 결정하는데 문제가 발생할 수 있다. 그 이유는 거리 정보는 원래는 실수지만 이미지 좌표는 정수이기 때문이다. $ Z_{err} $를 연산해보면 다음과 같다. 

$$ Z_{err} = Z_n - Z = \frac{fH}{y + n} - \frac{fH}{y} $$

$$ = \frac{-nfH}{y^2 + yn} = \frac{-nfH}{\frac{f^2H^2}{Z^2} + n\frac{fH}{Z}} = \| \frac{-nZ^2}{fH + nZ} \| $$

이 때, $ Z_n $ 은 우리가 예측한 값을 의미하고, Z는 실제 값을 의미한다. n은 y값에서 조금의 오차가 발생한 픽셀값으로 2라면 실제값이 100px일 때, 추정값이 98px 또는 102px라고 예측한 값을 의미한다. 이 오차가 생기는 원인은 두 가지가 있다. 하나는 float 에서 int로 변환하면서 생기는 오차일 것이고, 두번째는 bbox를 잘못 추정했을 때 생기는 오차일 것이다. n은 대체로 1로 가정하여 사용한다.

식을 보면, $Z^2$이므로 먼 거리의 값일수록 더 큰 오차가 발생한다는 것을 알 수 있다.

- f = 740pixel
- H = 1.2m
- n = 1 pixel error
- Z_err/Z = 5% error

일 때, Z를 구해보면, 일단 `fH`에 비해 Z는 작은 값이므로 생략을 했을 때 $ Z_{err} = \frac{Z^2}{fH} , Z = 0.05 * 740 * 1.2 = 44m $

5%의 오차를 가지는 지점이 45m정도이고, 90m에서는 10%정도의 오차를 가진다. 얼핏보면 커보이는 수치이지만, ACC를 설계하는 과점에서 거리만을 따지는 것이 아닌 거리의 비율과 상대 속도로 판단하기 때문에 이정도의 오차는 괜찮다고 한다.

<br>

### range rate

range rate에서는 scale 변경에 대한 계산 방법과 이산 시간 차이를 계산하는 방법을 소개한다. 

간단한 수식을 통해 속도를 구한다. $ v = \frac{\triangle Z}{\triangle t} $

<br>

- **Scale**

scale이란 w의 너비를 가지는 차량 두개가 Z, Z'에 대해 w, w'로 표현된다면, 다음과 같은 식을 따른다. $ w = \frac{fW}{Z} , w' = \frac{fW}{Z'} $

이 때, v는 

$$ v = \frac{\triangle Z}{\triangle t} = \frac{Z' - Z}{\triangle t} = \frac{Z \frac{w - w'}{w'}}{\triangle t} $$

이다. 이 때, s를 $ \frac{w - w'}{w'} $ 라고 한다면 v는 $ v = \frac{Zs}{\triangle t} $ 이 된다. scale이 필요한 이유는 같은 너비의 객체가 가까울 때와 멀 때에 대한 좌표들이 선형적으로 비례되는 관계가 아니기 때문이다. 이들의 관계를 연산하기 위해서는 비선형적인 matrix를 구해야 하고 이를 위해 scale을 사용한다.

<br>

- **Error**

scale의 변화는 t, t' 두 지점에서 촬영한 이미지에서 차량을 정렬하여 계산할 수 있다. 이 때 사용하는 다양한 이미지 정렬 기술이 있는데, 이미지 패치(객체에 대한 사각형)이 수 백 픽셀이라면 0.1 pixel의 정렬 오류가 발생할 수 있다. 예를 들어 이미지에서 75m 거리에 있는  소형 자동차는 15x15 픽셀로 표현되는 경우가 있다. 즉 매우 작은 객체로 표현되어 있다는 것이다. 이 때 차량을 정렬하는 오류가 발생할 수 있다. 오류는 이미지에서 대상의 크기에 따라 달라진다. 따라서 대상의 크기에 대한 값인 스케일 오류(S_acc)를 차량 이미지 너비로 나눈 정렬 오류(S_err)로 정의한다.

$$ s_{acc} = \frac{s_{err}}{w} = \frac{s_{err}Z}{fW} $$

<br>

그렇다면 아까 구한 v를 v_err에 대한 값으로 생성할 수 있다. 범위(거리, Z)가 정확하다면 상대 속도 오류(v_err)는 다음과 같다.

$$ v_{err} = \frac{Zs_{acc}}{\triangle t} = \frac{Z^2 s_{err}}{fW \triangle t} $$

이 때 중요한 점은
1. 상대 속도 오류는 상대 속도와 무관하다.
2. 상대 속도 오차는 거리 제곱에 따라 증가한다.
3. 상대 속도 오류는 시간대에 반비례한다. 즉 시간 간격이 더 멀리 떨어져 있는 이미지를 사용하면 더 정확한 상대 속도를 얻는다.
4. 시야가 좁은(초점 거리가 큰 카메라)를 사용하면 오류가 줄어들고, 정확도가 선형적으로 증가한다.

상대 속도 오류가 작으면 상대 속도를 예측할 정확도가 높아진다는 것을 의미한다.

<br>

이제 상대 속도가 아닌 속도 오류(v_zerr)를 계산하고자 한다.

$$ Z_{err} \approx \frac{nZ^2}{fH} (where  n \approx 1) , v = \frac{Zs}{\triangle t} $$

$$ v_{zerr} = \frac{Z_{err}s}{\triangle t} = \frac{nZ^2}{fH} \frac{s}{\triangle t} = \frac{nZv}{fH} $$

이 때, 속도 오류를 고려한 거리 오류는 다음과 같다.

$$ v_{err} = \frac{Z^2s_{err}}{fH \triangle t} + \frac{nZv}{fH} $$

<br>

예를 들어 한 차량에 대해 
- Z = 30m
- f = 740pixels
- W = 1.2
- h = 1.2
- v = 0
- $\triangle t $ = 0.1s 

라고 한다면 v_err는

$$ v_{err} = \frac{Z^2s_{err}}{fH \triangle t} + \frac{nZv}{fH} = \frac{30^2 * 0.1}{740 * 1.2 * 0.1} + \frac{30 * 0}{740 * 1.2} = 1m/s $$

따라서 30m거리에 있는 차량에 대한 상대 속도 오류는 1m/s 정도 오류가 날 수 있다. 이 때, △t를 증가시키면 v_err가 감소된다. 비슷한 움직임을 하는 차량이라면 오랜 시간 동안 차량을 검출할 수 있다.

<br>

그러나 △t가 무한정 늘어난다고 해서 지속적으로 결과가 좋아지는 것은 아니다. 일정한 가속도의 경우 위치와 속도는 다음과 같이 표현된다.

$$ Z(\triangle t) = \frac{1}{2}a\triangle t^2 + v\triangle t + Z_0$$

$$ v(\triangle t) = a\triangle t + v_0 $$

여기서 a는 상대 가속도를 의미하고, 두 시간대에서 거리 차이를 계산하고 두 시간대로 나눈다.

$$ \triangle Z = Z(\triangle t) - Z_0 = \frac{1}{2}a\triangle t^2 + v_0\triangle t $$

$$ \frac{\triangle Z}{\triangle t} = \frac{1}{2}a\triangle t + v_0 -\> v $$

<br>

마지막 식에 따르면 속도가 무한정 올라가면 속도도 무한정 증가되어야 한다. 그러나 속도가 충분히 커지면 오류가 커지게 된다. 따라서 v_err 식에 추가 항을 생성한다.

$$ v_{err} = \frac{Z^2s_{err}}{fH \triangle t} + \frac{nZv}{fH} + \frac{1}{2}a\triangle t $$

<br>

이렇게 하면 △t에 의존하는 항이 두 개가 되고, 첫번째 항은 △t에 반비례, 두번째 항은 비례한다. 그래서 위의 식을 미분하여 v_err가 최소가 되는 최적의 △t를 찾는다.

$$ -\frac{Z^2s_{err}}{fW\triangle t^2} + \frac{1}{2}a = 0 $$

$$ \triangle t = \sqrt{\frac{2Z^2 s_{err}}{fWa}} $$ 이므로, 이를 원래의 v_err에 대입한다.

<br>

$$ v_{err} = Z\sqrt{\frac{2as_{err}}{fa}} + \frac{nZv}{fH} $$

<br>

정리하면
1. 최적의 △t는 거리 Z에 선형 속도 오류를 가진다.
2. 가속도가 0인 경우 최적 △t는 무한대이다. 따라서 실제 시스템에서는 `△t = 2s`로 제한한다.

<br>

<br>