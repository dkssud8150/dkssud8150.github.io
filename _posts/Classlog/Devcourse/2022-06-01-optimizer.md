---
title:    "[데브코스] 16주차 - Visual-SLAM MAP and Non-linear Optimizer, Loop Closure "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-06-01 19:20:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, Visual-SLAM]
toc: true
comments: true
math: true
---

<br>

이때까지는 Projection에 대해 공부했다. 이제 SLAM으로 다시 돌아와보자.

<br>

# Least Squares

SLAM이란 Simultaneous Localization And Mapping으로 동시적 위치 추정 및 지도 작성이다. 최소자승법(Least Squares)은 SLAM 문제를 풀 때 사용하는 최적화 방법론이다. 최적화란 어떤 방정식에서 완벽하게 떨어지는 정답값을 찾지 못할 때 에러가 가장 작은 값을 찾아내는 기법이다. least squares는 에러에 제곱 연산을 통해 에러가 크면 클수록 기하급수적으로 커지게 된다. 

least square는 Over-determined-system에 해당하는데, 즉 미지수의 개수보다 방정식의 개수가 더 많은 시스템을 의미한다. SLAM에서는 항상 over-determined-system에 해당한다. 그 이유는 뽑고자 하는 feature의 수보다 센서 데이터가 항상 많기 때문이다.

원래의 GT값과 비교해서 우리가 추정한 값과의 차이를 error라 하고, 이 error들의 합이 최소가 되도록 하는 값들을 찾는 것이 최적화라 할 수 있다.

<br>

## Maximum-a-posteriori estimation is SLAM

sensor데이터에는 2가지 종류가 있다. *proprioceptive sensing*, *exteroceptive sensing*이고, 각각에 대한 수식은 **motion model**, **obsevation model**이 있다. 먼저 전자의 수식에 대해 알아보자.

<br>

- motion model

<img src="/assets/img/dev/week16/day3/motion_model.png">

motion model은 현재 위치에 대한 state를 추정하는 수식이고, 현재 state, $ x_k^r $에 대한 값은 다음과 같다.

$ x_k^r = \begin{bmatrix} x^r(k) \\ y^r(k) \\ \theta^r(k) \end{bmatrix} $

$ x_k^r $ 는 지난 프레임에서의 state, $ x_{k-1}^r $와 관측한 odometry 또는 control 값, motion noise값 3개를 입력으로 받아 구한다.

<br>

- observation model

<img src="/assets/img/dev/week16/day3/observation_model.png">

observation model은 landmark의 위치에 관한 state를 추정하는 수식이고, landmark와의 거리에 대한 값 $ d_{z_k} $, 각도에 대한 값 $ \theta_{z_k} $의 수식은 다음과 같다.

$ d_{z_k} = \sqrt{(x^r(k) - l_i^x)^2 - (y^r(k) - l_i^y)^2} + \epsilon_z $ 

$ \theta_{z_k} = atan2(l_i^y - y^r(k), l_i^x - x^r(k)) - \theta^r(k) + \epsilon_z $ 

이 때, observation model에 대한 값은 motion model의 값 $ x_k^r $과 $ l_i, \epsilon_z $을 입력으로 하여 구한다. 이 때, $ l_i $는 차량에 대한 상대적인 위치를 나타내고, $ \epsilon_z $는 노이즈에 해당한다.

거리는 간단한 거리를 구하는 수식에 노이즈를 더한 값이고, 각도의 경우도 x방향, y방향으로의 삼각함수를 사용하여 구한다. 

<br>

<br>

SLAM state의 관점으로 본다면, SLAM에서의 현재 State는 1개의 robot pose와 다수의 관측 가능한 landmark의 위치를 담고 있을 것이다. 다음 프레임으로 가면 1개의 SLAM state가 하나 더 생길 것이고, 여러 개의 SLAM state를 통해 노이즈를 제거해줄 수 있다. 그러나 여러 개의 state가 생기면 노이즈도 그만큼 증가한다. 따라서 시간이 지날수록 노이즈가 증가해서 Uncertainty가 증가한다.

<br>

증가하는 노이즈의 패턴 분석을 통해, motion model과 observation model을 기반으로 노이즈 누적에도 안정적인 최적의 robot pose와 최적의 landmark위치를 추정하고자 한다. 

지금까지 쌓아온 데이터의 확률 분포와 현재 설정한 최적의 파라미터에 대해 실제 데이터가 믿음직한지에 대한 판단을 통계학적으로 *belief*라고 한다. 

센서 데이터는 관측 후에 나타나는 사후 데이터이므로 이 사후 데이터를 가장 잘 표현하는 확률을 찾는 것, 즉 최대 사후 확률(Maximum a posteriori, MAP)를 찾는 것이 최종적인 목표이다. 다르게 표현하면 지금까지의 motion model과, observation model에 대한 확률 분포를 찾고, 그 확률 분포를 정확하게 표현하는 **state** 값을 찾는 것이 목표이다. 그를 위해 노이즈를 가장 최소가 되는 확률 분포를 찾아야 한다.

<br>

<br>

MAP estimation을 푸는 방법을 알아보자. 최적의 state, `x`를 구해야 한다.

- motion model

proprioceptive 데이터(odometry)를 u = {u_1,u_2,...,u_k}라 하고, motion model을 `f`라 했을 때, 현재 state, `x_k`는 다음과 같다. 

$ x_k = f(x_{k-1}, u_k, \epsilon_u) $

이 때, $\epsilon_u$는 노이즈인데, 우리가 하고자 하는 것은 노이즈가 가장 작은 확률 분포를 찾아야 하므로 0이라 가정하고 식을 세우면 다음과 같다.

$ x \approx f(x_{k-1, u_k}) =\> x_k - f(x_{k-1}, u_k) \approx 0 $ (1)

<br>

- observation model

exteroceptive 데이터(landmark의 상대적 위치)를 z = {z_1,z_2,...,z_k}라 하고, observation model을 `h`라 했을 때, landmark와의 거리와 방향, `z_{k,j}`는 다음과 같다.

$ z_{k,j} = h(x_k, l_j, \epsilon_z) \; =\> \; z_{x_k,l_j} \approx h(x_k, k_j) \; =\> z_{k,j} - h(x_k,l_j) \approx 0 $ (2)

<br>

이 둘의 식은 결국 error값을 의미하므로, 이 두 식을 활용하여 **최적의 state**를 찾는다.

$$ argmin_{x_0,...,x_k} \sum_{i=1}^k \| x_i - f(x_{i-1}, u_i) \|^2 + \sum_{(i,j) \in \beta } \| z_{i,j} - h(x_i, l_j) \|^2 $$

least square 기법을 사용하기 때문에 각 error에 제곱했다. 위의 식을 통해 error의 최소값이 아닌 최소가 되는 state를 구한다.

<br>

<br>

# Graph-based SLAM

SLAM에는 두가지 방법론이 있다.
1. Incremental SLAM (i.e. Online SLAM)
- Filter이 적용된 SLAM
  - Particle filter, Kalman filter

2. Batch SLAM  (i.e. Offline SLAM)
  - Graph-based SLAM

<br>

## Incremental SLAM

Incremental SLAM의 가장 큰 특징은 가장 최근 state만 추정한다는 것이다. Markov chain assumption을 기본 전제로 하여 바로 이전의 state와 observation 정보만 있으면 새로운 state를 계산한다. 즉 새로운 state는 이전 state에 의존한다.

따라서 계산이 간단하므로 굉장히 빠르게 연산이 된다. 가장 최신 정보를 출력하기 때문에 실시간으로 사용할 수 있다 하여 *online SLAM*이라고도 불렸다.

<br>

## Batch SLAM 

Incremental SLAM과는 달리 Batch SLAM은 여러 시점에서의 state들을 한번에 추론한다. 그래서 전체적인 연산량이 증가하여 과거에 *offline SLAM*이라고 불렸다. 하지만 현대에는 다양한 트릭을 사용해서 Batch SLAM을 실시간으로 사용이 가능하다.

<br>

Incremental SLAM이 Batch SLAM보다 빠르게 동작하지만 정확도가 다소 떨어진다. 그러나 최근들어 한 논문에서 Batch SLAM이 Incremental SLAM보다 빠르다 하여 현재는 거의 대부분이 **Batch SLAM**을 사용한다. 

<br>

<br>

- Factor graph

<img src="/assets/img/dev/week16/day3/factor_graph.png">

graph를 다루는 여러 가지 방식이 연구되다가 **Factor graph**방식이 생겨났다. Factor graph는 노드와 edge로 이루어져 있는 graph로, 노드에는 robot state나 landmark state가 저장되고, edge에는 motion model 정보나 observation model 정보가 저장된다. 이 edge에 저장되는 형태를 `Factor`라 부른다.

Factor graph는 특정 노드에 연결되는 Factor들의 error를 통해 최적화한다. 

<br>

- Graph-based SLAM

<img src="/assets/img/dev/week16/day3/graph_based_slam.png">

Factor graph의 사용보다 least squares 를 쉽게 적용할 수 있는 방식으로 **Graph-based SLAM** 방식이 생겨났다. graph로 표현함으로써 얻게 되는 장점은 robot pose와 observation 정보를 쉽게 파악할 수 있다는 점이고, 그로 인해 graph안에 loop가 생겨난다. 이 loop를 최적화함으로써 loop속에 누적되는 uncertainty를 제거하고, loop안에 있는 모든 node에 대한 최적의 값을 찾을 수 있게 되었다. loop가 발생하게 되면 error를 해소시켜줌으로써 루프를 완성시켜줄 수 있다. 이를 **Loop closure**이라 한다.

<img src="/assets/img/dev/week16/day3/loop_closure.jpg">

<br>

<img src="/assets/img/dev/week16/day3/graph_based_slam_node.png">

그리고, SLAM에서의 파이프라인이 생겨났다. Front-end에서는 node와 edges를 생성했고, back-end에서는 graph를 최적화했다.

<br>

<br>

# Bundle Adjustment

<img src="/assets/img/dev/week16/day3/ba.png">
[이미지 출처 - 장형기님 블로그](http://www.cv-learn.com/20210313-ba/)

마찬가지로 지난 글에서의 Triangulation을 배웠는데, 이는 2view geometry에 대한 내용이었다. Bundle Adjustment는 한 단계 더 나아가 N-view geometry에 대한 내용이다. N개의 프레임 또는 N개의 카메라가 존재하고, 그에 따른 각각의 Rotation, translation이 존재한다. 이 때, 서로의 2D-2D correspondence를 공유하며, 3d point인 landmark에 대한 거리도 공유하고 있다고 가정한다. 그리고 1개의 landmark마다 2개 이상의 2D-3D correspondence를 가지고 있다.

<br>

모든 값들이 다 계산이 되어 있지만, 매 프레임마다 motion model 정보와 observation model 정보에 노이즈가 계속 누적되므로 이를 처리하기 위해 Batch SLAM기법을 적용한다. 2D-2D correspondence, 2D-3D correspondence 등의 값들을 활용해서 camera pose, 3D landmark position를 보정해주는 작업을 Bundle Adjustment라 한다. 보정을 한다는 것은 graph 최적화를 통해 uncertainty를 해소해준다는 것을 의미한다. graph 최적화를 위해서는 위에서 배웠듯이 motion model에서의 error, observation model에서의 error이 필요하다. VSLAM에서는 이 두가지의 error를 **ReProjection Error** 하나로 간편하게 표현이 가능하다. landmark를 image plane에 재투영했을 때 생기는 error는 pixel 단위로 표현된다.

3D landmark position과 camera pose가 완벽한 값이라고 가정하면 3D landmark를 image plane에 투영했을 때는 정확한 keypoint위치로 맞아떨어지겠지만, 모든 센서는 노이즈를 가지고 있기 때문에 조금의 오차가 발생한다. 이 때 reprojection error에 대한 function이 $ \pi $이고, landmark를 image plane으로 투영한 위치와 원래 keypoint와의 오차를 $ \triangle z_{ij} $에 해당할 때 오차가 제일 작아지는 keypoint는 다음과 같이 표현할 수 있다. landmark에서의 좌표를 P로, image plane에서의 좌표를 C로 표현되어 있다.

$ argmin_x \sum_i \sum_j || x_{ij} - \pi(P_j,C_i) ||_{w_{ij}}^2 $

<img src="/assets/img/dev/week16/day3/reprojection_error.png">

<br>

landmark재투영에 대한 error도 있지만, camera pose나 motion model, observation model에 대한 error도 구해줄 수 있다.

<br>

최적화한다는 것은 결국 total reprojection error를 최소화되는 최적의 camera pose와 landmark를 찾는 것이다. 그러나 image projection 과정이 linear하지 않기 때문에 단순 미분을 통한 최적화가 불가능하다. 그래서 비선형 공간에서 선형 공간으로 근사화시켜서 최적화를 수행하려고 한다. 그 방법으로 `Gauss-Newton method`, `Levenberg-Marquardt method`가 있다. 

<br>

<br>

# Nonlinear optimization

Non-linear 최적화에 사용되는 대표적인 방법들 중 **Gauss-Newton method**를 사용해보고자 한다.

## Gauss-Newton method

우선 BA를 푸는데 필요한 파라미터의 개수는 총 3D landmark position (X,Y,Z = 3) + Extrinsic parameter (Rx,Ry,Rz, tx,ty,tz = 6) + intrinsic parameter (fx,fy, cx,cy,s = 5) + scale factor (1)로 1개의 3D landmark당 15개의 파라미터를 가진다.

이것들을 State 벡터로 표현하면 다음과 같다.

landmark state $ x_l = \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} $

camera state $ x_{cam} = \begin{bmatrix} t_x ,\ t_y, \ y_z, \ R_x, \ R_y, \ R_z, \ \lambda, \ f_x, \ f_y, \ c_{x}, \ c_y ,\ s \end{bmatrix}^T $

x_l은 3D landmark에 대한 값, x_cam은 extrinsic matrix와 scale값, intrinsic matrix로 이루어져 있다.

$ State \; Vector \;\; x = \begin{bmatrix} x_{cam}, \ x_{l_1}, \ x_{l_2}, \cdots \ ,x_{l_n} \end{bmatrix}^T $

<br>

그렇다면 reprojection error를 최소화하는 State vector는 어떤 값을 가지는가에 대한 것이 BA 최적화이고, 이 문제를 풀기 위해 **노이즈가 가우시안 분포**를 가지는 *least squares optimization*을 수행할 것이다. 가우시안으로 가정하는 이유는 가우시안 노이즈 데이터를 사용할 경우 간편한 *Maximum Likelihood Estimation*로 바뀌게 되고, MLE는 최적의 값을 가지는 것이 가능하기 때문이다.

<br>

아까 사용하던 least square 식에 가우시안 노이즈 분포를 추가해준다.

$ argmin_x \sum_i \sum_i \| e_i(x) \| \; =\> \; argmin_x \sum_i \sum_j \| e_i(x)^T \Omega_i e_i(x) \| $

이 때, 오메가는 covariance matrix 또는 information matrix라고 부른다. 

<br>

최적의 값을 x\*라고 했을 때, $ x^{*} = argmin_x E(x) $라 표현하고, 위치 x 에서 x\*로의 변화량을 $ \triangle x $라 했을 때, 

$ \triangle x = x^{*} - x $ 이고, $ argmin_x $ 대신 $ argmin_{\triangle x}$ 에 대한 값으로 정리하면 다음과 같다.

$ x^{*} = argmin_{\triangle x} E(x + \triangle x) $

그런데, $ E(x) = \sum_i \| e_i{x}^T \Omega_i e_i(x) \| $이므로 최종적인 x*는 다음과 같다.

$ x^{*} = argmin_{\triangle x} \sum_i \| e_i(x_i + \triangle x)^T \Omega_i e_i(x_i+\triangle x) \| $

<br>

$ e_i(x_i + \triangle x) $ 가 non-linear하기 때문에 linear하게 근사시키기 위해 미분을 해준다. 미분을 할 때는 테일러 급수를 사용한다.

$ \cfrac{\partial (e_i(x_i + \triangle x))}{\partial x} \approx e_i(x) + J_i \triangle x =\> e_i + J_i \triangle x $

J는 Jacobian을 뜻하고, 간편하게 표현하였다. 그래서 다시 x*에 대한 식으로 넘어와 대입해주면 다음과 같다. 

$ x^{*} = argmin_{\triangle x} \sum_i [e_i + J_i \triangle x]^T \Omega_i [e_i + J_i \triangle x]$

$ x^{*} = argmin_{\triangle x} (\sum_i e_i^T\Omega_i e_i) + 2(\sum_i e_i\Omega_i J_i)\triangle x + \triangle x^T(\sum_i J_i^T \Omega_i J_i)\triangle x $

<br>

이 때, $ (\sum_i e_i^T\Omega_i e_i) = C \;,\; \sum_i e_i\Omega_i J_i = b^T \;,\; \sum_i J_i^T \Omega_i J_i = H $라고 가정한다면,

$ x^{*} = argmin_{\triangle x} C + 2b^T\triangle x + \triangle x^T H \triangle x $

이므로, 이는 $\triangle x$에 대한 2차 방정식으로 간주할 수 있다.

<br>

2차 방정식으로 생각하면 최적화가 간단해진다. 2차방정식은 미분을 해서 0이 되는 값은 최대 또는 최소이다. 그러나 위의 식에서는 항상 최소가 나오게 되어있다.

$ \triangle x$ 에 대해 미분을 하면

$ \cfrac{\partial}{\partial \triangle x} [C + 2b^T \triangle x + \triangle x^T H \triangle x] = 2b + 2H\triangle x = 0 $

따라서 $ \triangle x = - H^{-1}b $가 되고, $ \triangle x $의 의미는 현재 위치 x_0에서 최적의 x로 가기 위한 변화량을 의미한다.

이 때의 최적의 x는 global한 최적값이 아니라 local minia에 해당한다. 위의 식이 의미하는 바는 현재의 값보다 낮은 값으로 찾아갈 것이고, 이를 통해 시간이 지속될수록 점차 minumum에 다가가는 것을 확인할 수 있다.

<br>

그러나 현실적으로는 H matrix가 굉장히 크기 때문에 계산하는 것은 매우 복잡하다. 

<img src="/assets/img/dev/week16/day3/params.png">

이미지 출처 : cyrill stachniss 교수님 lectures

2만장의 이미지가 있을 때, 각각의 이미지마다 18개의 feature를 뽑고, 각각의 landmark는 3번 정도 관찰된다고 가정해보자. 이런 경우 Jacobian matrix 1개가 3.5 x 10^11 개의 값이 존재한다. 이를 inverse하는 것은 불가능하다.

<br>

<img src="/assets/img/dev/week16/day3/sparsity.png">

이미지 출처 : cv-learn.com 블로그 글

그래서 matrix의 특성을 이용해보고자 한다. Jacobian matrix를 살펴보면 대부분의 element가 비어있다. 그 이유는 factor들마다의 연관성을 미분한 matrix인데, node들은 인접한 것과는 연결이 되겠지만, 멀리 떨어져 있는 것과는 거의 연결이 되어 있지 않다.  거의 비어있는 matrix를 sparsity matrix라고 부른다.

<br>

<img src="/assets/img/dev/week16/day3/bH.png">

b matrix의 경우도 Jacobian이 거의 비어있으면 b도 거의 비어있게 출력된다. H matrix도 동일하게 Jacobian matrix가 거의 비어있어서 비어있는 matrix가 형성된다.

그러나 식을 보면 전체의 합으로 구성되어 있으므로 b는 sparsity matrix가 모여 dense한 matrix가 만들어지지만, H는 대칭적인 형태가 구성되어 여전히 sparsity한 특성을 띈다.

<br>

<img src="/assets/img/dev/week16/day3/optimize.png">

H의 sparsity한 특성을 활용하여 inverse matrix를 빠르게 구할 수 있을 것이다. 그러나 좋은 방법은 아니다.

<br>

<br>

## Schur Complement

H matrix에 대해 sparsity한 특성을 활용하는 것이 아닌 H matrix의 형태를 분석해서 연산하는 방법이 있다.

<img src="/assets/img/dev/week16/day3/schur_complement.png">

H matrix의 형태는 4가지 구조로 나뉘어져 있다. 그래서 H를 4가지로 나누어서 표현할 수 있다. A는 Camera에 대한 정보로 H_S, C는 3D structure에 대한 정보로 H_C로 표현된다. B는 H_SC를 의미한다.

$ \begin{bmatrix} H_S \ H_{SC} \\ H_{SC}^T \ H_C \end{bmatrix} \begin{bmatrix} \delta_S \\ \delta_C \end{bmatrix} = \begin{bmatrix} \varepsilon_S \\ \varepsilon_C \end{bmatrix}$

1행의 값들이 3D structure에 대한 정보, 2행의 값들이 Camera Parameter에 대한 정보이다. 그 후 양변에 특정 행렬을 곱해 H_SC^T를 제거한다.

$ \begin{bmatrix} H_S \ H_{SC} \\ H_{SC}^T \ H_C \end{bmatrix} \begin{bmatrix} \delta_S \\ \delta_C \end{bmatrix} \begin{bmatrix} I \ 0 \\ -H_{SC}^TH_S^{-1} \ I \end{bmatrix} = \begin{bmatrix} \varepsilon_S \\ \varepsilon_C \end{bmatrix}\begin{bmatrix} I \ 0 \\ -H_{SC}^TH_S^{-1} \ I \end{bmatrix} $

$ \begin{bmatrix} H_S \ H_{SC} \\ 0 \ H_C - H_{SC}^TH_S^{-1}H_{SC} \end{bmatrix} \begin{bmatrix} \delta_S \\ \delta_C \end{bmatrix} = \begin{bmatrix} \varepsilon_S \\ \varepsilon_C - \varepsilon_S H_{SC}^TH_S{-1} \end{bmatrix}$

이로써 2행의 값들로만 방정식을 풀 수 있게 되었다. 

$ (H_C - H_{SC}^T H_S^{-1}H_{SC})\delta_C = \varepsilon_S H_{SC}^T H_S{-1}$

이 때, 좌항의 괄호 안에 있는 식을 **Schur Complement** 식이라 한다.

H_C와 H_SC는 우리가 가지고 있는 값이다. $ H_S^{-1} $는 안의 블록들이 3x3 diagonal matrix로 이루어져 있어 구하기 쉽다. 그러면 또 다시 `Ax = b` 의 구조를 가지게 된다. 

그러나 좌항의 것들을 그대로 우항을 넘기기는 쉽지 않다. 그래서 Schur complement를 1개의 큰 matrix로 만든 후 inverse 하기 편한 matrix로 분해한다. 분해하는 방법으로는 `LU decomposition` 또는 `Cholesky decomposition`을 많이 사용한다. 

<br>

그 후 이렇게 구한 $ \delta_C $를 통해 $ \delta_S $를 구하면 된다.

$ \delta_S = H_S^{-1}(\varepsilon_C - H_{SC}^T \delta_C)$

여기서 $ \delta_C $의 의미는 우리가 가지고 있는 H matrix의 카메라 파라미터들이 어떻게 바뀌어야 $ \delta_S $와 같은지에 대한 값이고, $ \delta_S $는 3D landmark position이 얼마나 바뀌어야 $ \delta_C $와 같아지는지에 대한 값이다.

<br>

<br>

## Outlier rejection

least squares를 할 때 조심해야 할 부분이 있다. least squares 알고리즘이 outlier에 매우 취약한 알고리즘이다. 따라서 least squares를 사용할 때는 M-estimator 커널과 같은 기법을 사용하여 데이터 분포속에서 outlier를 optimization 식에서 제외시켜야 한다. 

<br>

<br>

### Nonliear optimization Libraries

SLAM에서 optimization을 수행하기 위해 이때까지 배운 복잡한 과정들을 library에 잘 정리되어 있다.

- [Google Ceres-solver](https://code.google.com/p/ceres-solver/)
- [G2o](https://openslam.org/g2o.html)
- [GTSAM](https://collab.cc.gatech.edu/borg/gtsam/)


