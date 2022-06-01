---
title:    "[데브코스] 15주차 - Visual-SLAM Non-linear Optimizer and Loop Closure "
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

Factor graph의 사용보다 least squares 를 쉽게 적용할 수 있는 방식으로 **Graph-based SLAM** 방식이 생겨났다. graph로 표현함으로써 얻게 되는 장점은 robot pose와 observation 정보를 쉽게 파악할 수 있다는 점이고, 그로 인해 graph안에 loop가 생겨난다. 이 loop를 최적화함으로써 loop속에 누적되는 uncertainty를 제거하고, loop안에 있는 모든 node에 대한 최적의 값을 찾을 수 있게 되었다. 

<br>

<img src="/assets/img/dev/week16/day3/graph_based_slam_node.png">

그리고, SLAM에서의 파이프라인이 생겨났다. Front-end에서는 node와 edges를 생성했고, back-end에서는 graph를 최적화했다.

