---
title:    "[데브코스] 7주차 - DeepLearning Multi Layer Perceptron"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-04-14 14:40:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, deeplearning]
toc: true
comments: true
math: true
#image:
#  src: /assets/img/dev/week7/day4/main.png
#  width: 500
#  height: 500
---

<br>

# 신경망

사람의 뉴런의 집합을 신경망이라 한다. 뉴런은 두뇌의 가장 작은 정보처리 단위이고, 세포체는 연산을하고, 수상돌기는 신호를 수신, 축삭은 처리 결과를 전송한다. 이러한 사람의 뉴런을 따럿 컴퓨터에 인공지능으로 구성하여 퍼셉트론, 인공신경망을 만들었다. 세포체, 수상돌기, 축삭, 시냅스가 인공신경망에서 각각 노드, 입력, 출력, 가중치에 해당한다.

<br>

인공 신경망에서는 다양한 분류 방식에 따라 많은 종류가 있다. 신호 처리 방식에 따라 전방 신경망/순환 신경망로 나뉠 수 있고, 신경망의 깊이에 따라 얕은 신경망/깊은 신경망로 나뉠 수 있다. 또는 결정론적 신경망과 확률론적 신경망으로도 나뉠 수 있다.

- 결정론 신경망
모델의 매개변수와 조건에 의해 출력이 완전히 결정되는 신경망을 말한다. 즉, 동일한 입력에서는 반드시 동일한 출력이 나오는 신경망이다.

- 확률론 신경망
고유의 임의성을 가지고 매개변수와 조건이 같더라도 다른 출력을 가지는 신경망이다.

<br>

# 퍼셉트론

<img src="/assets/img/dev/week9/day4/perceptron.png">

node, weight, layer 과 같은 새로운 개념의 구조를 도입한 모델이다. 처음으로 `학습`이라는 알고리즘을 제안했다. 현재의 딥러닝 모델은 퍼셉트론을 깊게 만들어서 결과를 도출하는 것이므로 중요한 기반이 되는 모델이다.

## 퍼셉트론의 구조

1. 입력층
퍼센트론을 포함한 모든 모델은 d차원의 입력 벡터를 받는다. i번째 노드는 특징 벡터 $ x = (x_1, x_2, \cdots , x_d)^T $ 의 요소 x_i를 담당한다. 항상 1이 입력되는 평향(bias) 노드도 있다. 이 bias을 통해 임계점이 0으로 옮겨진다.

2. 연산(입력과 출력 사이)
i번째 입력 노드와 출력 노드를 연결할 때 가중치 w_i를 가진다. 퍼셉트론은 단일 층 구조로 간주한다.

3. 출력층
계단함수를 사용했으므로 한 개의 노드에 의해 수치(+1 or -1)을 출력한다.

## 퍼셉트론의 동작

선형 연산과 비선형 연산이 있다. 선형 연산에는 입력값과 가중치가 곱해지고 모두 더하는 연산이고, 비선형 연산에는 활성함수(계단함수)가 이에 해당된다. 

$ s = w^Tx + w_0 $

편향항은 b또는 w_0라고 표기하는데, 이를 w안의 벡터로 추가하게 되면, $ x = (1,x_1,x_2, \cdots, x_d)^T, w = (w_0,w_1,w_2,\cdots,w_d)^T $ 가 된다. 따라서 이를 간단하게 만들면 다음과 같은 식이 만들어진다.

$ y = \tau(w^Tx) $

<br>

### OR 논리 게이트

OR 논리 게이트란 입력값이 (0,0)만 -1, 나머지인 (1,0),(0,1),(1,1) 은 모두 1로 출력되는 퍼셉트론을 말한다. 그렇다면 AND 논리 게이트는 (1,1)은 1이고, 이외에는 다 -1이 출력될 것이다.

<img src="/assets/img/dev/week9/day4/orgate.png">
<img src="/assets/img/dev/week9/day4/andgate.png">

첫번째 그림은 OR, 두번째 그림은 AND 논리 게이트에 대한 좌표 그림이다.

결정 직선 d(x)는 $ d(x) = d(x_1,x_2) = w_1x_1 + w_2x_2 + w_0 = 0 $ 이다. 이 때, w_1,w_2는 직선의 기울기, w_0는 절편(편향)을 결정한다. 결정 직선이란 특징 공간을 두 부분 공간으로 이분할 하는 분류기 역할을 하는 직선이다. 이 때는 결정 직선이 (0,0)이외에는 다 1이 나오기 때문에 (0,0)과 나머지를 구분해야 한다.

식에서 0이 나오는 이유는 원래는 $ w_1x_1 + w_2x_2 = -w_0 $ 와 같이 w_0가 임계값에 대한 값인데, 이를 좌항으로 옮긴 것이다. -w_0인 이유는 임계값을 0으로 좌표를 옮기기 위해서이다.

이를 d차원 공간으로 일반화하면 $ d(x) = w_1x_1 + w_2x_2 + \cdots + w_dx_d + w_0 = 0 $

<br>

AND 논리 게이트를 코드화하면 다음과 같다. 하지만 이는 벡터가 아닌 상수값으로 만들어진 것이다.

```python
def AND(x1,x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
```

theta는 편향값을 말한다.

상수가 아닌 벡터로 표현하게 되면 다음과 같다.

```python
import numpy as np

# 함수로 표현
def AND(x1,x2):                 # 모두 1일때만 1, 나머지는 0
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):               # 모두 1일때만 0, 나머지는 1
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # and와 부호만 반대
    b = 0.7                    # and와 부호만 반대
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):                 # 1이 존재하면 1, 나머지는 0
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])   # and와 가중치만 다르다.
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

<br>

<br>

## 퍼셉트론의 학습

### 목적함수

- 목적함수 상세 설계

$ J(w) = \sum_{x_k in Y} -y_k(w_Tx_k) $

이 때, Y는 w가 틀리는 샘플의 집합, 즉 오판단하는 샘플의 집합을 말한다. 이 식이 퍼셉트론의 목적함수로 적합한지를 판단해보면

- 임의의 샘플 x_k 가 Y에 속한다면 퍼셉트론의 예측값 $w^Tx_k$와 실제값 y_k는 부호가 달라야 한다. 즉 정답이 1인데, 예측값이 -1이거나, 답이 -1인데 예측값이 1이므로 실제값과 예측값은 항상 부호가 다르다.
- Y가 크다는 것은 틀리는 개수가 크다는 것이고, Y가 커질수록 J(w)가 커진다.
- Y가 공집합, 즉 틀린 것이 없다면 J(w) = 0 이다

따라서 목적함수로 사용이 가능하다.

<br>

### 경사하강법

J(w)의 기울기를 이용하여 최소값을 찾는다. 이를 위해 가중치를 갱신할 때는 $ \theta = \theta - \rho g $를 사용하는데, 이 때 $\theta$는 w와 같다. 경사도 g를 얻기 위해 w_i에 대해 편미분을 한다.

$ \frac{\partial J(w)}{\partial w_i} = \sum_{x_k \in Y} \frac{\partial(-y_k(w_0x_{k0} + w_1x_{k1} + \cdots + w_ix_{ki} + w_dx_{kd}))}{\partial w_i} = \sum_{x_k \in Y} -y_kx_{ki} $

w_i는 w벡터의 i번째 요소이므로 가중치 w는 $ w = {w_0,w_1,\cdots,w_i,\cdots,w_d} $ 이다. 그렇기 때문에

가중치 갱신 식은 $ w_{i+1} = w_i + \rho\sum_{x_k \in Y} y_kx_{ki} $ 가 된다. 이를 `델타 규칙`이라 부른다. 델타 규칙은 퍼셉트론의 학습 방법에서만 해당된다. 여기서 $ \rho $는 learning rate(학습률)이다. 이 learning rate는 학습을 할 때 가장 중요한 파라미터이다. 이 값이 너무 크면 최저점을 넘어서서 계속 진동이 되기도 하고, 너무 작으면 지역 최저점에 갇혀버리거나 너무 느리게 학습이 될 수도 있기 때문이다.

>결정 직선과 가중치 w_k는 서로 직각을 이룬다.

<br>

퍼셉트론의 한계가 존재한다. 데이터가 직선을 통해 이분화가 된다면 퍼셉트론을 사용하면 되지만, 거의 모든 데이터는 직선으로 분류하기 어렵다. 이를 해결하기 위해 다층 퍼셉트론을 만들게 된다.

<br>

<br>

# 다층 퍼셉트론

XOR 논리 게이트가 다층 퍼셉트론에 해당된다. 즉, 하나가 1이고, 하나가 0인 상태만 1을 출력한다.

| --- | --- | --- |
| x1 | x2 | output |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |

<img src="/assets/img/dev/week9/day4/xorgate.png">

이 상황에서는 선형 분류기로는 한계가 존재한다.

다층 퍼셉트론의 특징
- **은닉층**을 둔다. 즉 입력층, 출력층 이외에 중간에 층이 하나 더 존재한다.
- **시그모이드 활성함수**를 사용한다. 기존의 퍼셉트론은 계단함수로 활성함수를 만들었지만, 이 함수는 적절하지 않다. 그래서 다층 퍼셉트론에서는 시그모이드함수를 활성함수로 사용한다. 출력을 신뢰도로 간주함으로서 더 융통성 있게 의사결정이 가능해졌다.
- **오류 역전파 알고리즘**을 사용한다. 여러 층이 순차적으로 이어져 있기에 역방향을 진행하면서 한층씩 그레디언트를 계산하고 가중치를 갱신한다.

<br>

원래의 퍼셉트론을 2개를 병렬 결합하면 원래 공간 $ x = (x_1,x_2)^T $ 를 새로운 특징 공간 $ z = (z_1, z_2)^T $ 로 변환이 된다. 즉, 새로운 특징 공간 z에서 선형 분리가 가능해진다. 

<img src="/assets/img/dev/week9/day4/mlp.png">

<br>

## 다층 퍼셉트론의 용량

특징 공간 x가 2개의 벡터를 가진다면, x1,x2의 2차원 공간에서 w1,w2를 통해 직선의 방정식을 그릴 수 있다. 이를 새로운 영역 z에 점으로 투영한다면 3차원의 점으로 변환될 것이고, 층을 지날수록 더 높은 차원으로 변환이 될 수 있다. 

그렇다면, 3개의 퍼셉트론을 결합한 경우 2차원 공간을 n개 영역으로 나누고, 각 영역을 3차원 점으로 변환하고, 계단함수를 활성함수로 가정한다면 3차원에 점으로 변환된다. 따라서 p개의 퍼셉트론을 결합하면 p차원 공간으로 변환되는 것과 같다. 

<br>

하나의 은닉층은 특징 공간을 다른 특징 공간으로 매핑하는 것이므로 함수의 근사표현으로 생각해볼 수 있고, 이 은닉층을 여러 개 쓴다면 내가 원하는 공간을 변환하는 근사 함수라고 표현할 수 있다. 

## 활성함수

활성함수가 원래는 계단함수였으나 좀 더 부드러운 공간 분할을 위해 시그모이드 함수로 바꾸었다. 그렇게 되면 원래의 계단함수는 영역을 점으로 변환하지만, 시그모이드와 같은 함수들은 영역을 영역으로 변환한다. 

<img src="/assets/img/dev/week9/day4/sigmoid.png">

<br>

활성함수로 많이 사용되는 함수들은 다음과 같다.

<img src="/assets/img/dev/week9/day4/function.png">

시그모이드나 tanh는 a가 커질수록 계단함수에 가까워진다. 다층 퍼셉트론에서는 sigmoid나 tanh를 많이 사용했고, 딥러닝에서는 마지막에 ReLU를 가장 많이 사용한다. 딥러닝에서 sigmoid나 tanh를 사용하지 않는 이유는 그래프의 모양과 같이 0이나 1에 가까운 값을 내는 값들에 대해서는 gradient를 없앤다.(= vanishing gradient)

<br>

<img src="/assets/img/cs231n/2021-09-22/0018.jpg">

예를 들어 데이터 x와 출력이 존재할 때 sigmoid를 사용한 모델에서 backpropagation를 위해 gradient를 계산한다면, 일단 (dL/dσ)가 있을 것이고, sigmoid gate를 지나 local sigmoid function의 gradient인 (dL/dx)를 구하기 위해 chain rule를 적용한다.

따라서 $ \frac{\partial L}{\partial x} = \frac{\partial σ}{\partial x} * \frac{\partial L}{\partial σ} $ 가 된다.

<br>

x = 0 일 때는 backprop가 잘 진행될 것이다. 그러나 x = -10 일 때의 gradient($\frac{\partial L}{\partial x} $)는 그림에서 보듯이 `0`이다. 그렇게 되면, 0이 backpropagation 될 것이고, 그로 인해 뒤에 전달되는 모든 gradient는 모두 죽어(=0)버린다. x = 10 일 때도 마찬가지로 모든 gradient가 0 이 된다.

따라서 **x가 아주 크거나 아주 작다면 gradient가 계속 0으로 죽어버린다.** 이를 gradient vanish 또는 vanishing gradient라 한다.

> [참고 자료](https://dkssud8150.github.io/posts/cs231n6/)

사실 실제 뉴런의 출력도 sigmoid보다 ReLU의 형태가 더 가깝다. 따라서 ReLU를 많이 사용한다.

<br>

## 매개변수

은닉 층이 1개인 퍼셉트론을 2층 퍼셉트론, 은닉층이 2개인 퍼셉트론을 3층 퍼셉트론이라 한다. 은닉층이 4개 이상인 퍼셉트론을 깊은 신경망이라 한다. 은닉층의 개수가 너무 많으면 과잉적합(overfitting), 너무 적으면 과소적합(underfitting)이 발생한다.

은닉층이 기하학적으로 봤을 때는 새로운 특징 공간으로 변환해주는 것과 같지만, 의미론적으로 바라봤을 때 이는 입력 벡터에 대해 내가 원하는 부분을 뽑아내겠다는 특징 추출기라 할 수 있다. 입력이 들어오면 내가 만들어놓은 가중치, 즉 벡터 값들과 곱해서 원하는 결과를 얻어내는 것이다. 예를 들어, 내가 이미지를 모델에 넣었을 때 학습된 가중치들과 곱해져서 컴퓨터가 해당 이미지의 특징들이 추상화된 형태가 될 것이다. 

현대 기계학습에서는 이를 `특징학습`이라 부른다. 

<br>

<br>

## 오류 역전파 알고리즘

### 손실함수

가장 일반적인 손실함수는 MSE(Mean Squared Error)이 있다. L2 norm을 사용해서 다음과 같이 정의된다.

$ e = \frac{1}{2n}\sum_{i=1}^n || y_i - o_i ||_2^2 $

여기서 o는 출력 벡터이고, y는 실제값인데, 이를 원핫 인코딩 형태로 만들어 `부류 벡터`라는 벡터를 만든다.. 즉, 0과 1로만 구성되어 있는데, 기댓값 즉 내가 원하는 값인지 아닌지에 대한 값을 나타내는 것으로 이를 통해 error를 구한다. 

<br>

### 연산 그래프 (computational graph)

연산을 그래프로 표현한 것을 연산 그래프라 한다.

<img src="https://media.vlpt.us/images/guide333/post/c2e92214-64a1-44ca-b7ec-b303f6866e9f/Screenshot%20from%202021-01-11%2023-58-02.png">

<br>

연산 그래프를 진행할 때, 그래디언트를 계산하기 위해서는 연쇄 법칙을 사용한다. 예를 들어, k(x) = f(g(h(i(x)))) 일 때, k'(x)를 구하기 위해서는 다음과 같은 식이 될 것이다.

$$ i\prime(x) = f\prime(g(h(i(x)))) * g\prime(h(i(x))) * h\prime(i(x)) * i\prime(x) $$

<br>

그렇다면 우리가 전방 계산을 할 때마다 prime을 구해놓으면 그것을 통해 역전파를 하면 된다.

동일한 f(활성함수)를 연속으로 사용하고, 입력이 w -\> x -\> y -\> z 일 때의 그래디언트를 구하면 

$ \frac{\partial z}{\partial w} = \frac{\partial z}{\partial y} \frac{\partial y}{\partial x} \frac{\partial x}{\partial w} $
$ = f\prime(y)f\prime(x)f\prime(w) $
$ = f\prime(f(f(w)))f\prime(f(w))f\prime(w) $

<img src="/assets/img/dev/week9/day4/chainrule.png">

<br>

우리가 실제로 궁금한 미분은 가중치에 대한 예측값의 그래디언트이다. 2층 퍼셉트론에서의 매개 변수 $ \theta = {U^1, U^2} $ 가 있다고 하면, 손실 함수 J($\theta$) 는 다음과 같다.

$ J(\theta) = \frac{1}{2}||y - o(\theta)||_2^2 $

이 때, y는 부류 벡터, o는 예측값이다. $J({U^1, U^2}) $의 최저점을 찾기위해 이를 미분하면

$ U^1 = U^1 - \rho \frac{\partial J}{\partial U^1} $
$ U^2 = U^2 - \rho \frac{\partial J}{\partial U^2} $

연산의 순서는 x -\> U^1 -\> U^2 -\> o 이다.

<br>

<img src="https://media.vlpt.us/images/lilpark/post/999cfcfb-b53f-40f4-b872-42f42da67a39/image.png">

다시 한 번 살펴보자면, 우리가 궁금한 것은 가중치 x,y 에 대한 결과값 L의 그래디언트이다. 여기서 upstream은 뒤로 올라가는 방향의 gradient, downstream은 전방으로 전달되는 방향으로의 gradient이다. 

$ \frac{\partial L}{\partial x} $ 은 chain rule에 의해 분해가 되어 $ \frac{\partial L}{\partial z} \frac{\partial z}{\partial x} $ 로 변환될 수 있다. z도 가중치이므로 x 와 z를 연산하여 출력이 되고, 그 출력이 L이 된다. y도 동일하다.

forward, backprop 두 가지를 따로 바라보게 되면

- forward

입력값 `in`, 이 활성함수 `f`를 거쳐 `out`이 된다. 

<br>

- backward
이에 대해 미분을 해보면

$ \frac{\partial \epsilon}{\partial in} = \frac{\partial \epsilon}{\partial out} \cdot \frac{\partial out}{\partial in} = \frac{\partial \epsilon}{\partial out} \cdot f\prime(in)$

이 때, $\epsilon$ 은 error이고, $ \frac{\partial \epsilon}{\partial out} $ 은 output gradient이다. $ \frac{\partial out}{\partial in} $ 은 local gradient이다. 

그래서 입력에 대한 에러의 gradient는 output gradient(출력에 대한 에러의 gradient) * local gradient(입력에 대한 출력의 gradient) 이 된다. local gradient는 out = f(in) 이므로 in에 대한 out의 미분은 $f\prime(in)$ 이 된다.

<br>

#### 곱셈의 역전파

forward의 식은 다음과 같다.

$ out = in_1 \cdot in_2 $

<br>

이에 대해 역전파를 진행해보면

$ \frac{\partial \epsilon}{\partial in} = \frac{\partial \epsilon}{\partial out} \cdot \frac{\partial out}{\partial in} = \frac{\partial \epsilon}{\partial out} \cdot in_2$

in_2가 되는 이유는 곱셈의 연산을 생각해보면 `X * Y`에서 X에 대해 미분을 하면 `Y`, Y에 대해 미분하면 `X`이다. 따라서 in_1에 대해 미분을 하면 in_2가 남는다.

<br>

이 연산을 코드로 구현하면 다음과 같다. 이는 pytorch에서 이미 구현되어 있다. 각 연산마다의 grad_z 즉 output gradient를 저장해놓고 이를 backprop에 사용한다.

```python
class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x,y)
        z = x * y
        return z
    
    @staticmethod
    def backward(ctx, grad_z):
        x,y = ctx.saved_tensors
        grad_x = y * grad_z # dz/dx * dL/dz
        grad_y = x * grad_z # dz/dy * dL/dz
        return grad_x, grad_y
```

<br>

#### 덧셈의 역전파

- forward

$ out = \sum_i in_i $

- backpropagation

$ \frac{\partial \epsilon}{\partial in_i} = \frac{\partial \epsilon}{\partial out} \cdot 1 = \frac{\partial \epsilon}{\partial out}$

덧셈의 경우 `X+Y`를 X, Y 각각에 대해 미분하면 모두 1이 나온다. 

<br>

#### S형 활성함수의 역전파

$ \frac{\partial \epsilon}{\partial in} = \frac{\partial \epsilon}{\partial out} \cdot \sigma \prime(in) = \frac{\partial \epsilon}{\partial out} \cdot [\sigma(in) (1 - \sigma(in))] $

시그모이드 함수의 형태를 생각해보면 다음과 같다. 

<img src="/assets/img/dev/week9/day4/sig.jpg">

이를 미분하면 $ [\sigma(in) (1 - \sigma(in))] $ 이 만들어진다.

<br>

#### 최대화 역전파

$ out = max_i\{in_i\} $

max{0,x} 일 경우, 0보다 작으면 0이 되고, 0보다 크면 output gradient를 전달해준다.

<br>

ReLU가 이에 해당되는데, ReLU와 같은 형태는 미분을 하면 임계값까지는 0이다가, 임계값 이후에는 1이 나온다. 그래서 local gradient는 1 또는 0이 된다.

<br>

#### 전개(fanout) 역전파

<img src="/assets/img/dev/week9/day4/fanout.jpg">

fanout이란 두 가지의 출력이 존재하고, 이 둘을 합쳐서 또 다른 출력을 생성한다.

$ x = x(t), y = y(t) => z = f(x,y) $

이 때는 역전파를 하기 위해서는 x,y 방향 각각을 gradient를 구해서 더하면 된다.

$ \frac{\partial z}{\partial t} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial t} +  \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial t} $

<br>

위는 두 가지의 출력이지만, 이를 일반화 시키게 되면

$ z = f(n_1,n_2, \cdots , n_k, \cdots) , and , n_k = n_k(in)$

의 형태라면 이를 미분하게 되면 다음과 같다.

$ \frac{\partial z}{\partial in} = \sum_k \frac{\partial z}{\partial n_k} \cdot \frac{\partial n_k}{\partial in} = \sum_{k} \frac{\partial z}{\partial in_k} $

<br>

#### 신경망 역전파

<img src="/assets/img/dev/week9/day4/backprop.jpg">

위의 덧셈, max, 곱셈에 대한 역전파를 모두 사용한다. 여기서 gradient vanish가 발생하는 이유는 중간에 시그모이드 함수가 있을 때, 값이 너무 크거나 너무 작으면 gradient가 0으로 나오고, 그 뒤로는 계속 0이 나오게 된다.

<br>

<img src="https://media.vlpt.us/images/lilpark/post/fc4c568a-d8d0-4e3c-8b29-8d851742f4b5/image.png">

<br>

<br>

도함수의 종류를 바라보면 3가지가 있다

1. scalar to scalar

스칼라에 대한 스칼라는 상수가 나온다.

2. vector to scalar

출력값은 하나(L)지만, 입력값이 $ x = (x_1,x_2,x_3, ...)^T $ 이라면 이에 대한 미분을 **gradient**라 한다. vector 각각이 영향을 주고, 그에 대해 L이 달라지므로 1~N개의 미분이 다 나올 것이고, 이는 벡터 형태이다.

이 때 표기를 $ \nabla_x z $ 라 한다. 즉, 스칼라 z에 대한 벡터 x의 미분 벡터이다.

3. vector to vector

출력도 벡터, 입력도 벡터라면 y1에 대한 모든 요소 x1,x2,x3..이 있고, y2도 x1,x2,x3 에 대한 미분이 다 있을 것이다. 그렇기에 출력되는 값은 행렬로 표현된다. 이 행렬을 **Jacobian**이라 한다.

$$ \frac{\partial y}{\partial x} = [\frac{\partial y_1}{\partial x_1} \cdots \frac{\partial y_1}{\partial x_m} ]
= [\frac{\partial y_1}{\partial x_1} \cdots \frac{\partial y_1}{\partial x_m} \vdots \ddots \vdots \frac{\partial y_n}{\partial x_1} \cdots \frac{\partial y_n}{\partial x_m}] $$

<br>

만약 퍼셉트론이 2층이 있어서 x -\> y -\> z 인데, x가 vector, y도 vector, z 가 scalar이면 $ \frac{\partial y}{\partial x} $ 는 야코비안, $ \frac{\partial z}{\partial y} $는 gradient 형태이다.

<br>

위의 vector는 다 2차원이지만, 실제 딥러닝에 사용되는 차원은 3차원이다. 미니 배치를 사용해서 3차원을 만든 형태를 집어넣는데, 이 때문에 연산량이 배치 단위에 따라 달라지는 것이다.

<br>

<br>

# MLP 코드 구현

### Pytorch

- autograd
pytorch의 autograd라는 패키지는 텐서의 모든 연산에 대한 자동 미분을 제공한다.

- tensor
    - torch.tensor 클래스에는 required_grad 속성이 있는데, 이를 true로 설정하면 해당 텐서에서 이루어진 모든 연산을 추적한다.
    - 계산이 완료된 후 backward()를 호출하면 모든 그래디언트를 자동으로 계산하며 이 그래디언트는 .grad 속성에 누적된다.
    - tensor가 기록 추적하는 것을 멈추게 하려면 해당 줄에 .detach()를 사용하거나 .with torch_no_grad()를 사용한다. 이를 사용하는 이유는 모델을 추론할 때 사용하거나 numpy로 변환할 때는 grad를 추적하지 않아야 하기 때문이다.
    - 각 tensor는 .grad_fn 속성을 가지고 있는데, 이는 tensor를 생성한 Function을 참조한다. 그러나 사용자가 만든 tensor는 예외고, 사용자가 만들지 않은 tensor에서의 연산으로 생긴 텐서는 모두 Function을 참조한다.
    - 도함수를 계산할 때는 .backward를 호출한다.

<br>

```python
import torch

x = torch.ones(2, 2, requires_grad = True)

print(x)

print(x.grad_fn)

# -----------------------------#

tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
None
```

사용자가 직접 선언해준 tensor이므로 None으로 출력된다.

```python
y = x + 2

print(y.grad_fn)

# -----------------------------#

<AddBackward0 object at 0x7ff8f2d294d0>
```

선언해주지 않았지만, tensor의 연산에 의해 만들어진 y는 True, 즉 requires_grad가 True로 설정된다.

requires_grad를 True로 설정했기 때문에 x의 grad_fn은 True로 출력된다.


```python
print(x.requires_grad)

x.requires_grad_(False)

print(x.requires_grad)

# -----------------------------#

True
False
```

`requires_grad_` 메서드를 통해 requires_grad를 변경시켜줄 수 있다.

<br>

```python
x.requires_grad_(True) # 위에서 False로 설정했으므로 다시 True로 변경

z = y**3
out = z.mean()

y.retain_grad()
z.retain_grad()
out.backward()

print(x.grad)
print(y.grad)
print(z.grad)
print(x.is_leaf) # 이것이 leaf 노드인지 확인 -> 즉 가장 끝단의 노드인지 확인

out.backward()

# -----------------------------#

tensor([[6.7500, 6.7500],
        [6.7500, 6.7500]])
tensor([[6.7500, 6.7500],
        [6.7500, 6.7500]])
tensor([[0.2500, 0.2500],
        [0.2500, 0.2500]])
True

RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
```

중요한 것은 backward()를 여러 번 하려면 retain_graph = True로 설정해줘야 한다. 그렇지 않으면 에러가 난다.

<br>

```python
x = torch.ones(2, 2, requires_grad = True)
y = x + 2

z = y**3
out = z.mean()

y.retain_grad()
out.backward(retain_graph=True)

out.backward() 

print(x.grad)
print(y.grad)
print(z.grad) 

# -----------------------------#

tensor([[13.5000, 13.5000],
        [13.5000, 13.5000]])
tensor([[13.5000, 13.5000],
        [13.5000, 13.5000]])
None
```

z.retain_grad() 를 호출하지 않으면 grad를 저장하지 않으므로 grad가 없다.

<br>

```python
x = torch.randn(3, requires_grad=True)
print(x)
y = x * 2
v = torch.tensor([0.1,1.0,0.0001], dtype=torch.float)
y.backward(v) # 아무것도 지정하지 않으면 default로 1이 들어가서 값이 추출된다.

print(x.grad) # 미분 후에 결과값

# -----------------------------#

tensor([ 0.5778,  1.1385, -0.4793], requires_grad=True)
tensor([2.0000e-01, 2.0000e+00, 2.0000e-04])
```

<br>

#### 신경망 구현하기

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.l1 = nn.Linear(4, 128)
    self.l2 = nn.Linear(128, 64)
    self.l3 = nn.Linear(64, 32)
    self.l4 = nn.Linear(32, 16)
    self.l5 = nn.Linear(16, 3)

    self.bn1 = nn.BatchNorm1d(128) # batch normalization
    self.bn2 = nn.BatchNorm1d(64)
    self.bn3 = nn.BatchNorm1d(32)

    self.act = nn.ReLU()            # activation function

  def forward(self, x):
    x = self.act(self.bn1((self.l1(x))))
    x = self.act(self.bn2((self.l2(x))))
    x = self.act(self.bn3((self.l3(x))))
    x = self.act(self.l4(x))
    x = self.l5(x)

    return x

criterion = nn.CrossEntropyLoss()

x,y = torch.randn([4,4]), torch.tensor([1,0,2,0])

net = Net()
output = net(x)
loss = criterion(output, y)
print(loss.item())

net.zero_grad()         # 저장되어 있는 grad를 다 지워야 한다.
print(net.l5.bias.grad) # zero grad 했으므로 none

print(net.l5.bias.is_leaf) # leaf노드란 아래 자식 노드가 없는 노드

loss.backward()

print(net.l5.bias.grad)     

# ----------------------------- #

1.0462257862091064
None
True
tensor([-0.1753,  0.1098,  0.0655])
```

역전파를 진행해주었기 때문에 grad가 기록되어 있다.

<br>

```python
params = list(net.parameters())
print(len(params)) # 각 층마다 weight 와 bias
print(params[0].size()) #  0번째 층의 weight 와 bias
 
# ----------------------------- #

16
torch.Size([128, 4])
```

각 층마다의 weight와 bias를 저장해놓고 있으며, 이를 호출하면 모든 층의 weight와 bias를 리턴할 수 있다. 각 벡터마다 측의 weight를 저장하고 있다.

<br>

##### iris 데이터를 불러와서 MLP 모델에 학습시키기

```python
from torch.utils.data import DataLoader, TensorDataset
import pasdas as pd

from sklearn.datasets import load_iris

dataset = load_iris()
data = dataset.data
label = dataset.target

print(label)

# ----------------------------- #

.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
                
    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================
...

```

`from sklearn.datasets import load_iris`에는 여러 데이터셋이 저장되어 있다. `dataset.DESCR`을 하면 description 즉 설명들이 저장되어 있다.

target은 정답 라벨을 의미한다.

<br>

```python
print(data.shape)
print(label.shape)

# ----------------------------- #

(150, 4)
(150,)
```

150개의 데이터가 있고, 각각의 4개의 속성을 가지고 있는 것을 확인할 수 있다.

<br>

```python
from sklean.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.25)

print(x_train.shape)
print(x_test.shape)

# ----------------------------- #

(122,4)
(38,4)
```

train_test_split은 데이터셋을 분리해주는 기능을 한다. test_size를 통해 test 데이터셋의 크기를 지정해준다.

<br>

```python
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()

y_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()

train_dataset = TensorDataset(x_train, y_train)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle = True)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.l1 = nn.Linear(4, 128)
    self.l2 = nn.Linear(128, 64)
    self.l3 = nn.Linear(64, 32)
    self.l4 = nn.Linear(32, 16)
    self.l5 = nn.Linear(16, 3)

    self.bn1 = nn.BatchNorm1d(128)
    self.bn2 = nn.BatchNorm1d(64)
    self.bn3 = nn.BatchNorm1d(32)

    self.act = nn.ReLU() 

  def forward(self, x):
    x = self.act(self.bn1((self.l1(x))))
    x = self.act(self.bn2((self.l2(x))))
    x = self.act(self.bn3((self.l3(x))))
    x = self.act(self.l4(x))
    x = self.l5(x)

    return x


optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 20

losses = []
accures = []

for epoch in range(epochs):
  epoch_loss = 0
  epoch_accur = 0
  for batch, (x, y) in train_loader:
    optimizer.zero_grad()

    output = net(x)

    loss = criterion(output, y)
    loss.backward()

    optimizer.step()

    _, prediction = torch.max(output, dim=1)
    accur = (prediction == y).sum().item()
    epoch_loss += loss.item()
    epoch_accur += accur

  epoch_loss /= len(train_loader)
  epoch_accur /= len(x_train)

  
  losses.append(epoch_loss)
  accures.append(epoch_accur)
```

- iris로 데이터를 불러오면 numpy로 되어 있으므로 이를 tensor로 변환한다. 
-dataloader이라는 함수를 통해 학습에 필요한 차원으로 변경시킨다. batch_size를 지정해주고, shuffle이란 데이터 샘플의 순서를 섞을지 말지에 대한 인자이다. 
- 최적화 알고리즘은 SGD(stochastic gradient descent)를 사용했다. 
- 손실함수로는 crossentropy를 사용했다. 
- epoch이란 학습 반복 횟수를 지정해주는 것이다.

반복을 하면서 에측에 대한 loss를 구하고, 역전파를 진행하고, 그에 대해 가중치를 최적화 한다. 그리고 출력은 [4,1]과 같은 형태로 출력이 되는데, 이는 확률이다. 특정 라벨에 대한 출력이므로 여기서 가장 큰 것을 예측 라벨이라고 지정해준다. 그것이 정답과 같은지를 다 비교해서 더하면 정확도가 나올 것이다. 마지막에는 평균 loss와 accuracy를 알아야 하므로 데이터 샘플 개수만큼 나눠준다. 

<br>

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))
plt.subplots_adjust(wspace=0.2)

plt.subplot(1,2,1)
plt.title("loss")
plt.plot(losses)
plt.xlabel("epochs")

plt.subplot(1,2,2)
plt.title("accuracy")
plt.plot(accures)
plt.xlabel("epochs")
```

결과를 그래프로 보면 다음과 같다.

<img src="/assets/img/dev/week9/day4/matplot.png">

<br>

- inference

```python
output = net(x_test)
print(torch.max(output, dim=1))
_, prediction = torch.max(output, dim=1)
accuracy = round((prediction = y_test).sum().item() / len(y_test),4)

print(round(accuracy,4))

# -----------------------------#

torch.return_types.max(
values=tensor([0.2726, 1.2251, 0.5746, 0.2792, 0.4174, 1.6891, 0.2693, 1.3956, 0.2245,
        1.5383, 0.5694, 0.3931, 1.6798, 0.4401, 0.8937, 1.5574, 1.4948, 0.4101,
        2.0672, 0.9091, 0.3115, 0.8186, 1.6430, 1.2262, 1.7812, 0.5254, 1.0002,
        0.0993, 0.4572, 2.2780, 0.4420, 0.5622, 0.4398, 1.8892, 0.9751, 1.6513,
        0.4286, 1.4288], grad_fn=<MaxBackward0>),
indices=tensor([1, 0, 2, 2, 1, 0, 1, 0, 2, 0, 2, 1, 0, 1, 2, 0, 0, 1, 2, 2, 1, 2, 2, 2,
        0, 1, 2, 1, 1, 0, 1, 2, 1, 0, 2, 0, 2, 0]))

0.9211
```

정확도가 0.92가 나왔다. 모델의 구성에 비해 너무 높게 나왔긴 했다. 

<br>

<br>

### Tensorflow

- Autograd

- 그래디언트 테이프
텐서플로우는 자동 미분을 위한 `tf.GradientTape` API를 제공한다. 이는 컨텍스트 안에서 실행된 모든 연산을 테이프에 **기록** 한다.

<br>

```python
import tensorflow as tf

x = tf.ones((2,2))

with tf.GradientTape() as t:
  t.watch(x) # x값을 본다.
  y = tf.reduce_sum(x)
  print('y: ' ,y)
  z = tf.multiply(y, y)
  print("z :", z)

dz_dx = t.gradient(z, x)
print(dz_dx)

for i in [0,1]:
  for j in [0,1]:
    assert dz_dx[i][j].numpy == 8.0 ## 값이 틀릴 경우 assertionerror가 발생

dz_dy = t.gradient(z, y)
print(dz_dy)
assert dz_dy.numpy() == 8.0 

# ----------------------------- #

y : tf.Tensor(4.0, shape=(), dtype=float32)

z : tf.Tensor(16.0, shape=(), dtype=float32)

tf.Tensor(
[[8. 8.]
 [8. 8.]], shape=(2, 2), dtype=float32)

tf.Tensor(8.0, shape=(), dtype=float32)
```

이 때, gradient를 호출하면 gradienttape에 포함된 리소스가 해제된다. 따라서 여러 그레디언트를 계산하려면 다음과 같이 정의해야 한다.

```python
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x) 
    y = x * x
    z = y * y # z = x^4

dz_dx = t.gradient(z, x)
dy_dx = t.gradient(y, x)

print(dz_dx, "\n", dy_dx)

# ----------------------------- #

tf.Tensor(108.0, shape=(), dtype=float32) 
tf.Tensor(6.0, shape=(), dtype=float32)

del t
```

반복적으로 본 후에는 삭제를 꼭 해주어야 한다.

<br>

#### 고계도(Higher-order) 그래디언트

gradientTape 컨텍스트 매니저 안에 있는 연산들은 자동 미분을 위해 기록된다. 만약 이 컨텍스트 안에서 그래디언트를 계산하면 해당 그레디언트 연산 또한 기록된다.

```python
x = tf.Variable(1.0)

with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x**3
    dy_dx = t2.gradient(y, x)
d2y_dx2 = t1.gradient(dy_dx, x)

print(dy_dx.numpy())
print(d2y_dx2.numpy())

# ----------------------------- #

3.0
6.0
```

<br>

### 신경망 구현하기

sequential을 사용했을 때의 코드는 다음과 같다.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
     # 
     layers.Dense(2, activation='relu', name='layer1'),
     layers.Dense(3, activation='relu', name='layer2'),
     layers.Dense(4, name='layer3'),
    ]
)

x = tf.ones((3,3))

y = model(x)
print(y)

# ----------------------------- #

tf.Tensor(
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]], shape=(3, 4), dtype=float32)
```

<br>

3x3 행렬이 노드가 2개인 층 -\> 3개인 층 -\> 4개인 층을 거쳐 출력값이 한 벡터당 4개로 출력이 되는 것을 볼 수 있다.

<br>

sequential을 사용하지 않고 층을 쌓을 수 있다.

```python
layer1 = layers.Dense(2, activation='relu', name='layer1')
layer2 = layers.Dense(3, activation='relu', name='layer2')
layer3 = layers.Dense(4, name='layer3')

x = tf.ones((3,3))
y = layer3(layer2(layer1(x)))
print(y)

# ----------------------------- #

tf.Tensor(
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]], shape=(3, 4), dtype=float32)
```

<br>

Sequential을 사용하는 것이 좋아보인다. 여기서 add 함수를 사용하여 층을 쌓을 수 있다.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(3, activation='relu'))
model.add(layers.Dense(4))

x = tf.ones((3,3))

y = model(x)
print(y)

# ----------------------------- #

tf.Tensor(
[[-0.6980011 -1.1421962  0.5842113  1.1211115]
 [-0.6980011 -1.1421962  0.5842113  1.1211115]
 [-0.6980011 -1.1421962  0.5842113  1.1211115]], shape=(3, 4), dtype=float32)
```

<br>

add가 되는 것처럼 `pop` 메서드도 사용이 가능하다.

```python
model.pop()
print(len(model.layers))
2
```

<br>

##### 패션 MNIST 사용한 분류

패션 MNIST데이터에는 10개의 카테고리와 70000개의 흑백이미지가 포함되어 있다. 이미지의 해상도는 28x28이다. 훈련 데이터셋은 6만장, 테스트 데이터셋은 1만장을 사용한다.

```python
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

# ----------------------------- #

(60000, 28, 28)
```

```python
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.show()
```

<img src="/assets/img/dev/week9/day4/mnist.png">

<br>

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

신경망 모델에 주입하기 전에 값의 범위를 0~1로 normalize한다.

```python
plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(train_images[i],cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])

plt.show()
```

정규화된 이미지를 여러 장 본다.

<img src="/assets/img/dev/week9/day4/subplot.png">

<br>

```python
model = keras.Sequential([
          keras.layers.Flatten(input_shape=(28,28)),
          keras.layers.Dense(128, activation='relu'),
          keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# ----------------------------- #

Epoch 1/5
1875/1875 [==============================] - 7s 3ms/step - loss: 0.4982 - accuracy: 0.8246
Epoch 2/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.3744 - accuracy: 0.8662
Epoch 3/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.3359 - accuracy: 0.8782
Epoch 4/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.3115 - accuracy: 0.8861
Epoch 5/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2932 - accuracy: 0.8922
<keras.callbacks.History at 0x7ff871842590>
```

모델은 이와 같고, 입력을 위해 이미지 행렬을 벡터 형태로 편다. 그리고 마지막에는 확률값 출력을 위해 softmax를 사용했다.

compile은 모델에 도구들을 지정해준다. adam 이외에 SGD 등을 사용할 수 있다. loss에는 예측값이 정수값으로 나올 경우 sparse_categorical_crossentropy를 사용한다고 한다.

그 후 fit을 통해 실제 학습이 진행된다.

<br>

케라스에는 모델을 시각화하는 함수가 있다.

```python
keras.utils.plot_model(model, show_shapes=True)
```

<img src="/assets/img/dev/week9/day4/plot_model.png">

<br>

- validation

모델을 통해 검증을 한다.

```python
test_loss, test_acc= model.evaluate(test_images, test_labels, verbose=2)

print("test loss : ", test_loss)
print("test accuracy : ", test_acc)

# ----------------------------- #

313/313 - 1s - loss: 0.3475 - accuracy: 0.8763 - 792ms/epoch - 3ms/step
test loss :  0.3474982678890228
test accuracy :  0.8762999773025513
```

- inference

그 후 평가를 진행한다.

```python
prediction = model.predict(test_images)
prediction[0]

np.argmax(prediction[0]) # 9
np.argmax(prediction[0]) == test_labels[0] # True

# ----------------------------- #

array([2.8452354e-09, 9.4890495e-09, 9.5172211e-08, 4.2439359e-09,
       1.4116972e-07, 2.8057180e-03, 4.7427403e-07, 1.1753553e-02,
       1.5509060e-05, 9.8542446e-01], dtype=float32)

9
True
```

prediction에는 모든 10개의 신뢰도를 나타낸다. 그 후 argmax를 사용하면 가장 예측값이 높은 레이블의 인덱스가 출력된다. 그것을 test_label과 비교하여 정답인지 확인한다.