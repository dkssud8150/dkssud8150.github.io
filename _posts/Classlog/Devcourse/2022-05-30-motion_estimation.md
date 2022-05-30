---
title:    "[데브코스] 15주차 - Visual-SLAM motion estimation "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-05-30 16:20:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, Visual-SLAM]
toc: true
comments: true
math: true
---

<br>

# Epipolor Geometry

3D point estimation을 수행할 때, 왜 2장의 이미지가 필요할까?

카메라 투영 시 3D에서 2D로의 mapping 관계는 이전 글에서 많이 다뤘다. 그러나 2D에서 3D로의 mapping 관계를 풀기 위해 intrinsic과 extrinsic matrix를 활용할 수 있지만, extrinsic을 알지 못하는 경우가 너무 많다. 사실 이 extrinsic matrix를 안다는 것은 실제 3d point를 안다는 것과 같다. SLAM이나 mapping 기술들 대부분이 결국 extrinsic를 구하는 자체가 목적이 된다. 

그렇다면 Depth값을 이미 알고 있다면, SLAM을 굳이 사용하지 않아도 되는 거라 생각할 수 있다. 그러나 Depth카메라의 경우 노이즈가 많이 생겨서 오히려 더 부정확하게 3D point를 추정될 수 있다. 그래서 여러 장의 이미지를 사용하여 노이즈를 제거한다. 

또 그렇다면 Depth값을 정확하게 알고 있다면, SLAM을 사용하지 않아도 되는가? 그건 아니다. SLAM의 목적은 mapping도 있지만, localization도 필요하고, 맵을 확장시키는 기능도 있다.

<br>

<br>

<img src="/assets/img/dev/week16/day1/epipolar.png">

위의 이미지는 2장의 이미지를 사용하고 있는 모습이다. 바라보는 시점은 다르지만, 동일한 3D point를 바라보는 2개의 이미지가 있는 경우, 이를 `2-view geometry`라고 한다. monocular camera의 경우 1개는 previous image, 나머지 1개는 현재 image라 볼 수 있다. stereo camera인 경우에는 동시간대 왼쪽 오른쪽 카메라에 대한 이미지라 생각할 수 있다.

이런 2-view geometry를 다른 말로 **epipolar geometry**라고도 한다. image plane 1에서의 3D point-\> 2D point로의 mapping을 할 때의 projection line이 있고, 이는 위의 그림에서 초록색 선에 해당한다. 이제부터는 초록색 선을 `ray`라 부르고자 한다. Visual SLAM을 수행할 때, 3D point가 2D image point로 projection될 때의 직선을 ray라고 한다. 이 ray를 image plane2에서도 바라볼 수 있을 것이다. ray를 image plane2에 투영되는 것을 **재투영**이라 한다. image plane2안에 그려진 ray를 투영한 직선을 **epipolar line**이라 한다. image plane1과 image plane2가 함께 바라보는 3D point는 무조건 epipolar line위에 존재한다.

image plane1과 image plane2의 rotation과 translatino을 정확하게 구하기 위해서는 image plane1에 존재하는 x와 image plane2에 존재하는 x'의 연관성이 반드시 필요하다.

<br>

<img src="/assets/img/dev/week16/day1/epipole.png">

이번에는 2-view에서 image plane1이 3개의 3D point를 바라보고 있다고 생각해보자. 각각의 point들마다의 ray1,2,3이 존재할 것이고, image plane2에 재투영하면 3개의 epipolar line이 존재한다. 이 3개의 line이 한 점에서 만날 때, 이 점을 **epipole**이라 한다. 이 점은 모든 ray가 시작되는 지점인 *optical center*이다. 간단하게 보면 점들이 3차원으로 뻗어나가고 있을 때, 이 ray들은 한 점에서 만나고, 이를 image plane2에서 바라보고 있으니 epipolar line이 만나는 점은 당연히 ray들이 교차하는 지점인 optical center이다.

즉, epipole은 반대편 이미지의 optical center가 투영된 것이다. 지난 카메라의 위치가 보이는 것이거나 stereo 일 경우에는 옆의 카메라를 바라보고 있다고 생각할 수 있다.

<br>

<img src="/assets/img/dev/week16/day1/equal_direction_2view.png">

만약 카메라가 같은 방향을 바라보고 있다면, 위와 같은 상황이 발생할 것이고, 이 때는 epipole이 존재하지 않고, 따라서 ray1,2,3은 모두 평행하다. stereo 카메라의 경우 calibration과정을 거치고 나서 *rectification*이라는 과정을 수행한다. 이는 카메라의 위치들을 조정해서 모두 같은 방향을 바라보도록 만드는 과정이다. 

이러한 과정을 수행하는 이유는 그림에서 힌트가 있다. 이러한 경우 모든 epipolar line들이 가로로 선이 그어지게 되고, 그렇게 되면 두 이미지의 연관성을 찾고 싶으면 x1,x2,x3 각각의 row를 찾으면 된다. x1에 대한 correspondence를 찾고 싶으면 l1을 탐색하고, x2에 대한 correspondence를 찾고 싶으면 l2를 탐색하면 된다.

대각선의 선에서부터 correspondence를 찾는 것은 오래걸리고 메모리도 많이 차지할 것이다. 그러나 가로선의 경우에는 픽셀의 정보도 붙어있어서 쉽게 매칭을 해줄 수 있다.

<br>

<img src="/assets/img/dev/week16/day1/straight_direction_2view.png">

monocular 카메라에 대해 로봇이나 자동차가 직진을 하고 있다면, 위의 그림과 같은 상황이 발생한다. 이 경우에는 epipole1과 epipole2가 직선에 놓이게 된다. 즉 이미지 상에서의 동일한 위치에 형성될 것이고, epipolar line은 방사 형태가 될 것이다. 이 경우는 굳이 어렵게 correspondence를 구하지 않아도 rotation이 없다는 것을 알 수 있고, translation값만 알면 곧바로 이동거리를 판단할 수 있다.

<br>

<img src="/assets/img/dev/week16/day1/epipolar_plane.png">

다시 epipolar geometry로 돌아와서 좌측 이미지와 우측 이미지의 optical center들과 3D point를 연결하면 삼각형이 만들어진다. 이 삼각형을 **epipolar plane**이라 한다. 이 plane은 epipolar line과 epipole의 정보를 모두 가지고 있다. 그리고 좌측 이미지의 optical center와 우측 이미지의 optical center를 연결하는 직선을 **baseline**이라 한다. 이 직선은 두 카메라 간의 거리값을 표현한다. baseline이 길면 길수록 삼각측량의 정확도가 커지기도 한다.

![image](https://user-images.githubusercontent.com/33013780/170954401-14488fa1-f4b5-4f45-855d-47dbf1dd5877.png)

baseline이 이미지 plane을 통과하면 통과하는 점이 epipole을 이룬다. 즉 epipole들은 baseline 직선 위에 올라가 있다는 것을 알 수 있다. 이미지를 통해 수많은 3D point를 관찰할 수 있다. 3D point, `x`를 위아래로 움직여보면 그에 맞게 epipolar plane이 생기는데, 이들은 baseline을 중점으로 회전을 한다는 것을 확인할 수 있다. 존재할 수 있는 epipolar plane은 굉장히 많을텐데, 이 모든 가능성을 **epipolar pencil**이라 한다.

이 모든 특징들을 조합해서 correspondence가 존재하기 위한 조건들을 3가지로 요약할 수 있다.
1. 3D point, `x`는 epipolar plane위에 있어야 한다.
2. 모든 epipolar line들은 epipole과 교차해야 한다.
3. 모든 baseline은 epipole과 교차해야 하며 epipolar plane은 baseline을 반드시 포함하고 있어야 한다.

이러한 조건들을 기하학적 표현으로 *geometry constraints*라 한다. 어떤 현상이 나타나기 위해서 존재해야 하는 조건들을 의미한다.

<br>

<br>

# Essential / Fundamental matrix

essential matrix와 fundamental matrix는 VSLAM에서 epipolar geometry가 사용되는 개념이다.

e-matrix(essential matrix)는 epipolar constraint에 대한 정보를 담고 있는 3x3 matrix이다. 카메라 간의 모션을 추정하기 위해서는 correspondence를 정확하게 아는 것이 중요하다. 이 correspondence를 정확하게 얻어낼 수 있도록 하는 것이 epipolar constraint이기 때문에, 이 e-matrix를 얻어내는 것이 굉장히 중요하다. **epipolar constraint가 뜻하는 것은 결국 두 카메라간의 회전과 이동을 담고 있다.**

<img src="/assets/img/dev/week16/day1/essential_matrix.png">

$ E = t * R $

$ \widetilde{x}^T E \widetilde{x} = 0 $

E는 essential matrix이고, 이는 2개의 2d point를 이어주는 역할을 한다. `t`는 3x1 matrix translation, `R`는 3x3 matrix의 rotation을 뜻하고, 이 둘은 벡터곱 연산을 수행한다. 벡터곱 연산을 수행하기 위해서는 차원이 같아야 하므로 3x1 translation matrix를 대칭행렬(https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=sw4r&logNo=221357931060) 형태로 바꿔주어 3x3 matrix로 변환시킨다. 결국 essential matrix는 3x3 matrix로 구성될 것이고, 이를 SVD를 통해 t와 R로 분해한다.

<br>

그러나 이 essential matrix를 그대로 사용하려면 한가지 오류가 있다. 이 x가 어떻게 표현되어 있는지 모른다는 것이다. essential matrix를 표현할 때는 image point, `x`는 normalized image plane에서의 점으로 표현한다. 그러나 실제 이미지를 다루고자 할 때는 픽셀로서 좌표를 표현해야 하기에 normalized된 좌표를 픽셀 좌표로 변환해줘야 한다. 이 과정에 필요한 matrix가 instrinsic matrix인거고, instrinsic matrix까지 결합하여 matrix를 표현한 것을 **Fundamental matrix**라 한다. Fundamental matrix, `F`는 intrinsic matrix, `K`에 대해 다음과 같이 표현된다. 

$ F = (k'^{-1})^T E K^{-1} $

$ F = (K'^{-1})^T [t_x] R K^{-1} $

$ \widetilde{x}'^T F \widetilde{x} = 0 $

이 때, $ [t_x] $ 는 translation matrix를 대칭행렬한 형태를 의미하고, k',k는 stereo camera의 경우 두 이미지 plane은 각각의 intrinsic matrix를 가지기 때문에, 분리하여 표현해주었다. `-1`는 inverse를 의미한다.

<br>

<img src="/assets/img/dev/week16/day1/fundamental_matrix.png">

이 fundamental matrix를 사용하여 좌측 이미지의 점 $ \widetilde{x} $와 F-matrix를 곱하면 우측 이미지에서의 epipolar line, `l'`이 된다. 

$ l' = F\widetilde{x} $

그리고, 아까 봤던 식을 통해 새로운관계식을 구상할 수 있다.

$ \widetilde{x}'^T F \widetilde{x} = 0 $

$ \widetilde{x}'^T l' = 0 $

즉, 우측 이미지의 좌표와 우측 이미지에서의 epipolar line과 곱해주면 0이 된다. line과 point가 교차한다면 0이 된다는 정의가 그대로 적용된다. 이를 통해 correspondence를 굉장히 빠르게 구할 수 있다. 우측 이미지에서의 픽셀이 왼쪽이미지에서는 어떤 픽셀과 correspondence를 이루는가에 대한 문제를 풀기 위해 좌측 이미지에서 모든 픽셀에 가능성을 둘 필요없이 우측 이미지에서의 좌표와 F-matrix를 곱해줘서 나오는 line과 벡터 연산을 해서 0이 나오는 픽셀만 사용하면 되는 것이다.

<br>

Essential matrix와 Fundamental matrix를 구하는 방법은 각각 5-point algorithm, 8-point algorithm이 있다. 최소 5개, 8개의 correspondence가 존재할 때 Essential matrix, Fundamental matrix를 구할 수 있다.

![image](https://user-images.githubusercontent.com/33013780/170962155-79c176c6-2db3-493d-b7d7-0c13136c5fd5.png)

Fundamental matrix의 경우 7DoF를 가지고 있다. translation(tx,ty,tz), totation(rx,ry,rz), focal length(f)와 c(principal point)가 있는데, 여기서 scale 값을 구할 수 없기 때문에 1개를 빼서 7이 된다. essential matrix는 5DoF, translation 3, rotation 3에서 1이 빠져서 5가 된다.

DoF(degree of freedom)를 알아야 이 값을 추론하기 위해서 가져야 하는 최소한의 데이터의 수가 어떤지 알 수 있고, 아무런 prior없이 값을 추론하기 위해서는 데이터가 요구하는 DoF의 수만큼의 데이터가 필요하기 때문에 DoF를 알고 있어야 한다. 이처럼 최소한의 데이터의 수로 문제를 해결하는 알고리즘을 `minimum-solver`라 한다. 5DoF의 essensial matrix를 추론하기 위해서는 5개의 데이터가 필요하므로 이는 minimum-solver에 해당하지만, 7DoF의 fundamental matrix를 추론하기 위해서는 8개의 데이터가 필요하므로 이는 non-minimum-solver에 해당한다.

fundamental matrix를 추론하기 위해 7개의 데이터를 사용해도 되지만, 오류를 잡기도 어렵고, 내부 구조가 어려워져서 대체로 8개로 추론하도록 한다.

<br>

essential matrix나 fundamental matrix를 구하는 방법은 OpenCV에 잘 구현되어 있다. essential matrix의 경우는 `recoverPose()`, fundamental matrix의 경우는 `findFundamentalMat()`이다.

- [recoverPose() OpenCV docs](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1b2f149ee4b033c4dfe539f87338e243)
- [findFundamentalMat() OpenCV docs](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga59b0d57f46f8677fb5904294a23d404a)

<br>

<br>

# RANSAC



<br>

<br>

### 참고하면 좋은 논문
- [BRIEF : Binary Robust Independent Elementary Features](https://link.springer.com/chapter/10.1007/978-3-642-15561-1_56)
- [ORB : an efficient alternative to SIFT or SURF](https://ieeexplore.ieee.org/abstract/document/6126544)
