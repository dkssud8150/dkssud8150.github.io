---
title:    "[데브코스] 14주차 - ImageProcessing geometrical distance estimation"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-05-16 14:20:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, ImageProcessing]
toc: true
comments: true
math: true
---

<br>

거리 추정을 위해 여러 가지 과정을 작성할 것이다. 먼저 거리를 추정하는 방법에는 크게 2가지가 있다.

1. homography
2. geometric method using FOV

homography의 경우 치명적인 단점이 있다. 지면이 평면이어야만 하는데, 실제 상황에서는 평면일 경우가 거의 드물다. 오르막길, 내리막길은 물론이고, 급출발, 급정거를 할 때도 pitch가 생겨서 평면이 아니게 된다. 그래서 homography를 사용한 방법은 간단하게 설명하고 2번으로 넘어가겠다.

# distance estimation using homography method

```python
"""
순서
1. 실측을 통한 obj point , img point 설정
2. intrinsic, distort coefficients 정보 불러오기
3. undistort
4. obj point -> homogeneous 좌표계로 변환
5. solvePnP, drawFrameAxes로 좌표축 확인, 0,0,0 좌표로 그려져야 맞는것
6. findHomography 로 perspective matrix 추출
7. image point -> homogeneous 좌표계로 변환
8. homography와 image point를 내적하면 x,y,z 가 추출됨.
"""

import os, json
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

visualize = False
testing = False

def estimation_distance():
    ######## step 2. intrinsic, distort coefficients information
    calibration_jsonfile_path = os.path.join("./calibration.json")
    with open(calibration_jsonfile_path, 'r') as calibration_file:
        calibration_info = json.load(calibration_file)
        intrinsic = calibration_info['intrinsic']
        fx, fy, cx, cy = intrinsic['fx'], intrinsic['fy'], intrinsic['cx'], intrinsic['cy']
        camera_matrix = np.array([
                                [fx,      0.00000,      cx],
                                [0.00000,      fy,      cy],
                                [0.00000, 0.00000, 1.00000],
                                ], dtype=np.float64)
        dist_coeff = np.array([intrinsic['distortion_coff']],
                                dtype=np.float64)

    img_file_path = "./source/images/img_1.jpg"
    img = cv2.imread(img_file_path)


    

    ######## step 3. undistort
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeff, None, None, (img.shape[1], img.shape[0]), 5)
    undistort_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    """
    높이 16.5cm, 가로 25cm, 세로 30cm
    """

    img_points = np.array([
                        [170, 426], # bottom left
                        [459, 426], # bottom right
                        [193, 409], # second left
                        [433, 409], # second right
                        [206, 380], # third left
                        [416, 380], # third right
                        [216, 366], # top left
                        [403, 366], # top right
                        ], dtype=np.float32)


    obj_points = np.array([
                        [1, 00.0, 00.0], # bottom left
                        [1, 25.0, 00.0], # bottom right
                        [1, 00.0, 10.0], # second left
                        [1, 25.0, 10.0], # second right
                        [1, 00.0, 20.0], # third left
                        [1, 25.0, 20.0], # third right
                        [1, 00.0, 30.0], # top left
                        [1, 25.0, 30.0], # top right
                        ], dtype=np.float32)

    data_size = len(img_points)

    obj_points = obj_points / obj_points[0,0]
    homo_obj_points = cv2.hconcat([obj_points[:,1], obj_points[:,2], obj_points[:,0]])
    homo_obj_points[:,1] = 0 - homo_obj_points[:,1]

    if visualize:
        # PLT SHOW for search pixel value
        # both_img = cv2.hconcat([undistort_img, img])
        plt_img = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2RGB)
        plt.imshow(plt_img)
        plt.show()

        ######## get rotation , translation vector using obj points and img points
        _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, distCoeffs=None, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)
        undistort_img = cv2.drawFrameAxes(undistort_img, camera_matrix, distCoeffs=dist_coeff, rvec=rvec, tvec=tvec, length=2, thickness=5)

        ######## image points, object points 의 pair 쌍이 맞는지 재투영을 통해 확인
        proj_image_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, None)
        for proj_image_point, img_point in zip(proj_image_points, img_points):
            print(img_point, proj_image_point, "\n")

        ######## 카메라 축 확인
        img = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    homography, _ = cv2.findHomography(img_points, homo_obj_points)

    print(f"data_size {data_size}")

    return homography

if __name__ =="__main__":
    # get homography
    img = cv2.imread("img_1.jpg", cv2.IMREAD_ANYCOLOR)


    ### bbox example
    top_left     = (160,255) #(416,248)
    top_right    = (168,255) #(426,248)
    bottom_left  = (160,247) #(417,258)
    bottom_right = (168,247) #(427,258)

    center_point = (bottom_right[0] - bottom_left[0], bottom_left[1] - top_left[1])

    homography = estimation_distance()
    img_point = np.array([center_point[0],center_point[1], 1], dtype=np.float32)

    ######## inference
    estimation = np.dot(homography, img_point)
    x,y,z = estimation[0] ,estimation[1],estimation[2]
    distance = math.sqrt((x/z)**2 + (y/z)**2 + (z/z)**2) 

    ######## visualize
    cv2.rectangle(img, (top_left), (bottom_right), (255,255,0), 2)
    cv2.putText(img, "distance : "+str(round(distance,2)) + "cm", top_left, 2, 0.5, (10,250,10),1)

    cv2.imshow("img", img)
    cv2.waitKey()

    print(f"homography matrix \n{homography}\n")    
    print(f"distance {round(distance,2)}cm")    

```

- 순서
1. 좌표축 원점으로 정할 객체를 기준으로 실측을 통한 obj point , img point 설정
2. intrinsic, distortion coefficients 정보 불러오기
3. undistort
4. obj point -> homogeneous 좌표계로 변환
5. solvePnP, drawFrameAxes로 좌표축 확인, 0,0,0 좌표로 그려져야 맞는것
6. findHomography 로 perspective matrix 추출
7. image point -> homogeneous 좌표계로 변환
8. homography와 image point를 내적하면 x,y,z 가 추출됨.

<br>

이렇게하면 실제 객체와의 거리를 구할 수 있다. 이 때 주의해야 할 점은 homography랑 내적을 하면 x,y,z에서 z가 1이어야 하지만, 1이 아니다. 따라서 z를 나눠주어야 한다.

```markdown
homography matrix 
[[-7.97942714e-02 -4.84927735e-02  3.48389453e+01]
 [ 3.63617110e-07 -2.79608421e-01  1.20439194e+02]
 [-5.94557243e-07 -4.39631252e-03  1.00000000e+00]]

distance 127.46cm
```

<img src="/assets/img/dev/week14/homography.png">

<br>

homography matrix라는 것 자체가 2차원에서 3차원으로 투영하기 위한 행렬이다. 그래서 image points에서 object points로 가기 위한 3x3 행렬이고, object points를 잡을 때 후륜축을 잡을지 카메라 위치를 잡을지에 따라 중심 좌표와 객체와의 거리를 추정할 수 있다. homography를 하면 카메라의 높이를 고려하지 않아도 되지만 지면 자체가 평면이라고 생각하고 사용하는 알고리즘이므로 지면이 평면이 아니라면 사용할 수 없다고 한다. 

<br>




