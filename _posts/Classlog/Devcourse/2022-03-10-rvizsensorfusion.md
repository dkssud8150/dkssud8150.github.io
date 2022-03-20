---
title:    "[데브코스] 4주차 - RVIZ에서 모터와 센서 통합 "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-10 21:14:00 +0800
categories: [Classlog, devcourse]
tags: [ros, rviz, sensor, devcourse]
toc: True
comments: True
image:
  src: /assets/img/dev/week4/day4/sensorfusion.png
  width: 800
  height: 500
---

<br>

# RVIZ에서 모터와 센서 통합하기

RVIZ 가상공간에서 8자 주행하는 자이카에 라이다 센서와 IMU센서의 뷰어를 통합해본다. 3D 모델링된 차량이 8자 주행을 하면서 주변 장애물가지의 거리값을 Range로 표시하고 IMU 센싱값에 따라 차체가 기울어지도록 한다.

![](/assets/img/dev/week4/day4/2022-03-18-00-39-22.png)

1. driver노드 -\> /xycar_motor토픽 -\> converter노드 -\> /joint_states -\> RVIZ viewer, rviz_odom 노드 -\> /odom 토픽 -\> rviz
2. lidar_topic.bag -\> rosbag -\> /scan 토픽 -\> lidar_urdf.py 파이썬 -\> scan1~4 토픽 4개 -\> rviz
3. imu_data.txt -\> imu_data_generator 노드 -\> /imu 토픽 -\> rviz

odom노드는 imu토픽과 joint_states 토픽을 받아서 /odom 토픽을 발행해야 한다.

<br>

<br>

1. 패키지 이름 : rviz_all
2. urdf 파일 : rviz_all.urdf -\> xycar_3d.urdf + lidar_urdf.urdf
3. rviz 파일 : rviz_all.rviz -\> rviz_odom.rviz 복사
4. python 파일 : odom_imu.py
  - rviz_odom.py를 수정, imu 토픽을 구독하여 획득한 쿼터니언 값을 odometry 데이터에 넣어준다.
  - odometry정보를 차체에 해당하는 base_link에 연결, imu값에 따라 차체가 움직이게 한다.
5. launch 파일 : rviz_all.launch
  - urdf, rviz, robot_state_publisher는 기본적으로 작성
  - 노드 실행
    - 자동차 8자 주행 : odom_8_drive.py, odom_imu.py, converter.py
    - 라이다 토픽 발행 : rosbag, lidar_urdf.py
    - IMU 토픽 발행 : imu_generator.py

<br>

<br>

## 1. 패키지 제작

```bash
$ catkin_create_pkg rviz_all rospy tf geometry_msgs urdf rviz xacro
$ cm
```

```bash
$ mkdir rviz_all/launch rviz_all/urdf rviz_all/rviz
```

<br>

## 2. rviz_all.urdf

기존의 파일을 수정할 것이다.

```xml
<?xml version="1.0" ?>
<robot name="xycar" xmlns:xacro="http://www.ros.org/wiki/xacro">
    <link name="base_link"/>

        <!-- baselink => baseplate -->
        <link name="baseplate">
            <visual>
                <material name="acrylic"/>
                <origin rpy="0 0 0" xyz="0 0 0"/>
                <geometry>
                    <box size="0.5 0.2 0.07"/>
                </geometry>
            </visual>
        </link>

        <joint name="base_link_to_baseplate" type="fixed">
            <parent link="base_link"/>
            <child link="baseplate"/>
            <origin rpy="0 0 0" xyz="0 0 0"/>
        </joint>


        <!-- baseplate => front_mount -->
        <link name="front_mount">
            <visual>
                <material name="blue"/>
                <origin rpy="0 0 0" xyz="-0.105 0 0"/>
                <geometry>
                    <box size="0.5 0.12 0.01"/>
                </geometry>
            </visual>
        </link>

        <joint name="baseplate_to_front_mount" type="fixed">
            <parent link="baseplate"/>
            <child link="front_mount"/>
            <origin rpy="0 0 0" xyz="0.105 0 -0.059"/>
        </joint>

        <!-- lidar data -->
        <!-- front -->
        <link name="front" />
        <joint name="baseplate_to_front" type="fixed">
            <parent link="baseplate"/>
            <child link="front"/>
            <origin rpy="0 0 0" xyz="0.25 0 0"/>
        </joint>

        <!-- back -->
        <link name="back" />
        <joint name="baseplate_to_back" type="fixed">
            <parent link="baseplate"/>
            <child link="back"/>
            <origin rpy="0 0 3.14" xyz="-0.25 0 0"/>
        </joint>

        <!-- left -->
        <link name="left" />
        <joint name="baseplate_to_left" type="fixed">
            <parent link="baseplate"/>
            <child link="left"/>
            <origin rpy="0 0 1.57" xyz="0 0.1 0"/>
        </joint>

        <!-- right -->
        <link name="right" />
        <joint name="baseplate_to_right" type="fixed">
            <parent link="baseplate"/>
            <child link="right"/>
            <origin rpy="0 0 -1.57" xyz="0 -0.1 0"/>
        </joint>



        <link name="front_shaft">
            <visual>
                <material name="black"/>
                <origin rpy="1.57 0 0" xyz="0 0 0"/>
                <geometry>
                    <cylinder radius="0.018" length="0.285"/>
                </geometry>
            </visual>
        </link>

        <joint name="front_mount_to_front_shaft" type="fixed">
            <parent link="front_mount"/>
            <child link="front_shaft"/>
            <origin rpy="0 0 0" xyz="0.105 0 -0.059"/>
        </joint>
        
        <link name="rear_shaft">
            <visual>
                <material name="black"/>
                <origin rpy="1.57 0 0" xyz="0 0 0"/>
                <geometry>
                    <cylinder radius="0.018" length="0.285"/>
                </geometry>
            </visual>
        </link>

        <joint name="front_mount_to_rear_shaft" type="fixed">
            <parent link="front_mount"/>
            <child link="rear_shaft"/>
            <origin rpy="0 0 0" xyz="-0.305 0 -0.059"/>
        </joint>



        <link name="front_right_hinge">
            <visual>
                <material name="white"/>
                <origin rpy="0 0 0" xyz="0 0 0"/>
                <geometry>
                    <cylinder radius="0.015"/>
                </geometry>
            </visual>
        </link>

        <joint name="front_right_hinge_joint" type="revolute">
            <parent link="front_shaft"/>
            <child link="front_right_hinge"/>
            <origin rpy="0 0 0" xyz="0 -0.1425 0"/>
            <axis xyz="0.0 0.0 1.0"/>
            <limit lower="-0.34" upper="0.34" effort="10" velocity="100"/>
        </joint>


        <link name="front_left_hinge">
            <visual>
                <material name="white"/>
                <origin rpy="0 0 0" xyz="0 0 0"/>
                <geometry>
                    <cylinder radius="0.015"/>
                </geometry>
            </visual>
        </link>

        <joint name="front_left_hinge_joint" type="revolute">
            <parent link="front_shaft"/>
            <child link="front_left_hinge"/>
            <origin rpy="0 0 0" xyz="0 0.1425 0"/>
            <axis xyz="0.0 0.0 1.0"/>
            <limit lower="-0.34" upper="0.34" effort="10" velocity="100"/>
        </joint>





        <link name="front_right_wheel">
            <visual>
                <material name="black"/>
                <origin rpy="1.57 0 0" xyz="0 0 0"/>
                <geometry>
                    <cylinder length="0.064" radius="0.07"/>
                </geometry>
            </visual>
        </link>

        <joint name="front_right_wheel_joint" type="continuous">
            <parent link="front_right_hinge"/>
            <child link="front_right_wheel"/>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <axis xyz="0.0 1.0 0.0"/>
            <limit effort="10" velocity="100"/>
        </joint>
        

        <link name="front_left_wheel">
            <visual>
                <material name="black"/>
                <origin rpy="1.57 0 0" xyz="0 0 0"/>
                <geometry>
                    <cylinder length="0.064" radius="0.07"/>
                </geometry>
            </visual>
        </link>

        <joint name="front_left_wheel_joint" type="continuous">
            <parent link="front_left_hinge"/>
            <child link="front_left_wheel"/>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <axis xyz="0.0 1.0 0.0"/>
            <limit effort="10" velocity="100"/>
        </joint>


        <link name="rear_right_wheel">
            <visual>
                <material name="black"/>
                <origin rpy="1.57 0 0" xyz="0 0 0"/>
                <geometry>
                    <cylinder length="0.064" radius="0.07"/>
                </geometry>
            </visual>
        </link>

        <joint name="rear_right_wheel_joint" type="continuous">
            <parent link="rear_shaft"/>
            <child link="rear_right_wheel"/>
            <origin rpy="0 0 0" xyz="0 -0.14 0"/>
            <axis xyz="0.0 1.0 0.0"/>
            <limit effort="10" velocity="100"/>
        </joint>


        <link name="rear_left_wheel">
            <visual>
                <material name="black"/>
                <origin rpy="1.57 0 0" xyz="0 0 0"/>
                <geometry>
                    <cylinder length="0.064" radius="0.07"/>
                </geometry>
            </visual>
        </link>

        <joint name="rear_left_wheel_joint" type="continuous">
            <parent link="rear_left_hinge"/>
            <child link="rear_left_wheel"/>
            <origin rpy="0 0 0" xyz="0 0.14 0"/>
            <axis xyz="0.0 1.0 0.0"/>
            <limit effort="10" velocity="100"/>
        </joint>

    <material name="black">
        <color rgba="0.0 0.0 0.0 1.0"/>
    </material>
    ...
</robot>
```

<br>

## 3. rviz_all.rviz

이는 설정파일이므로 rviz로 들어가서 수정하면 되는데, 다른 rviz파일을 복사해서 수정하면 편하다. 
- 라이다센서의 시각화에 쓰이는 `Range` 설정
- by_topic에서 scan1,2,3,4 에 대한 `Range` 설정
- 궤적 표시를 위해 `odometry` 추가
  - topic = /odom 입력
  - keep = 100
  - shaft length = 0.05
  - head length = 0.1

`save`하고 나간다.

<br>

## 4. rviz_all.launch

```xml
<launch>
  <param name="robot_description" ... />
  <prarm name="use_gui" value="true"/>
  <node name="rviz_visualizer" .../>
  <node name="robot_state_publisher" .../>

  <node name="driver" pkg="rviz_xycar" type="odom_8_drive.py"/>
  <node name="odometry" pkg="rviz_all" type="odom_imu.py"/>
  <node name="motor" pkg="rviz_xycar" type="converter.py"/>

  <node name="rosbag_play" pkg="rosbag" type="play" output="screen" required="true" args="$(find rviz_lidar)/src/lidar_topic.bag"/>
  <node name="lidar" pkg="rviz_lidar" type="lidar_urdf.py" outout="screen"/>
  <node name="imu" pkg="rviz_imu" type="imu_generator.py"/>
</launch>
```

<br>

## 5. odom_imu.py

imu데이터를 차량의 odometry에 적용한다. odometry 데이터를 생성하는 rviz_odom.py를 수정하여 odom_imu를 제작한다. imu토픽을 구독하여 획득한 쿼터니언 값을 odometry 데이터에 넣는다.

```python
def callback_imu(msg):
  global Imudata
  Imudata[0] = msg.orientation.x
  Imudata[1] = msg.orientation.y
  Imudata[2] = msg.orientation.z
  Imudata[3] = msg.orientation.w

odom_quat = Imudata

odom_broadcaster.sendTransform(
  (x_,y_,0.),
  odom_quat,
  current_time,
  "base_link",,
  "odom"
)
```

<br>

## 실행 결과

```bash
$ roslaunch rviz_all rviz_all.launch
```


