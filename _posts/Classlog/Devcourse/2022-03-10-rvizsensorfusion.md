---
title:    "[데브코스] 4주차 - ROS Motor and sensor integration in RVIZ "
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


  <!-- 차체 -->
  <link name="front_shaft">
    <visual>
      <material name="black"/>
      <origin rpy="1.57 0 0" xyz="0 0 0"/>
      <geometry>
        <cylinder length="0.285" radius="0.018"/>
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
        <cylinder length="0.285" radius="0.018"/>
      </geometry>
    </visual>
  </link>
  <joint name="rear_mount_to_rear_shaft" type="fixed">
    <parent link="front_mount"/>
    <child link="rear_shaft"/>
    <origin rpy="0 0 0" xyz="-0.305 0 -0.059"/>
  </joint>
  

  <link name="front_right_hinge">
    <visual>
      <material name="white"/>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.015"/>
      </geometry>
    </visual>
  </link>
  <joint name="front_right_hinge_joint" type="revolute">
    <parent link="front_shaft"/>
    <child link="front_right_hinge"/>
    <origin rpy="0 0 0" xyz="0 -0.1425 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" lower="-0.34" upper="0.34" velocity="100"/>
  </joint>


  <link name="front_left_hinge">
    <visual>
      <material name="white"/>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.015"/>
      </geometry>
    </visual>
  </link>
  <joint name="front_left_hinge_joint" type="revolute">
    <parent link="front_shaft"/>
    <child link="front_left_hinge"/>
    <origin rpy="0 0 0" xyz="0 0.1425 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" lower="-0.34" upper="0.34" velocity="100"/>
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
    <axis xyz="0 1 0"/>
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
    <axis xyz="0 1 0"/>
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
    <axis xyz="0 1 0"/>
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
    <parent link="rear_shaft"/>
    <child link="rear_left_wheel"/>
    <origin rpy="0 0 0" xyz="0 0.14 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="10" velocity="100"/>
  </joint>

  <!-- 색상 정보 -->
  <material name="black">
      <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
      <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
      <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
      <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  <material name="orange">
      <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
      <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
      <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
      <color rgba="1.0 1.0 1.0 1.0"/>
  </material>
  <material name="acrylic">
      <color rgba="1.0 1.0 1.0 0.4"/>
  </material>
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
    <param name="robot_description" textfile="$(find rviz_all)/urdf/rviz_all.urdf" />
    <param name="use_gui" value="true"/>

    <node name="rviz_visualizer" pkg="rviz" type="rviz" required="true"
    args="-d $(find rviz_all)/rviz/rviz_all.rviz"/>
    <node name="robot_state_publisher" pkg="robot_state_publisher" type="state_publisher"/>

    <node name="driver" pkg="rviz_xycar" type="odom_8_drive.py" />
    <node name="odometry" pkg="rviz_xycar" type="rviz_odom.py" />
    <node name="converter" pkg="rviz_xycar" type="converter.py" />

    <node name="rosbag_play" pkg="rosbag" type="play" output="screen" 
        required="true" args="$(find rviz_lidar)/src/lidar_topic.bag" />

    <node name="lidar" pkg="rviz_lidar" type="lidar_urdf.py" output="screen" />
	<node name="imu" pkg="rviz_imu" type="imu_generator.py" />
</launch>
```

<br>

## 5. odom_imu.py

imu데이터를 차량의 odometry에 적용한다. odometry 데이터를 생성하는 rviz_odom.py를 수정하여 odom_imu를 제작한다. imu토픽을 구독하여 획득한 쿼터니언 값을 odometry 데이터에 넣는다.

```python
#!/usr/bin/env python

import math, rospy, tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu




def callback(msg):
    global Angle
    Angle = msg.position[msg.name.index("front_left_hinge_joint")]

rospy.Subscriber('joint_states', JointState, callback)

def callback_imu(msg):
    global Imudata
    Imudata[0] = msg.orientation.x
    Imudata[1] = msg.orientation.y
    Imudata[2] = msg.orientation.z
    Imudata[3] = msg.orientation.w    

rospy.Subscriber('imu', Imu, callback_imu)

rospy.init_node("odometry_publisher")

odom_pub = rospy.Publisher("odom", Odometry, queue_size=50)

odom_broadcaster = tf.TransformBroadcaster()

cur_time = rospy.Time.now()
prv_time = rospy.Time.now()

rate = rospy.Rate(30.0)

cur_speed = 0.4
wheel_base = 0.2
x = 0
y = 0
yaw = 0
Angle = 0

while not rospy.is_shutdown():
    cur_time = rospy.Time.now() 
    dt = (cur_time - prv_time).to_sec()

    cur_steering_ang = Angle
    cur_angluar_velocity = cur_speed * math.tan(cur_steering_ang) / wheel_base

    x_dot = cur_speed * math.cos(yaw)
    y_dot = cur_speed * math.sin(yaw)
    x += x_dot * dt
    y += y_dot * dt
    yaw += cur_angluar_velocity * dt

    odom_quat = Imudata

    odom_broadcaster.sendTransform(
        (x, y, 0),
        odom_quat,
        cur_time,
        "base_link",
        "odom"
    )   

    odom = Odometry()
    odom.header.stamp = cur_time
    odom.header.frame_id = "odom"
    
    odom.pose.pose = Pose(Point(x, y, 0), Quaternion(*odom_quat))
    odom.child_frame_id = "base_link"
    odom_pub.publish(odom)

    prv_time = cur_time
    rate.sleep()
```

<br>

## 실행 결과

```bash
$ roslaunch rviz_all rviz_all.launch
```

<br>

<img src="/assets/img/dev/week4/day4/rviz_all_frame.png">

이 그림은 urdf의 구성도이다. 어떻게 연결되었는지 볼 수 있다.

```bash
$ rosrun tf view_frames
```

이 식을 실행하면 구성도를 pdf로 추출해준다.

<br>

<img src="/assets/img/dev/week4/day4/rviz_all_rviz.png">

이는 실행한 rviz화면이다. 8자 주행을 하고 있고, lidar 센서를 통해 얻은 거리 정보를 활용하여 원뿔 모양이 나타나고 있으며, Imu 센서를 통해 얻은 기울기 정보를 활용하여 차체가 기울기가 어떻게 되고 있는지를 볼 수 있다.

<br>

<img src="/assets/img/dev/week4/day4/rviz_all_rqtgraph.png">

이는 rqt_graph이다.
