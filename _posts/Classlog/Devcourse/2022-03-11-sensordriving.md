---
title:    "[데브코스] 4주차 - 센서를 활용한 자율주행 "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-11 15:14:00 +0800
categories: [Classlog, devcourse]
tags: [ros, urdf, rviz,devcourse]
toc: True
comments: True
# image:
#   src: /assets/img/dev/week4/day3/lidar.jpeg
#   width: 800
#   height: 500
---

<br>

# 라이다를 통한 장애물 회피

- lidar_drive 패키지를 사용
- lidar_gostop.py , lidar_gostop.launch

```xml
<!-- lidar_gostop.launch -->
<launch>
    <include file="$(find xycar_motor)/launch/xycar_motor.launch"/>
    <include file="$(find xycar_lidar)/launch/lidar_noviewer.launch"/>
    <node name="lidar_driver" pkg="lidar_drive" type="lidar_gostop.py" output="screen" />
</launch>
```

```python
#!/usr/bin/env python

# import msg file
import rospy, time
from sensor_msgs.msg import LaserScan 
from xycar_msgs.msg import xycar_motor 


motor_msg = xycar_motor()
distance = [] # prepare storage to save the distance value for lidar

# if topic about lidar enter, define the callback function to implement
def callback(data):
    global distance, motor_msg
    distance = data.ranges

# define function for going
def drive_go():
    global motor_msg
    motor_msg.speed = 5
    motor_msg.angle = 0
    pub.publish(motor_msg)

# define function for stoping
def drive_stop():
    global motor_msg
    motor_msg.speed = 0
    motor_msg.angle = 0
    pub.publish(motor_msg)

# make node and define sub and pub node
rospy.init_node('lidar_driver')
rospy.Subscriber('/scan', LaserScan, callback, queue_size = 1)
pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size = 1)

# ready to connect lidar
time.sleep(3)

# scan from 60 degree to 120 degree, because front side is 90 degree
while not rospy.is_shutdown():
    ok = 0
    for degree in range(60, 120):
        if distance[degree] <= 0.3:
            ok += 1
        if ok > 3:          # if more than three are lower 30cm, do stop
            drive_stop()
            break
    if ok <= 3:             # else go
        drive_go()
```

<br>

- 실행

```bash
$ roslaunch lidar_drive lidar_gostop.launch
```

<br>

<br>

# 초음파를 통한 장애물 회피

- ultra_drive 패키지
- ultra_drive.launch, ultra_drive.py

```xml
<!-- ultra_drive.launch -->
<launch>
    <include file="$(find xycar_motor)/launch/xycar_motor.launch"/>

    <node name="xycar_ultra" pkg="xycar_ultrasonic" type="ultra_ultrasonic.py" output="screen" />
    <node name="ultra_driver" pkg="ultra_drive" type="ultra_gostop.py" output="screen" />
</launch>
```

```python
#!/usr/bin/env python

# ultra_gostop.py

# import msg file
import rospy, time
from std_msgs.msg import Int32MultiArray
from xycar_msgs.msg import xycar_motor 


motor_msg = xycar_motor()
ultra_msg = None # prepare storage to save the distance value for ultrasonic

# if topic about ultra enter, define the callback function to implement
def callback(data):
    global ultra_msg
    ultra_msg = data.data

# define function for going
def drive_go():
    global motor_msg, pub
    motor_msg.speed = 5
    motor_msg.angle = 0
    pub.publish(motor_msg)

# define function for stoping
def drive_stop():
    global motor_msg, pub
    motor_msg.speed = 0
    motor_msg.angle = 0
    pub.publish(motor_msg)

# make node and define sub and pub node
rospy.init_node('ultra_driver')
rospy.Subscriber('/xycar_ultrasonic', Int32MultiArray, callback, queue_size = 1)
pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size = 1)

# ready to connect ultra
time.sleep(2)

# scan from 60 degree to 120 degree, because front side is 90 degree
while not rospy.is_shutdown():
    if ultra_msg[2] > 0 and ultra_msg[2] < 10:
        drive_stop()  # if front ultrasonic sensor the detected distance information is 
                      # 0 < distance < 10cm, do stop
    else:             # other situation is infinition, so then there are no obstacles.
        drive_go()    # therefore, let's go
```

<br>

- 실행

```bash
$ roslaunch ultra_drive ultra_gostop.launch
```