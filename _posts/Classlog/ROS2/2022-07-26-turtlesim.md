---
title:    "[ROS2 이론] turtlesim 실행"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-07-26 03:23:00 +0800
categories: [Classlog, ROS2]
tags: [ROS2]
toc: True
comments: True
--- 

| 개발 환경을 설정하는 것은 [해당 블로그](https://dkssud8150.github.io/posts/introduction/)를 확인해보길 바란다.

&nbsp;

- [007 패키지 설치와 노드 실행](https://cafe.naver.com/openrt/24065)

# Turtlesim 이란?







# Turtlesim 패키지 설치

turtlesim은 ROS의 기본 기능을 가지고 있어 튜토리얼로 많이 사용된다.

```bash
sudo apt install ros-foxy-turtlesim
```

turtlesim 패키지에 포함된 노드가 어떤 것들이 있는지 확인해보자.

```bash
$ ros2 pkg executables turtlesim
turtlesim draw_square
turtlesim mimic
turtlesim turtle_teleop_key
turtlesim turtlesim_node
```

- draw_square : 사각형 모양으로 turtle을 움직이게 하는 노드
- mimic : 유저가 지정한 토픽으로 동일 움직임의 turtlesim_node를 복수개 실행시킬 수 있는 노드
- turtle_teleop_key : turtlesim_node를 움직이게 하는 속도 값을 퍼블리시하는 노드
- turtlesim_node : turtle_teleop_key로부터 속도 값을 토픽으로 받아 움직이게 하는 간단 2D 시뮬레이터 노드

&nbsp;

# Turtlesim 패키지 노드 실행

**turtlesim_node** 노드와 **turtle_teleop_key** 노드를 실행시키면 파란색 창에 거북이가 보인다. **turtle_teleop_key** 노드를 실행시키면 화살표키로 **turtlesim_node** 노드의 거북이를 움직일 수 있게 된다. 

```bash
ros2 run turtlesim turtlesim_node
```

```bash
ros2 run turtlesim turtle_teleop_key
```

<img src="/assets/img/ros2/turtlesim1.png">

<img src="/assets/img/ros2/turtlesim2.png">

&nbsp;

- 키보드로 움직일 수도 있는데, 눌려진 키보드의 키 값이 어떻게 메시지로 전달되는지 확인해봐야 한다. 키 값은 linear velocity와 angular velocity가 포함된 geometry_msgs 패키지의 Twist 메시지 형태로 보내지고 있다.

&nbsp;

그렇기에 직접 터미널로 토픽을 발행하여 조종해보자.

그 전에 노드, 토픽, 서비스, 액션에 대해 알고 있어야 한다.

&nbsp;

- [008 ROS2 노드와 데이터 통신](https://cafe.naver.com/openrt/24086)

노드와 메시지 통신은 앞서 글에서 배웠기 때문에, 간단히 설명했다.

- 노드(node)
  - 최소 단위의 실행 가능한 프로세스
  - 각 노드는 서로 유기적으로 message로 연결되어 사용된다.
- 메시지 통신(message communication)
  - 노드와 노드 사이에 입력과 출력 데이터를 서로 주고받게 설계해야 한다. 주고 받는 데이터를 메시지(message)라 한다.
  - 주고받는 방식을 메시지 통신(message communication)이라 한다.
  - 주고받는 통신 방법에 따라 토픽, 서비스, 액션, 파라미터로 나뉜다.

&nbsp;

- 토픽

<img src="/assets/img/ros2/topic.png">

그림에서처럼 NodeA - NodeB, NodeA - NodeC 처럼 비동기식 단방향 메시지 송수신 방식을 **토픽**(topic)이라 하고, 메시지를 발간하는 *Publisher*, 메시지를 구독하는 *Subscriber*가 존재한다.

&nbsp;

- 서비스

<img src="/assets/img/ros2/service.png">

NodeB - NodeC 처럼 동기식 양방향 메시지 송수신 방식을 **서비스**(Service)라 하고, 서비스를 요청(request)하는 쪽을 *service client*라 하고, 서비스를 응답(response)하는 쪽을 *service server*라 한다.

&nbsp;

- 액션

<img src="/assets/img/ros2/action.png">

NodeA - NodeB 처럼 비동기식 + 동기식 양방향 메시지 송수신 방식을 액션(action)이라 하고, 액션 목표(Goal)을 지정하는 **action client**와 액션 목표를 받아 특정 태스크를 수행하면서 중간 결과값에 해당하는 액션 피드백(Feedback)과 최종 결과값에 해당하는 액션 결과(result)를 전송하는 action server간의 통신을 액션(action)이라 볼 수 있다.

즉 `Action Goal -> Action Feedback -> Action Result` 로 진행된다.

액션의 구현 방식을 더 자세히 살펴보면 토픽과 서비스의 혼합이라 볼 수 있는데, 액션 목표와 액션 결과를 전달하는 방식은 서비스와 같고, 액션 피드백은 토픽과 같은 메시지 전송 방식이다. 그래서 피드백의 퍼블리셔는 Action Server라 할 수 있고, 피드백의 서브스크라이버는 Action client와 service client라 할 수 있다.

&nbsp;

- 파라미터

<img src="/assets/img/ros2/parameter.png">

각 노드에 **파라미터 관련 parameter server를 실행시켜 외부의 parameter client 간의 통신**으로 파라미터를 변경하는 것으로 서비스와 동일하다. 단 노드 내 매개변수 또는 글로벌 매개변수를 서비스 메시지 통신 방법을 사용하여 **노드 내부 또는 외부에서 쉽게 지정(set)하거나 변경할 수 있고, 쉽게 가져와(get) 사용**할 수 있다.

&nbsp;

이제 turtlesim에서 실행되는 통신을 살펴보자. 먼저 어떤 노드들이 실행되고, 어떤 토픽, 어떤 서비스와 액션이 있는지 확인해봐야 한다.

```bash
$ ros2 node list
/teleop_turtle
/turtlesim

$ ros2 topic list
/parameter_events
/rosout
/turtle1/cmd_vel
/turtle1/color_sensor
/turtle1/pose

$ ros2 service list
/clear
/kill
/reset
/spawn
/teleop_turtle/describe_parameters
/teleop_turtle/get_parameter_types
/teleop_turtle/get_parameters
/teleop_turtle/list_parameters
/teleop_turtle/set_parameters
/teleop_turtle/set_parameters_atomically
/turtle1/set_pen
/turtle1/teleport_absolute
/turtle1/teleport_relative
/turtlesim/describe_parameters
/turtlesim/get_parameter_types
/turtlesim/get_parameters
/turtlesim/list_parameters
/turtlesim/set_parameters
/turtlesim/set_parameters_atomically

$ ros2 action list
/turtle1/rotate_absolute
```

&nbsp;

그리고 rqt_graph로 시각화할 수 있다.

```bash
rqt_graph
```

<img src="/assets/img/ros2/turtlesim_rqt_graph.png">

&nbsp;

## turtlesim 노드 정보

먼저 노드를 자세히 살펴보고자 한다. turtlesim의 노드는 총 2개로 구성되어 있다.

```bash
$ ros2 node list
/teleop_turtle
/turtlesim
```

turtlesim_node 노드 파일은 turtlesim이라는 노드명으로 실행되어 있고, turtle_teleop_key 노드 파일은 teleop_turtle 이라는 노드명으로 실행되어 있다.

이 때, 동일 노드를 여러 개 실행하고자 하면 이전과 동일하게 `ros2 run turtlesim turtlesim_node`로 실행할 수 있는데, 이렇게 하면 동일한 노드 이름으로 생성된다.

만약 다른 노드 이름으로 설정하고자 한다면 아래 명령어로 실행한다.

```bash
ros2 run turtlesim turtlesim_node __node:=new_turtle
```

<img src="/assets/img/ros2/turtlesim_rqt_graph2.png">

&nbsp;

노드 이름만 변경되었고, 토픽인 `turtle1/cmd_vel`은 동일하다. 만약 teleop_turtle 노드를 이용하여 거북이를 움직이면 두 개의 노드의 거북이가 동일하게 움직인다. 이는 동일한 토픽을 이용하기 때문이다. 이 토픽 또한 토픽명을 변경하거나 name_space를 통해 바꿀 수 있다.

<img src="/assets/img/ros2/turtlesim_rqt_graph3.png">

```bash
$ ros2 node list
/new_turtle
/teleop_turtle
/turtlesim
```

&nbsp;

&nbsp;

노드의 정보를 확인하기 위해서는 `ros2 node info` 명령어를 사용해야 한다. 이를 통해 퍼블리셔, 서브스크라이버, 서비스, 액션, 파라미터 정보를 확인할 수 있다.

```bash
$ ros2 node info /turtlesim
/turtlesim
  Subscribers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /turtle1/cmd_vel: geometry_msgs/msg/Twist
  Publishers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /rosout: rcl_interfaces/msg/Log
    /turtle1/color_sensor: turtlesim/msg/Color
    /turtle1/pose: turtlesim/msg/Pose
  Service Servers:
    /clear: std_srvs/srv/Empty
    /kill: turtlesim/srv/Kill
    /reset: std_srvs/srv/Empty
    /spawn: turtlesim/srv/Spawn
    /turtle1/set_pen: turtlesim/srv/SetPen
    /turtle1/teleport_absolute: turtlesim/srv/TeleportAbsolute
    /turtle1/teleport_relative: turtlesim/srv/TeleportRelative
    /turtlesim/describe_parameters: rcl_interfaces/srv/DescribeParameters
    /turtlesim/get_parameter_types: rcl_interfaces/srv/GetParameterTypes
    /turtlesim/get_parameters: rcl_interfaces/srv/GetParameters
    /turtlesim/list_parameters: rcl_interfaces/srv/ListParameters
    /turtlesim/set_parameters: rcl_interfaces/srv/SetParameters
    /turtlesim/set_parameters_atomically: rcl_interfaces/srv/SetParametersAtomically
  Service Clients:

  Action Servers:
    /turtle1/rotate_absolute: turtlesim/action/RotateAbsolute
  Action Clients:
```

```bash
$ ros2 node info /teleop_turtle
/teleop_turtle
  Subscribers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
  Publishers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /rosout: rcl_interfaces/msg/Log
    /turtle1/cmd_vel: geometry_msgs/msg/Twist
  Service Servers:
    /teleop_turtle/describe_parameters: rcl_interfaces/srv/DescribeParameters
    /teleop_turtle/get_parameter_types: rcl_interfaces/srv/GetParameterTypes
    /teleop_turtle/get_parameters: rcl_interfaces/srv/GetParameters
    /teleop_turtle/list_parameters: rcl_interfaces/srv/ListParameters
    /teleop_turtle/set_parameters: rcl_interfaces/srv/SetParameters
    /teleop_turtle/set_parameters_atomically: rcl_interfaces/srv/SetParametersAtomically
  Service Clients:

  Action Servers:

  Action Clients:
    /turtle1/rotate_absolute: turtlesim/action/RotateAbsolute
```

&nbsp;

&nbsp;

## turtlesim 토픽

토픽은 비동기식 단방향 메시지 송수신 방식으로 메시지 형태로 메시지를 발행하는 퍼블리셔(publisher)와 메시지를 구동하는 서브스크라이버(subscriber) 간의 통신이라 볼 수 있다. 이는 1:1 통신을 기본으로 하지만, 1:N도 가능하고, 구성에 따라 N:1, N:N 통신도 가능하다.

하나의 노드가 퍼블리셔와 서브스크라이버를 동시에 수행할 수도 있다. 원한다면 자신이 발행한 토픽을 셀프 구독할 수 있게 구성할 수도 있다.

<img src="/assets/img/ros2/node_and_topic.png">

&nbsp;

- 토픽 목록 확인

```bash
ros2 run turtlesim turtlesim_node
```

```bash
ros2 run turtlesim turtle_teleop_key
```

```bash
$ ros2 node info /turtlesim
/turtlesim
  Subscribers:
    /turtle1/cmd_vel: geometry_msgs/msg/Twist
  Publishers:
    /turtle1/color_sensor: turtlesim/msg/Color
    /turtle1/pose: turtlesim/msg/Pose
...
```

토픽만을 보면 위와 같을 것이다. 

- turtlesim 노드
  - subscriber
    - geometry_msgs/msg/Twist 형태의 메시지인 turtle1/cmd_vel
  - publisher
    - turtlesim/msg/color_sensor 메시지 형태인 turtle1/color_sensor
    - turtlesim/msg/Pose 메시지 형태인 turtle1/pose

&nbsp;

간단한 리스트로 확인하기 위해서는 다음과 같이 명령어를 실행하면 된다. 동작 중인 모든 노드들의 토픽 정보를 확인할 수 있다. 이 때, `-t`는 메시지의 형태도 함께 표시하기 위한 옵션이다.

```bash
$ ros2 topic list -t
/parameter_events [rcl_interfaces/msg/ParameterEvent]
/rosout [rcl_interfaces/msg/Log]
/turtle1/cmd_vel [geometry_msgs/msg/Twist]
/turtle1/color_sensor [turtlesim/msg/Color]
/turtle1/pose [turtlesim/msg/Pose]
```

&nbsp;

- 토픽 정보 확인

하나의 토픽을 더 자세히 확인해보자.

```bash
$ ros2 topic info /turtle1/cmd_vel
Type: geometry_msgs/msg/Twist
Publisher count: 1
Subscription count: 1
```

메시지 타입은 **Twist**, 퍼블리셔 1개, 서브스크라이버 1개로 구성되어 있다.

&nbsp;

- 토픽 내용 확인

토픽의 내용을 확인해보자. `teleop_key`를 실행한 터미널에서 방향키를 눌러 거북이를 움직이게 되면, linear 와 angular 값이 담긴 토픽이 발행되고 있는 것을 볼 수 있다.

```bash
$ ros2 topic echo /turtle1/cmd_vel
linear:
  x: 2.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0
---
linear:
  x: 0.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 2.0
---
```

linear에 x,y,z , angular에 x,y,z 총 6개의 값으로 구성되어 있고, linear.x 값은 1.0m/s 단위로 구성되어 있다. 

모든 메시지는 meter, second, degree, kg 등 SI 단위를 기본으로 사용한다.

&nbsp;

- 메시지 크기 확인

메시지의 대역폭, 즉 송수신받는 토픽 메시지의 크기를 확인해보고자 한다. 크기 확인은 `ros2 topic bw`로 토픽의 초당 대역폭을 확인해볼 수 있다.

```bash
$ ros2 topic bw /turtle1/cmd_vel
Subscribed to [/turtle1/cmd_vel]
35 B/s from 2 messages
        Message size mean: 52 B min: 52 B max: 52 B
26 B/s from 2 messages
        Message size mean: 52 B min: 52 B max: 52 B
21 B/s from 2 messages
        Message size mean: 52 B min: 52 B max: 52 B
18 B/s from 2 messages
        Message size mean: 52 B min: 52 B max: 52 B
15 B/s from 2 messages
        Message size mean: 52 B min: 52 B max: 52 B
13 B/s from 2 messages
        Message size mean: 52 B min: 52 B max: 52 B
```

방향키를 통해 실행을 해보면 약 35 B/s 정도의 대역폭을 가지고 있고, 토픽을 보내지 않으면 값이 계속 감소된다.

&nbsp;

- 토픽 주기 확인

토픽의 전송 주기를 확인하기 위해서는 `ros2 topic hz`를 사용해야 한다. 


```bash
$ ros2 topic hz /turtle1/cmd_vel
average rate: 0.449
        min: 1.016s max: 3.439s std dev: 1.21158s window: 2
average rate: 0.293
        min: 1.016s max: 5.792s std dev: 1.94967s window: 3
```

/turtle1/cmd_vel 토픽을 발행하면 약 0.3 Hz를 가지고 있다. 이는 약 0.0003초에 한번씩 토픽을 발행한다.

&nbsp;

- 토픽 지연 시간 확인

토픽은 RMW 및 네트워크 장비를 거치기 때문에, latency가 반드시 존재한다. 지연 시간을 체크하는 방식은 메시지 내 header라는 stamp 메시지를 사용하여 체크한다.

```bash
$ ros2 topic delay /TOPIC_NAME
```

그러나 turtlesim에서는 stamp가 없기에 테스트는 수행하지 못했다.

&nbsp;

- 토픽 발행

토픽을 발행하는 방법은 `ros2 topic pub <topic-name> <msg type> "<args>"` 로 발행할 수 있다.

이 때, `--once`옵션을 사용하여 단 한번만 발행을 할 수 있다.

```bash
ros2 topic pub --once /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}"
```

<img src="/assets/img/ros2/pubonce_turtle.png">

이 때, linear은 m/s 단위이고, angular는 rad/s 단위이다.

&nbsp;

지속적인 발행을 원한다면 --once 옵션을 제거하고 --rate옵션을 사용하여 Hz 단위로 발행한다.

```bash
ros2 topic pub --rate 1 /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}"
```

<img src="/assets/img/ros2/pubrate_turtle.png">

&nbsp;

&nbsp;

## bag 파일 저장

토픽을 파일 형태로 저장할 수 있다. 이 때 저장하는 파일의 형태가 bag 이고, 저장된 토픽은 다시 불러와 동일한 타이밍에 재생할 수 있다. 이를 **rosbag**이라 한다.

명령하는 방법은 `ros2 bag record <topic_name1> <topic_name2> <topic_name3>` 와 같이 사용하여 원하는 토픽만 기록할 수 있다.

```bash
ros2 bag record <topic_name1> <topic_name2> <topic_name3>
```

&nbsp;

`-a`를 사용하면 전체 토픽을 저장할 수 있다.

```bash
ros2 bag record -a
```

&nbsp;

원하는 이름이 있다면 -o 를 사용하여 이름을 지정한다.

```bash
ros2 topic pub --rate 1 /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}"
```

```bash
### ros2 bag record -o <file-name> <topic-name>
$ ros2 bag record -o test.bag turtle1/cmd_vel
[INFO]: Opened database 'test.bag/test.bag_0.db3' for READ_WRITE.
[INFO]: Listening for topics...
[INFO]: Subscribed to topic '/turtle1/cmd_vel'
[INFO]: All requested topics are subscribed. Stopping discovery...
```

&nbsp;

- bag 파일 정보 확인

저장된 bag파일의 정보를 확인할 수 있다.

```bash
$ ros2 bag info test.bag
Files:             test.bag_0.db3
Bag size:          16.8 KiB
Storage id:        sqlite3
Duration:          1.984s
Start:             Jul 26 2022 14:43:24.586 (1658814204.586)
End:               Jul 26 2022 14:43:26.571 (1658814206.571)
Messages:          9
Topic information: Topic: /turtle1/cmd_vel | Type: geometry_msgs/msg/Twist | Count: 9 | Serialization Format: cdr
```

&nbsp;

- bag 파일 재생

저장된 bag 파일을 재생하여 토픽을 발행할 수 있다.

```bash
$ ros2 bag play test.bag
[INFO]: Opened database 'test.bag/test.bag_0.db3' for READ_ONLY.
```

<img src="/assets/img/ros2/bagplay_turtle.png">

&nbsp;

&nbsp;

## ROS 인터페이스

ROS 노드 간에 데이터를 주고받을 때 토픽, 서비스, 액션이 사용되는데, 이 때 사용되는 데이터의 형태를 ROS 인터페이스(interface)라 한다. ROS 인터페이스에는 ROS2에 새롭게 추가된 IDL(Interface Definition Language)와 ROS1부터 존재하던 msg, srv, action 등이 있다.

즉, 토픽, 서비스, 액션은 각각 msg, srv, action interface를 사용하고 있으며 정수, 소수, bool 등과 같은 단순 자료형을 기본으로 하고, 메시지 안에 메시지를 품는 구조도 있다.

&nbsp;

### 메시지 인터페이스 (message interface, msg)

지금까지 다루던 토픽의 형태가 msg 분류의 데이터 형태이다. 예를 들어 Twist 데이터 형태를 자세히 보면 Vector3 linear와 Vector3 angular 가 있는데, 이것이 메시지 안에 메시지를 품고 있는 형태이다. Vector3안에는 또 다시 float64 형태의 x,y,z 값이 존재한다.

- geometry_msgs/msgs/Twist
  - Vector3 linear
    - float64 x
    - float64 y
    - float64 z
  - Vector3 angular
    - float64 x
    - float64 y
    - float64 z

&nbsp;

```bash
$ ros2 interface show geometry_msgs/msg/Twist
Vector3  linear
Vector3  angular

$ ros2 interface show geometry_msgs/msg/Vector3
float64 x
float64 y
float64 z
```

&nbsp;

interface 명령어에는 show 이외에도 **list**, **package**, **packages**, **proto**가 있다.

- list : 현재 개발 환경의 모든 msg, srv, action 메시지를 보여준다.
- package + \<package name> : 지정한 패키지에 포함된 인터페이스를 보여준다.
- packages : msg, srv, action 인터페이스를 담고 있는 패키지의 목록을 보여준다.
- proto + \<interface> : 특정 인터페이스 형태를 입력하면 인터페이스의 기본 형태를 표시해준다.

&nbsp;

```bash
$ ros2 interface list
Messages:
    action_msgs/msg/GoalInfo
    action_msgs/msg/GoalStatus
    action_msgs/msg/GoalStatusArray
    ...
Services:
    action_msgs/srv/CancelGoal
    composition_interfaces/srv/ListNodes
    composition_interfaces/srv/LoadNode
    ...
Actions:
    action_tutorials_interfaces/action/Fibonacci
    example_interfaces/action/Fibonacci
    tf2_msgs/action/LookupTransform
    turtlesim/action/RotateAbsolute
```

토픽에 해당되는 msg 인터페이스 이외에도 서비스, 액션 메시지도 존재하고 이를 srv, action 인터페이스라 한다.

&nbsp;

```bash
$ ros2 interface packages
action_msgs
action_tutorials_interfaces
...
```

&nbsp;

```bash
$ ros2 interface package turtlesim
turtlesim/srv/Spawn
turtlesim/msg/Pose
turtlesim/action/RotateAbsolute
turtlesim/srv/Kill
turtlesim/srv/TeleportAbsolute
turtlesim/srv/SetPen
turtlesim/srv/TeleportRelative
turtlesim/msg/Color
```

이 때, msg는 토픽, srv는 service, action은 action 인터페이스에 해당한다.

&nbsp;

```bash
$ ros2 interface proto geometry_msgs/msg/Twist
"linear:
  x: 0.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0
"
```

&nbsp;

