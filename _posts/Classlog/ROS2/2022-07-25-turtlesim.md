---
title:    "[ROS2 이론] turtlesim 실행"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-07-25 23:23:00 +0800
categories: [Classlog, ROS2]
tags: [ROS2]
toc: True
comments: True
--- 

| 개발 환경을 설정하는 것은 [해당 블로그](https://dkssud8150.github.io/posts/introduction/)를 확인해보길 바란다.

&nbsp;

- [007 패키지 설치와 노드 실행](https://cafe.naver.com/openrt/24065)

# Turtlesim 이란?

| 추가 예정.

&nbsp;

&nbsp;

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

<img src="/assets/img/ros2/turtlesim/turtlesim1.png">

<img src="/assets/img/ros2/turtlesim/turtlesim2.png">

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

<img src="/assets/img/ros2/turtlesim/topic.png">

그림에서처럼 NodeA - NodeB, NodeA - NodeC 처럼 비동기식 단방향 메시지 송수신 방식을 **토픽**(topic)이라 하고, 메시지를 발간하는 *Publisher*, 메시지를 구독하는 *Subscriber*가 존재한다.

&nbsp;

- 서비스

<img src="/assets/img/ros2/turtlesim/topic_service.png">

NodeB - NodeC 처럼 동기식 양방향 메시지 송수신 방식을 **서비스**(Service)라 하고, 서비스를 요청(request)하는 쪽을 *service client*라 하고, 서비스를 응답(response)하는 쪽을 *service server*라 한다.

&nbsp;

- 액션

<img src="/assets/img/ros2/turtlesim/topic_service_action.png">

NodeA - NodeB 처럼 비동기식 + 동기식 양방향 메시지 송수신 방식을 액션(action)이라 하고, 액션 목표(Goal)을 지정하는 **action client**와 액션 목표를 받아 특정 태스크를 수행하면서 중간 결과값에 해당하는 액션 피드백(Feedback)과 최종 결과값에 해당하는 액션 결과(result)를 전송하는 action server간의 통신을 액션(action)이라 볼 수 있다.

즉 `Action Goal -> Action Feedback -> Action Result` 로 진행된다.

액션의 구현 방식을 더 자세히 살펴보면 토픽과 서비스의 혼합이라 볼 수 있는데, 액션 목표와 액션 결과를 전달하는 방식은 서비스와 같고, 액션 피드백은 토픽과 같은 메시지 전송 방식이다. 그래서 피드백의 퍼블리셔는 Action Server라 할 수 있고, 피드백의 서브스크라이버는 Action client와 service client라 할 수 있다.

&nbsp;

- 파라미터

<img src="/assets/img/ros2/turtlesim/parameter.png">

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

<img src="/assets/img/ros2/turtlesim/turtlesim_rqt_graph.png">

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

<img src="/assets/img/ros2/turtlesim/turtlesim_rqt_graph2.png">

&nbsp;

노드 이름만 변경되었고, 토픽인 `turtle1/cmd_vel`은 동일하다. 만약 teleop_turtle 노드를 이용하여 거북이를 움직이면 두 개의 노드의 거북이가 동일하게 움직인다. 이는 동일한 토픽을 이용하기 때문이다. 이 토픽 또한 토픽명을 변경하거나 name_space를 통해 바꿀 수 있다.

<img src="/assets/img/ros2/turtlesim/turtlesim_rqt_graph3.png">

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

- [009 ROS2 토픽 (topic)](https://cafe.naver.com/openrt/24101)

토픽은 비동기식 단방향 메시지 송수신 방식으로 메시지 형태로 메시지를 발행하는 퍼블리셔(publisher)와 메시지를 구동하는 서브스크라이버(subscriber) 간의 통신이라 볼 수 있다. 이는 1:1 통신을 기본으로 하지만, 1:N도 가능하고, 구성에 따라 N:1, N:N 통신도 가능하다.

하나의 노드가 퍼블리셔와 서브스크라이버를 동시에 수행할 수도 있다. 원한다면 자신이 발행한 토픽을 셀프 구독할 수 있게 구성할 수도 있다.

<img src="/assets/img/ros2/turtlesim/node_and_topic.png">

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

<img src="/assets/img/ros2/turtlesim/pubonce_turtle.png">

이 때, linear은 m/s 단위이고, angular는 rad/s 단위이다.

&nbsp;

지속적인 발행을 원한다면 --once 옵션을 제거하고 --rate옵션을 사용하여 Hz 단위로 발행한다.

```bash
ros2 topic pub --rate 1 /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}"
```

<img src="/assets/img/ros2/turtlesim/pubrate_turtle.png">

&nbsp;

&nbsp;

거북이를 조종할 때 `← ↑ ↓ →` 키를 사용하여 움직일 수 있는데, 터미널에도 나오듯이 F를 기준으로 `G,B,V,C,D,E,R,T`를 사용해서도 거북이를 움직일 수 있다.

<img src="/assets/img/ros2/turtlesim/movement_f.png">

R키를 누르면, 1.57 radian 방향인 π/2, 즉 위 방향을 바라보게 된다.



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

<img src="/assets/img/ros2/turtlesim/bagplay_turtle.png">

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

&nbsp;

# Turtlesim 서비스

- [010 ROS2 서비스 (service)](https://cafe.naver.com/openrt/24128)

&nbsp;

서비스는 동기식 양방향 메시지 송수신 방식으로 서비스의 요청(request)을 하는 쪽을 **service client**라 하며, 요청받은 서비스를 수행 후 서비스의 응답(response)을 하는 쪽을 **service server**라 한다.

결국 서비스는 특정 요청을 하는 클라이언트와 요청받은 일을 수행한 후 결괏값을 전달하는 서버 간의 통신이다. **서비스 요청 및 응답** 또한, msg 인터페이스의 변형인 **srv 인터페이스**라 한다.

<img src="/assets/img/ros2/turtlesim/service.png">

&nbsp;

서비스는 동일 서비스에 대해 복수의 클라이언트를 가질 수 있다. 단 응답은 서비스 요청이 있었던 클라이언트에 대해서만 응답하는 형태로, 위의 그림에서 Node C의 클라이언트가 Node B의 서버에게 요청을 했다면, B의 서버는 요청받은 서비스를 수행한 후 Node C의 클라이언트에게만 응답을 하게 된다.

&nbsp;

- 서비스 목록 확인

turtlesim 패키지를 통해 서비스 목록을 살펴보도록 한다. 목록을 살펴볼 때는 topic이랑 동일하게 `ros2 service list`로 확인할 수 있다.

&nbsp;

```bash
$ ros2 run turtlesim turtlesim_node
[INFO]: Starting turtlesim with node name /turtlesim
[INFO]: Spawning turtle [turtle1] at x=[5.544445], y=[5.544445], theta=[0.000000]
```

```bash
$ ros2 service list
/clear
/kill
/reset
/spawn
/turtle1/set_pen
/turtle1/teleport_absolute
/turtle1/teleport_relative
/turtlesim/describe_parameters
/turtlesim/get_parameter_types
/turtlesim/get_parameters
/turtlesim/list_parameters
/turtlesim/set_parameters
/turtlesim/set_parameters_atomically
```

파라미터가 붙어있는 부분을 제외한 7가지의 서비스에 대해 자세하게 알아보자.

&nbsp;

- 서비스 형태 확인

서비스 형태 확인 명령어는 `ros2 service type <service name>` 이다. 그러나 더 쉽게 확인하는 방법은 `ros2 service list -t`를 사용할 수 있다.

```bash
$ ros2 service type /clear
std_srvs/srv/Empty

$ ros2 service type /kill
turtlesim/srv/Kill
```

```bash
$ ros2 service list -t
/clear [std_srvs/srv/Empty]
/kill [turtlesim/srv/Kill]
/reset [std_srvs/srv/Empty]
/spawn [turtlesim/srv/Spawn]
/turtle1/set_pen [turtlesim/srv/SetPen]
/turtle1/teleport_absolute [turtlesim/srv/TeleportAbsolute]
/turtle1/teleport_relative [turtlesim/srv/TeleportRelative]
...
```

clear 서비스는 std_srvs/srv/Empty 형태이고, kill 서비스는 turtlesim/srv/Kill 형태임을 확인할 수 있다.

&nbsp;

- 서비스 찾기

서비스 형태를 통해 서비스명을 찾을 수 있다. `ros2 service find <service type>` 와 같이 서비스 형태를 넣으면 서비스명을 찾는다.

&nbsp;

## 서비스 요청

실제 서비스 서버에게 서비스를 요청(request)해본다. 서비스 요청은 `ros2 service call <service_name> <service_type> "<arguments>"`를 통해 보낼 수 있다. **/clear** 서비스는 turtlesim 노드를 동작할 때 표시되는 이동 궤적을 지우는 서비스이다.

```bash
$ ros2 run turtlesim turtle_teleop_key
```

<img src="/assets/img/ros2/turtlesim/turtle_move.png">

&nbsp;

```bash
$ ros2 service call /clear std_srvs/srv/Empty
waiting for service to become available...
requester: making request: std_srvs.srv.Empty_Request()

response:
std_srvs.srv.Empty_Response()
```

뒤에 argument 부분이 없는 이유는 해당 서비스가 아무런 내용이 없는 형태로도 사용이 가능하기 때문이다.

<img src="/assets/img/ros2/turtlesim/turtle_clear.png">

&nbsp;

**/kill** 서비스는 죽이고자 하는 거북이 이름을 서비스 요청의 내용을 입력하면 거북이가 사라지게 된다.

```bash
$ ros2 service call /kill turtlesim/srv/Kill "name: 'turtle1'"
waiting for service to become available...
requester: making request: turtlesim.srv.Kill_Request(name='turtle1')

response:
turtlesim.srv.Kill_Response()
```

<img src="/assets/img/ros2/turtlesim/turtle_kill.png">

&nbsp;

이번에는 **/reset** 서비스를 해보자. 이 서비스는 단어 그대로 모든 것을 리셋하여 거북이를 처음 상태로 되돌린다.

```bash
$ ros2 service call /reset std_srvs/srv/Empty
waiting for service to become available...
requester: making request: std_srvs.srv.Empty_Request()

response:
std_srvs.srv.Empty_Response()
```

<img src="/assets/img/ros2/turtlesim/turtle_reset.png">

&nbsp;

이제 **/set_pen** 을 통해 거북이의 궤적의 색과 크기를 지정해보자. r,g,b, width, offset을 통해 변경이 가능하다/

```bash
$ ros2 service call /turtle1/set_pen turtlesim/srv/SetPen "{r: 255, g: 255, b: 255, width: 10}"
```

<img src="/assets/img/ros2/turtlesim/turtle_set_pen.png">

&nbsp;

마지막으로 **/spawn** 서비스를 사용하여 해당 위치 및 자세에 맞게 거북이를 추가한다. 이름 옵션을 지정하지 않으면 자동으로 지정된다.

```bash
$ ros2 service call /spawn turtlesim/srv/Spawn "{x: 5.5, y: 9, theta: 1.57, name: 'leonardo'}"
waiting for service to become available...
requester: making request: turtlesim.srv.Spawn_Request(x=5.5, y=9.0, theta=1.57, name='leonardo')

response:
turtlesim.srv.Spawn_Response(name='leonardo')
```

<img src="/assets/img/ros2/turtlesim/turtle_spawn.png">

&nbsp;

- 서비스 인터페이스

서비스 또한 토픽과 마찬가지로 인터페이스를 가지고 있고, 파일로는 srv파일을 가르킨다. 서비스 인터페이스는 메시지 인터페이스의 확장형이라 볼 수 있다.

```bash
$ ros2 interface show turtlesim/srv/Spawn.srv
float32 x
float32 y
float32 theta
string name # Optional.  A unique name will be created and returned if this is empty
---
string name
```

Spawn.srv 에는 float32 형태의 x,y,theta, string 형태의 name 이라는 데이터가 들어 있다. 여기서 특이한 것은 `---`인데, 이는 구분자라 하여 요청과 응답을 나누어 사용하기 위한 구분이라 할 수 있다. 즉 구분자 위의 x,y,theta, name 부분은 서비스 요청에 해당하여 클라이언트에서 서버로 전송하는 값이다. 아래의 name은 응답 부분에 해당하여 지정된 서비스 요청을 수행하고, name 데이터를 클라이언트에 전송한다.

Spawn 서비스를 사용하려면 위치인 x,y와 자세인 theta, 거북이 이름인 name를 통해 거북이를 생성하고, 응답 부분의 name에 신규 거북이의 이름이 저장된다.

&nbsp;

&nbsp;

# Turtlesim 액션

- [011 ROS2 액션 (action)](https://cafe.naver.com/openrt/24142)

&nbsp;

<img src="/assets/img/ros2/turtlesim/action.png">

비동기식 + 동기식 양방향 메시지 송수신 방식으로 액션 **목표(goal)을 지정하는 액션 클라이언트**와 액션 목표를 받아 특정 태스크를 수행하면서 **중간 결괏값에 해당되는 액션 피드백**과 최종 결괏값에 해당되는 **액션 결과를 전송하는 맥션 서버 간의 통신**이라 할 수 있다.

&nbsp;

<img src="/assets/img/ros2/turtlesim/action2.png">

더 자세히 들여다보면, 액션 클라이언트는 서비스 클라이언트 3개와 서브스크라이버 2개로 구성되어 있고, 액션 서버는 서비스 서버 3개와 퍼블리셔 2개로 구성되어 있다. 액션 데이터는 action 인터페이스라 한다.

&nbsp;

ROS1에서의 액션은 목표, 피드백, 결과 값을 토픽으로만 주고 받았는데, ROS2에서는 토픽과 서비스 방식을 혼합하여 사용했다. 그 이유는 토픽으로만 액션을 구성했을 때는 비동기식 방식으로 구성되는데 반해, ROS2는 서비스와 토픽 방식의 혼합으로 인한 동기식 방식이므로 원하는 타이밍에 적절한 액션을 수행하게 되었다.

ROS에서는 **목표 상태**(goal_state)라는 것이 존재하는데, 이는 목표 값을 전달한 후의 상태 머신을 구동하여 액션의 프로세스를 쫓는다. 여기서 **상태머신**(Goal State Machine)은 액션 목표 전달의 이후 **액션의 상태 값을 액션 클라이언트에게 전달**하여 액션이 원할하게 동작할 수 있도록 한다.

&nbsp;

- 상태 머신 구조

<img src="/assets/img/ros2/turtlesim/goal_state_machine.png">

&nbsp;

위에서 G,B,V,C,D,E,R,T 키를 사용하는 것이 rotate_absolute 액션을 수행하는 것이고, 이 키들은 액션의 목표 값을 전달하는 목적으로 사용된다. F 액션 목표를 취소하는 키이다.

```bash
$ ros2 node info /turtlesim
/turtlesim
  ...
  Action Servers:
    /turtle1/rotate_absolute: turtlesim/action/RotateAbsolute
  Action Clients:
```

turtlesim 노드는 액션 서버 역할을 하고, `turtlesim/action/RotateAbsolute` 이라는 action 인터페이스를 사용하는 /turtle1/rotate_absolute 라는 이름의 액션 서버임을 확인할 수 있다.

&nbsp;

만약 위의 키들로 액션을 수행하게 되면, `ros2 run turtlesim turtlesim_node`를 실행한 터미널에 아래와 같이 표시된다.

```bash
$ ros2 node info /teleop_turtle
/teleop_turtle
  ...
  Action Servers:

  Action Clients:
    /turtle1/rotate_absolute: turtlesim/action/RotateAbsolute
```

`teleop_turtle` 노드가 액션 클라이언트 역할을 한다.

&nbsp;

&nbsp;

```bash
[INFO]: Rotation goal completed successfully
```

그러나 전달되는 과정에서 F 키를 눌러 액션 목표를 취소하게 되면 취소되었다고 표시되고, 수행하던 행동을 멈춘다.

```bash
[INFO]: Rotation goal canceled
```

&nbsp;

rotate_absolute 액션을 더 세분화하면 5가지로 구성된다.

```bash
/turtle1/rotate_absolute/_action/send_goal: turtlesim/action/RotateAbsolute_SendGoal

/turtle1/rotate_absolute/_action/cancel_goal: action_msgs/srv/CancelGoal

/turtle1/rotate_absolute/_action/status: action_msgs/msg/GoalStatusArray

/turtle1/rotate_absolute/_action/feedback: turtlesim/action/RotateAbsolute_FeedbackMessage

/turtle1/rotate_absolute/_action/get_result: turtlesim/action/RotateAbsolute_GetResult
```

추가적인 내용은 추후 설명한다고 한다.

&nbsp;

- 액션 정보 확인

`ros2 action info` 를 통해 액션의 정보를 확인할 수 있는데, 여기에는 액션의 이름, 액션의 서버와 클라이언트 노드의 이름과 개수를 확인할 수 있다.

```bash
$ ros2 action info /turtle1/rotate_absolute
Action: /turtle1/rotate_absolute
Action clients: 1
    /teleop_turtle
Action servers: 1
    /turtlesim
```

&nbsp;

- 액션 목표(action goal) 전달

`ros2 action send_goal <action name> <action type> "<value>"`를 통해 action을 보낼 수 있다.

```bash
$ ros2 action send_goal /turtle1/rotate_absolute turtlesim/action/RotateAbsolute "{theta: 1.5708}"
Waiting for an action server to become available...
Sending goal:
     theta: 1.5708

Goal accepted with ID: 760369ad7f7f4cc8af2618fa688b65f6

Result:
    delta: -0.16000032424926758

Goal finished with status: SUCCEEDED
```

거북이를 12시 방향인 1.57 radian 값을 목표로 주게 되면 전달한 목표 값과 액션 목표의 UID(Unique ID), 시작 위치부터 결괏값에 해당하는 위치까지의 변위 값인 delta를 결과로 보여준다. 그리고 전달 상태를 표시하게 된다.

여기에 `--feedback`이라는 옵션을 주면, 거북이가 목표값까지의 남은 회전량을 표시해준다.

```bash
$ ros2 action send_goal /turtle1/rotate_absolute turtlesim/action/RotateAbsolute "{theta: 1.5708}" --feedback
Waiting for an action server to become available...
Sending goal:
     theta: 1.5708

Feedback:
    remaining: 1.554800033569336

Goal accepted with ID: 21a4fdd053f84ea1935a9b3c582c9ac6

Feedback:
    remaining: 1.5388000011444092

Feedback:
    remaining: 1.5227999687194824

Feedback:
    remaining: 1.5067999362945557
...
Feedback:
    remaining: 0.05079972743988037

Feedback:
    remaining: 0.03479969501495361

Feedback:
    remaining: 0.018799662590026855

Result:
    delta: -1.536000370979309

Goal finished with status: SUCCEEDED
```

&nbsp;

- 액션 인터페이스

토픽, 서비스와 마찬가지로 액션도 인터페이스를 가지고 있다. action 파일이 이에 해당되고, 액션 인터페이스는 메시지 및 서비스 인터페이스의 확장형이라 볼 수 있다.

`ros2 interface show <action name>` 명령어로 액션 인터페이스의 정보를 확인할 수 있는데, float32 형태의 theta, delta, remaining 이라는 세 개의 데이터가 있다. 여기도 마찬가지로 `---`라는 구분자가 있고, 맨 위부터 액션 목표(goal), 액션 결과(result), 액션 피드백(feedback)으로 나누어져 있다. 즉, theta는 액션 목표, delta는 액션 결과, 액션 피드백(feedback)이다. 모든 데이터는 radian 단위를 사용한다.

```bash
$ ros2 interface show turtlesim/action/RotateAbsolute.action
# The desired heading in radians
float32 theta
---
# The angular displacement in radians to the starting position
float32 delta
---
# The remaining rotation in radians
float32 remaining
```

&nbsp;

&nbsp;

# ROS2 토픽/서비스/액션 정리

- [012 ROS2 토픽/서비스/액션 정리 및 비교](https://cafe.naver.com/openrt/24154)

&nbsp;

- 토픽, 서비스, 액션 비교

| | 토픽 (topic)​ | 서비스 (service) | 액션 (action)
| --- | --- | --- | --- |
연속성 | 연속성 | 일회성 | 복합 (토픽+서비스)
방향성 | 단방향 | 양방향 | 양방향
동기성 | 비동기 | 동기 | 동기 + 비동기
다자간 연결 | 1:1, 1:N, N:1, N:N(publisher:subscriber) | 1:1(server:client) | 1:1(server:client)
노드 역할 | 발행자 (publisher), 구독자 (subscriber) | 서버 (server), 클라언트 (client) | 서버 (server), 클라언트 (client)
동작 트리거 | 발행자 | 클라언트 | 클라언트
인터페이스 | msg 인터페이스 | srv 인터페이스 | action 인터페이스
CLI 명령어 | ros2 topic, ros2 interface | ros2 service, ros2 interface | ros2 action, ros2 interface
사용 예 | 센서 데이터, 로봇 상태, 로봇 좌표, 로봇 속도 명령 등 | LED 제어, 모터 토크 On/Off, IK/FK 계산, 이동 경로 계산 등 | 목적지로 이동, 물건 파지, 복합 태스크 등

&nbsp;

- msg, srv, action 인터페이스 비교

| | msg 인터페이스 | srv 인터페이스 | action 인터페이스 |
| --- | --- | --- | --- |
확장자 | *.msg | *.srv | *.action
데이터 | 토픽 데이터 (data) | 서비스 요청 (request), 서비스 응답 (response) | 액션 목표 (goal), 액션 결과 (result), 액션 피드백 (feedback)
형식 | fieldtype1 fieldname1, fieldtype2 fieldname2, fieldtype3 fieldname3 | fieldtype1 fieldname1, fieldtype2 fieldname2, fieldtype3 fieldname3, fieldtype4 fieldname4 | fieldtype1 fieldname1, fieldtype2 fieldname2, fieldtype3 fieldname3, fieldtype4 fieldname4, fieldtype5 fieldname5, fieldtype6 fieldname6
사용 예 | [geometry_msgs/msg/Twist], Vector3 linear, Vector3 angular | [turtlesim/srv/Spawn.srv], float32 x, float32 y , float32 theta , string name --- string name | [turtlesim/action/RotateAbsolute.action], float32 theta, float32 delta, float32 remaining

&nbsp;

&nbsp;