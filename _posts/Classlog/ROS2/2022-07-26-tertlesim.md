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

![](2022-07-26-04-38-10.png)

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

