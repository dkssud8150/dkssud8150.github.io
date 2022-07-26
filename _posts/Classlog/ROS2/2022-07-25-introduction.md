---
title:    "[ROS2 이론] 목차 및 개요"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-07-25 17:35:00 +0800
categories: [Classlog, ROS2]
tags: [ROS2]
toc: True
comments: True
---

표윤석 박사님의 네이버 카페 글을 참고로 하여 ROS2를 공부하고, 기록하고자 합니다.

- 기본 URL : https://cafe.naver.com/openrt/24070
- github : https://github.com/robotpilot/ros2-seminar-examples

&nbsp;

목차는 기본적으로 이론/프로그래밍으로 나뉘어져 있고, 심화 프로그래밍과 기타 등이 존재합니다. 저는 이론/프로그래밍을 위주로 공부할 예정입니다.

책으로도 존재하니 책으로 보고 싶으신 분들은 책으로 공부하시면 될 것 같습니다. 위의 참고 자료는 온라인 강의로 사용된 PDF파일 600페이지 분량이라고 합니다.

- [참고 교재 - ROS 로봇 프로그래밍](http://book.naver.com/bookdb/book_detail.nhn?bid=12443870)

&nbsp;

또한, 저는 일단 wsl2로 진행할 예정이니 오리지널과 다소 차이가 있을 수 있습니다. ROS2는 윈도우에서도 사용이 가능하긴 하나 본래의 ROS가 ubuntu에서 주로 사용하기에 저도 ubuntu 환경에서 사용하기 위해 wsl2를 사용하는 것입니다.

- wsl2에 ROS2 설치 참고 자료 : https://keep-steady.tistory.com/45

&nbsp;

&nbsp;

# why ROS2?

- [003 왜? ROS 2로 가야하는가?](https://cafe.naver.com/openrt/23868)
- [004 ROS 2의 중요 컨셉과 특징](https://cafe.naver.com/openrt/23889)
- [005 ROS1과 2의 차이점을 통해 알아보는 ROS2의 특징](https://cafe.naver.com/openrt/23965)

ROS1을 배우지 않았다면, 그냥 ROS2를 배워도 된다. 그러나 ROS1을 사용하던 사람이 ROS2를 사용할 이유는 잘 모를 수 있다. 

ROS2는 2014년 3월에 개발을 시작하여 2015년도에 알파 버전이 릴리즈되었으며, 1년 동안 8번의 릴리즈가 발표되었고, 2017년에 베타 버전이 작업되어 2017년 12월에 첫 공식 버전이 나왔다. 지금은 총 6개의 공식 배포판이 나와있다.

ROS1에서 ROS2를 갈아타야하는 가장 큰 이유는 ROS1의 최후의 버전이 발표되었기 때문이다. ROS1의 마지막 버전인 Noetic Ninjemys는 2020년 05월 23일에 릴리즈되었다. 이 릴리즈가 마지막 버전이라는 선언을 했다.

ROS1은 Linux만을 주로 지원하고 있었는데, ROS2는 Linux, windows, macOS 등 다양하게 지원하고 있다. 

&nbsp;

- ROS1과 ROS2의 차이점

<img src="/assets/img/ros2/ros1ros2.png">

&nbsp;

&nbsp;

# 1. 개발 환경 설정

- [001 ROS 2 개발 환경 구축](https://cafe.naver.com/openrt/25288)

위에서 말했다시피 먼저 WSL2에서 ROS2를 설치하는 과정을 먼저 설명하고자 한다. wsl2가 설치되어 있다는 가정하에 시작한다.

## ubuntu 환경 설정

```bash
sudo apt update && sudo apt upgrade
```

&nbsp;

- 지역 변수 설정

```bash
$ sudo locale-gen en_US en_US.UTF-8
$ sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
$ export LANG=en_US.UTF-8
```

&nbsp;

- 지역 변수 확인

```bash
$ locale
LANG=en_US.UTF-8
LANGUAGE=
LC_CTYPE="en_US.UTF-8"
LC_NUMERIC="en_US.UTF-8"
...
```

&nbsp;

- ROS2 환경 변수 설정

```bash
$ vim ~/.bashrc
```

vim이나 nano, gedit 등 자신이 원하는 편집기를 사용하면 된다. 기존에 존재했던 것들은 그대로 두고, 아래 부분을 추가한다. source, export는 모두 추가해주어야 하며, alias는 자신이 원하는 대로 추가하면 된다.

<details open>
    <summary> 내용 </summary>

```markdown
source /opt/ros/foxy/setup.bash
source ~/robot_ws/install/local_setup.bash

source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
source /usr/share/vcstool-completion/vcs.bash
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=~/robot_ws

export ROS_DOMAIN_ID=7
export ROS_NAMESPACE=robot1

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# export RMW_IMPLEMENTATION=rmw_connext_cpp
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# export RMW_IMPLEMENTATION=rmw_gurumdds_cpp

# export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity} {time}] [{name}]: {message} ({function_name}() at {file_name}:{line_number})'
export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}]: {message}'
export RCUTILS_COLORIZED_OUTPUT=1
export RCUTILS_LOGGING_USE_STDOUT=0
export RCUTILS_LOGGING_BUFFERED_STREAM=1

alias cw='cd ~/robot_ws'
alias cs='cd ~/robot_ws/src'
alias ccd='colcon_cd'

alias cb='cd ~/robot_ws && colcon build --symlink-install'
alias cbs='colcon build --symlink-install'
alias cbp='colcon build --symlink-install --packages-select'
alias cbu='colcon build --symlink-install --packages-up-to'
alias ct='colcon test'
alias ctp='colcon test --packages-select'
alias ctr='colcon test-result'

alias rt='ros2 topic list'
alias re='ros2 topic echo'
alias rn='ros2 node list'

alias killgazebo='killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient'

alias af='ament_flake8'
alias ac='ament_cpplint'

alias testpub='ros2 run demo_nodes_cpp talker'
alias testsub='ros2 run demo_nodes_cpp listener'
alias testpubimg='ros2 run image_tools cam2image'
alias testsubimg='ros2 run image_tools showimage'
```

</details>

&nbsp;

- IDE 설치

표윤석 박사님의 블로그에 보면, IDE로 vscode를 사용하는 것을 권장한다. vscode가 linux, window, macOS 모두 사용 가능한 크로스 플랫폼 에디터인 동시에 다양한 프로그래밍 언어를 지원하고 있고, `.pcd`와 같은 특수한 파일을 볼 수 있는 뷰어도 지원하고 있다.

추천해주시는 vscode에 extensions은 정말 많다. 필요한 것은 알아서 설치하길 바란다.

| 이름 | 설명 |
| --- | --- |
| C/C++ | C/C++ intellisense, 디버깅 및 코드 검색 |
| CMake | CMake 언어 지원 |
| CMake Tools | CMake 언어 지원 및 다양한 툴 |
| Python | 디버깅, intellisense, 코드 서식 지정, 리팩토딩 등 지원 |
| ROS | ROS 개발 지원 |
| URDF | URDF/xacro 지원 |
| Colcon Tasks | Colcon 명령어를 위한 VScode Task |

이 외에도 많으니 [https://cafe.naver.com/openrt/25288](https://cafe.naver.com/openrt/25288)를 참고하길 바란다.

&nbsp;

- 패키지 설치 및 gpg 설정 및 소스 설정

```bash
$ sudo apt update && sudo apt install curl gnupg2 lsb-release -y
$ sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
$ sudo apt update
```

&nbsp;

이 때, PUBKEY 에러가 나면 아래 코드를 실행한다.

참고 URL : https://answers.ros.org/question/325039/apt-update-fails-cannot-install-pkgs-key-not-working/?answer=325040#post-id-325040

```bash
#### remove old key
$ sudo apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116
OK

#### install new key
$ sudo -E apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

$ sudo apt update
```

&nbsp;

그래도 에러가 나면

```bash
$ curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
$ sudo apt update
```

or 

```bash
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
$ sudo apt update
```

&nbsp;

## ROS2 패키지 설치

```bash
$ sudo apt update
$ sudo apt install ros-foxy-desktop ros-foxy-rmw-fastrtps* ros-foxy-rmw-cyclonedds* -y
```

&nbsp;

- 패키지 설치 확인

```bash
$ echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
$ source ~/.bahsrc

$ ros2 run demo_nodes_cpp talker
[INFO] [1658742890.693325028] [talker]: Publishing: 'Hello World: 1'
[INFO] [1658742891.693441287] [talker]: Publishing: 'Hello World: 2'
[INFO] [1658742892.693642848] [talker]: Publishing: 'Hello World: 3'
...

$ ros2 run demo_nodes_py listener
[INFO] [1612912265.593335793] [listener]: I heard: [Hello World: 3]
[INFO] [1612912266.576514520] [listener]: I heard: [Hello World: 4]
[INFO] [1612912267.576780341] [listener]: I heard: [Hello World: 5]
[INFO] [1612912268.576769156] [listener]: I heard: [Hello World: 6]
[INFO] [1612912269.577142775] [listener]: I heard: [Hello World: 7]
...
```

이 두가지를 동시에 실행했을 때 문제가 없다면 ros2가 잘 설치된 것이다.

&nbsp;

&nbsp;

## ROS2 개발 툴 설치

로봇 프로그래밍에 필수로 사용되는 패키지들이다. 개발 환경 설치시 설치해두는 것이 좋다.

```bash
$ sudo apt update && sudo apt install -y build-essential cmake git libbullet-dev python3-colcon-common-extensions python3-flake8 python3-pip python3-pytest-cov python3-rosdep python3-setuptools python3-vcstool wget

$ python3 -m pip install -U argcomplete flake8-blind-except flake8-builtins flake8-class-newline flake8-comprehensions flake8-deprecated flake8-docstrings flake8-import-order flake8-quotes pytest-repeat pytest-rerunfailures pytest

$ sudo apt install --no-install-recommends -y libasio-dev libtinyxml2-dev libcunit1-dev
```

&nbsp;

&nbsp;

## ROS2 빌드 테스트

```bash
$ mkdir -p ~/robot_ws/src
$ cd ~/robot_ws
$ colcon build --symlink-install
```

&nbsp;

- 확인

<img src="/assets/img/ros2/colcon.png">

위 과정을 진행하면 워크스페이스 폴더에 src, build, install, log 폴더가 생성된 것을 확인할 수 있다.


| colcon이 뭐지?

&nbsp;

## VSCode의 개발 환경 설정

- User settings 설정

settings.json 은 VScode의 사용자별 글로벌 환경 설정을 지정하는 파일이다. 이 파일에 기술된 설정들은 모든 작업 공간에 적용되는 미니맵 사용, 세로 제한 줄 표시, 탭 사이즈 등이다.

| 추가 예정

&nbsp;

- C/C++ properties 설정

| 추가 예정

&nbsp;

- Tasks 설정

| 추가 예정

&nbsp;

- Launch 설정

| 추가 예정

&nbsp;

## QtCreator 설치

QtCreator는 ROS2의 rqt UI를 작성할 때 편리한 툴이다.

```bash
$ sudo apt install qtcreator -y

$ qtcreator
```

<img src="/assets/img/ros2/qtcreator.png">

&nbsp;

&nbsp;

# 2. ROS2와 Data Distribution Service

- [006 ROS2와 DDS](https://cafe.naver.com/openrt/24031)

ROS에 중요한 용어 정의 및 메시지, 메시지 통신에 대해 먼저 알아보고자 한다. 메시지 통신은 ROS 프로그래밍에 있어서 ROS1과 2의 공통된 중요한 핵심 개념이다.

ROS에는 프로그램의 재사용성을 극대화하기 위해 최소 단위의 실행 가능한 프로세서라고 정의하는 노드 단위로 프로그램을 작성하게 된다. 

- **패키지**(package) : 하나 이상의 노드 또는 노드 실행을 위한 정보 등을 묶어 놓은 것
- **메타패키지**(metapackage) : 패키지의 묶음
- **메시지**(message) : 노드와 노드 간 주고 받는 입력과 출력 데이터
    - 주고받는 방식을 메시지 통신이라 한다.
    - 메시지는 *int*, *float*, *point*, *bool*, *string* 등 다양한 변수 형태로 존재
    - 데이터 구조 및 메시지의 배열을 정할 수 있다.
    - 메시지를 주고받는 통신 방법에 따라 **토픽**(topic), **서비스**(service), **액션**(action), **파라미터**(parameter)로 구분된다. 
- **퍼블리셔** : 메시지를 보내는 노드
- **서브스크라이버** : 메시지를 받는 노드

&nbsp;

ROS1은 자체적으로 개발한 TCPROS와 같은 통신 라이브러리를 사용하고 있는 방면, ROS2에서는 OMG(Object Management Group)에 의해 표준화된 DDS(Data Distribution Service)의 퍼블리셔와 서브스크라이브 프로토콜인 **DDSI-RTPS**(Real Time Publish Subscribe)를 사용한다. DDSI-RTPS는 DDS의 중요 컨셉인 DCPS(Data-Centric Publish-Subscribe), DLRL(Data Local Reconstruction Layer)의 내용을 담아 재정한 통신프로토콜이다. DDS통신을 사용하는 이유는 ROS1과 같이 자체적으로 제작하기보다 산업용 표준인 DDS를 통신 미들웨어로 사용하기 위해서이다.

<img src="/assets/img/ros2/ros2node.png">

<img src="/assets/img/ros2/ros2structure.png">

DDS의 사용으로 노드 간 실시간 데이터 전송이 보장되고, 노드 간 동적 검색 기능을 지원하여 ROS1에서 노드를 관리하던 마스터 노드가 없이도 통신이 가능해졌다. 또한, TCP에서의 데이터 손실을 방지하고, 신뢰도를 높였다. 

&nbsp;

## DDS란?

그렇다면 과연 DDS가 무엇일까? DDS는 한국말로 데이터 분산 시스템인데, OMG 에서 표준을 정하고자 만든 트레이드마크이다. 간단하게는 데이터 통신을 위한 미들웨어이다. 

OMG DDS Foundation에서 정의한 DDS는 다음과 같다.

> The Data Distribution Service (DDS™) is a middleware protocol and API standard for data-centric connectivity from the Object Management Group® (OMG®). It integrates the components of a system together, providing low-latency data connectivity, extreme reliability, and a scalable architecture that business and mission-critical Internet of Things (IoT) applications need.
>
>In a distributed system, middleware is the software layer that lies between the operating system and applications. It enables the various components of a system to more easily communicate and share data. It simplifies the development of distributed systems by letting software developers focus on the specific purpose of their applications rather than the mechanics of passing information between applications and systems.

*출처 : https://www.dds-foundation.org/what-is-dds-3/*

&nbsp;

즉, DDS는 데이터를 중심으로 연결성을 같는 미들웨어이다. 이 미들웨어는 ISO 7계층에서 호스트 계층에 해당되는 4~7계층에 해당된다.

&nbsp;

## DDS의 특징

ROS2의 미들웨어로 사용할 때의 장점을 살펴보자.

- 산업 표준

DDS는 1989년 설립된 비영리 단체인 OMG(Object Management Group)가 관리하는 **산업 표준**이다. ROS1은 자체적인 TCPROS였기에, ROS2의 DDS 사용은 다양한 산업으로 넓혀갈 수 있는 발판이 될 수 있다.

&nbsp;

- OS 독립

DDS는 Linux, windows, macOS, Android 등 다양한 운영체제를 지원하고 있다.

&nbsp;

- 언어 독립

DDS는 미들웨어이므로 그 상위 레벨인 **사용자 코드 레벨에서는 DDS 사용을 위해 프로그래밍 언어를 바꿀 필요가 없다**. ROS2에서도 DDS를 RMW(ROS Middleware)로 디자인되어, 다양한 언어를 지원하는 ROS 클라이언트 라이브러리(ROS client Library)를 제작하여 멀티 프로그래밍 언어를 지원하고 있다. 각 언어에 맞게 rclpy, rclcpp, rclc, rcljava 와 같은 이름으로 지원한다.

<img src="/assets/img/ros2/rmw.png">

&nbsp;

- UDP 기반의 전송 방식

일반적으로 UDP 기반의 신뢰성 있는 멀티캐스트를 구현하여 시스템이 최신 네트워킹 인프라의 이점을 활용할 수 있다. TCP 대신 **UDP 기반의 통신은 여러 목적지로 동시에 데이터를 보낼 수 있다.** 

ROS2에서는 `ROS_DOMAIN_ID`라는 환경 변수로 도메인을 설정하게 된다. ROS2에서는 전역 공간이라 불리는 DDS Global Space라는 공간에 있는 토픽들에 대해 구독 및 발행을 할 수 있다.

&nbsp;

- 데이터 중심적 기능

DDS를 사용하면 제일 많이 보는 말 중 하나는 `Data Centric`이다. DDS에서도 DCPS(Data Centric Publish Subscribe)라는 개념이 있는데, 이는 **적절한 수신자에게 적절한 정보를 효율적으로 전달하는 것을 목표로 하는 발간 및 구독 방식이다.** 사용자 입장에서 **어떤 데이터인지, 어떤 형식인지, 어떻게 보낼지, 안전하게 보낼지에 대한 기능이 존재**한다는 것이다.

<img src="/assets/img/ros2/dcps.png">

&nbsp;

- 동적 검색

DDS는 동적 검색(Dynamic Discovery)를 제공한다. 응용 프로그램은 DDS의 동적 검색을 통하여 어떤 토픽이 지정 도메인 영역에 있으며 어떤 노드가 이를 발신하고 수신하는지 알 수 있게 된다. 이는 ROS 프로그래밍 시 데이터를 주고 받을 노드들의 IP주소 및 포트를 미리 입력하거나 따로 구성하지 않아도 되며, 사용하는 시스템 아키텍처의 차이점을 고려할 필요가 없기 때문에, 모든 운영 체제 또는 하드웨어 플랫폼에서 매우 쉽게 작업할 수 있다.

ROS1에서는 마스터 노드에 노드의 이름 및 메시지를 찾아서 연결시켜준다. ROS2에서는 마스터 노드가 없기에 노드를 DDS의 Participant  개념으로 취급하게 되었으며, **동적 검색 기능을 이용하여 DDS 미들웨어를 통해 직접 검색하고 노드를 연결할 수 있게 된다.**

&nbsp;

- 확장 가능한 아키텍처

**DDS를 사용함으로써 IoT, 항공 우주 산업과 같은 초대형 시스템에 확장이 가능**해졌다. 단일 표준 통신 계층에서 복잡성을 흡수함으로서 분산 시스템 개발을 더욱 단순화시켰다. 수백 수천개의 노드를 관리하거나 여러 대의 로봇, 주변 인프라를 통제하는 IT 기술 등에 적용시킬 수 있게 되었다.

&nbsp;

- 상호 운용성

DDS를 사용하고 있는 제품을 사용하면, A라는 회사의 제품을 사용하다가 B라는 회사 제품으로 변경이 가능하다. ROS2를 지원하는 업체는 ADLink, Eclipse, Eprosima, Gurum Network, RTI로 총 5곳이 있다. 

&nbsp;

- 서비스 품질(QoS)

노드 간의 DDS 통신 옵션을 설정하는 QoS는 퍼블리셔 및 서브스크라이버 등을 선언하고 사용할 때 매개변수처럼 QoS를 사용한다. 

```python
self.helloworld_publisher = self.create_publisher(String, 'helloworld', qos_profile)
```

DDS 사양상 설정 가능한 QoS 항목은 22가지인데, ROS2에서는 데이터 손실을 방지하여 **신뢰도**를 우선시(*reliable*) 할 수 있으며, UDP처럼 통신 속도를 최우선시하여 사용(best effort)할 수 있게 하는 **신뢰성**(*reliability*) 기능이 대표적으로 사용된다. 

그 외에는 다음과 같다.
- *history* : 통신 상태에 따라 정해진 사이즈만큼의 데이터를 **보관** 
- *Durability* : 데이터를 수신하는 서브스크라이버가 생성되기 전의 데이터를 **사용할지 폐기할지**에 대한 설정
- *Deadline* : 정해진 주기 안에 데이터가 발신 및 수신되지 않을 경우 **이벤트 함수를 실행**시키는 기능
- *Lifespan* : 정해진 주기 안에서 수신되는 데이터만 **유효 판정**하고 그렇지 않은 데이터는 삭제하는 기능
- *Liveliness* : 정해진 주기 안에서 노드 혹은 **토픽의 생사**를 확인

이 외에도 엄청 많다. 이러한 QoS 설정을 통해 DDS는 적시성, 트래픽 우선순위, 안정성 및 리소스 사용과 같은 데이터를 주고 받는 모든 측면을 사용자가 제어할 수 있게 되었다. 

&nbsp;

- 보안

DDS를 사용함으로서 보안 성능을 향상시켰다. DDS-Security라는 DDS 보안 사양을 ROS에 적용하여 보안에 대한 이슈를 통신단부터 해결했다. 또한, SROS2(Secure Robot Operating System 2)라는 툴을 개발하여 보안 관련 RCL 서포트 및 보안 관련 프로그래밍을 위한 툴킷을 만들어 배포했다.

&nbsp;

&nbsp;

## RMW 변경 방법

기본적으로 실행하게 되면 기본 RMW인 `rmw_fastrtps_cpp`가 사용된다. 만약 RMW를 변경하여 사용하려면 위에 개발 환경에서 hash(#)처리했던 설정들로 변경하여 사용하면 된다.

```markdown
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# export RMW_IMPLEMENTATION=rmw_connext_cpp
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# export RMW_IMPLEMENTATION=rmw_gurumdds_cpp
# export RMW_IMPLEMENTATION=rmw_opensplice_cpp
```

이를 변경하고 나서 노드를 실행하면 된다.

| rclcpp_tutorial은 [0025번 목차](https://cafe.naver.com/openrt/24451)를 확인하길 바란다. 단지 아래 내용을 확인하려면 별다른 설정없이 `ros2 run demo_nodes_cpp talker`으로 실행하면 된다.

```bash
$ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
$ ros2 run rclcpp_tutorial helloworld_publisher
[INFO]: Published message: Hello World: 0
[INFO]: Published message: Hello World: 1
[INFO]: Published message: Hello World: 2
[INFO]: Published message: Hello World: 3
[INFO]: Published message: Hello World: 4
...
```

만약 RMW_IMPLEMENTATION 설정을 각각 테스트하고자 한다면, 각 노드 별로 서로 다르게 RMW_IMPLEMENTATION 환경 변수로 선언할 수 있다. 

```bash
$ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
$ ros2 run rclcpp_tutorial helloworld_publisher
[INFO]: Published message: Hello World: 0
[INFO]: Published message: Hello World: 1
[INFO]: Published message: Hello World: 2
[INFO]: Published message: Hello World: 3
[INFO]: Published message: Hello World: 4
...
```

```bash
$ export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
$ ros2 run rclcpp_tutorial helloworld_subscriber
[INFO]: Received message: Hello World: 0
[INFO]: Received message: Hello World: 1
[INFO]: Received message: Hello World: 2
[INFO]: Received message: Hello World: 3
[INFO]: Received message: Hello World: 4
...
```

이렇게 실행해서 통신이 되면, 문제 없이 실행된 것이다.

&nbsp;

## Domain 변경 방법

UDP 멀티캐스트로 통신이 이루어지기 때문에, 별도의 설정을 하지 않으면 같은 네트워크의 모든 노드가 연결된다. 예를 들어 같은 연구실에서 동일 네트워크를 사용한다면 다른 연구원들의 노드의 데이터에 접근 가능하게 된다. 

그래서 다른 네트워크를 이용하도록 설정하거나 name space를 변경하여 사용하면 된다. 가장 간단한 방법으로는 DDS의 domain을 변경하는 것이다. 이는 `ROS_DOMAIN_ID` 라는 환경변수를 변경한다.

```bash
$ export ROS_DOMAIN_ID=11
$ ros2 run rclcpp_tutorial helloworld_publisher
[INFO]: Published message: Hello World: 0
[INFO]: Published message: Hello World: 1
[INFO]: Published message: Hello World: 2
[INFO]: Published message: Hello World: 3
[INFO]: Published message: Hello World: 4
```

```bash
$ export ROS_DOMAIN_ID=12
$ ros2 run rclcpp_tutorial helloworld_subscriber


$ export ROS_DOMAIN_ID=11
$ ros2 run rclcpp_tutorial helloworld_subscriber
[INFO]: Received message: Hello World: 89
[INFO]: Received message: Hello World: 90
...
```

도메인을 다르게 설정했기 때문에, 메시지를 받지 못하다가 같은 ID로 설정하니 구독이 진행되는 것을 확인할 수 있다. DOMAIN_ID는 대체로 0~232까지의 정수를 사용한다.

&nbsp;

&nbsp;

## QoS 테스트

위에서 QoS의 특정 중 reliable, reliability 기능을 설명했다. 이에 대해 간단한 테스트를 진행해보고자 한다.

이 테스트에서는 tc(traffic control)라는 리눅스 네트워크 트래픽 제어 유틸리티를 사용하여 임의의 데이터 손실(10%)를 만들어 reliability성을 테스트했다.

&nbsp;

기본적으로 Reliablility의 기본 설정은 RLIABLE 로 되어 있다. 그렇기에 데이터 손실이 있어도 TCP와 같이 ack로 매번 확인하여 손실된 데이터를 재전송하므로 잃어버리는 데이터가 없다. 그러나 손실이 존재할 때, 터미널 창을 잠시 멈추는 것을 확인할 수 있는데, 이 때 손실된 데이터를 순차적으로 재전송하고 다시 ack 작업을 하기 때문이다.

- Reliability가 BEST_EFFORT로 되어 있을 때

```bash
$ sudo tc qdisc add dev lo root netem loss 10%
```

tc의 데이터 손실을 10%로 설정한 후 동일하게 실행해보면, 1부터 순차적으로 데이터를 전송했지만, 서브스크라이버 노드에 데이터를 손실한 채 표시된다. 

&nbsp;

데이터 손실이 있어도 문제없는 데이터라면 Reliability를 **BEST_EFFORT**로 설정하면 속도가 더 빨라질 것이다.

&nbsp;

마지막으로 tc를 다시 되돌려놓는다.

```bash
sudo tc qdisc delete dev lo root netem loss 10%
```