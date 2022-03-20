# ROS 기초

# 1강

## ros의 기능 소개

[https://www.ros.org](https://www.ros.org)

ros : robot operating system

- 오픈소스 로봇 운영체제
    - 소스 무료 공개
    - 개방형 구조
- 로봇 소프트웨어를 개발하는 데 필요한 소프트웨어의 집합체
    - 소프트웨어 프레임워크
- 메타 운영체제 (meta os), 미들웨어 (middleware)
    - 소프트웨어 모듈 + 라이브러리 집합 + 도구 집합
    

ROS는 자동차 제어를 위한 미들웨어

- 각종 센서와 모터를 프로그래머가 편하게 사용할 수 있도록 지원

<img src="/assets/img/dev/week3/day1/ros.png">

ros는 실제 차에는 적용하기 힘드나 자이카와 같은 차에는 적용하기 적합하다.

특징

- 로봇 sw를 만들기 위한 코드의 재사용이 용이한 환경제공이 목표
    - 다양한 프로그래밍 언어를 지원 : C++, python
    - 표준화된 ros인터페이스를 따르는 hw 와 sw를 편하게 엮을 수 있음
    - 하드웨어 부품과 소프트웨어 부품을 조립하여 여러 응용 구성 가능
    - 대규모 실행 시스템 및 프로세스에도 적용 가능
- 다양한 도구들을 함께 제공
    - RVIZ, RQT, Gazebo
- 다양한 os 환경에서 통일된 방법으로 상호작용을 구현하는 것이 가능
    - linux, os x, windows, android, ios
    - 표준화된 통신 프로트콜을 따르는 이기종간의 메시지 교환이 가능
    

도구

- RVIZ
    - 시각화 도구
    - 센서데이터를 비롯한 주변환경 변화를 시각화
        
        <img src="/assets/img/dev/week3/day1/rviz.png">
        
- RQT
    - Qt 기반의 GUI 응용 개발 도구
    - 노드 연결 정보를 그래프로 표현
    - 사용자 상호작용을 UI를 갖춘 응용 개발에 이용
    - rqt_graph
        - 노드와 토픽의 관계 정보를 그래프로 출력
            
            ```bash
            $ rqt_graph
            ```
            
            <img src="/assets/img/dev/week3/day1/rqt.png">
            
- GAZEBO
    - 물리 엔진 기반의 3차원 시뮬레이터
    - 시뮬레이터 제작 및 모델링에 이용

ros버전

- ros indigo
- ros **kinetic**
- ros **melodic**

ros에서의 통신

노드(node) → 토픽(topic) → 노드(node)

프로세스 → 메시지 → 프로세스

---

# ros 핵심 기능

노드간 통신을 기반으로 전체 시스템을 구동시킨다

- 노드 : 하드웨어 부품 또는 소프트웨어 모듈에 노드가 하나씩 할당된다.

<img src="/assets/img/dev/week3/day1/master.png">

- 노드는 os의 도움을 받아 하드웨어 장치들을 제어한다.
- 노드들은 마스터의 도움을 받아 서로 메시지를 주고 받는다.
- 마스터는 노드간의 만남을 주선해주는 주선책과 같은 역할을 한다.

<img src="/assets/img/dev/week3/day1/master2.png">

서로 분리된 하드웨어 장치 안에 있는 노드들이 네트워크 연결을 통해 서로 통신하면서 하나의 단일 시스템으로서 동작하는 것이 가능하다.

---

# ros 구현  사례

- 라이다 + 카메라 + 모터 + sw모듈1 + sw모듈2
- 이는 하드웨어 장치 3개와 소프트웨어 모듈 2개를 함께 엮어서 원하는 기능을 구현
- 라이다와 카메라 정보를 상황인지 sw가 분석한 후 결과를 운전판단 sw로 보내고, 운전판단 sw가 제어명령을 생성해서 모터로 보내 차량을 움직이게 함
- 급제동 sw노드를 한 개 더 만들어서 라이다를 통해 거리가 너무 가까워지면 또는 급박한 상황일 때 빠르게 모터를 제어한다.
    
<img src="/assets/img/dev/week3/day1/fullstream.png">
    


---

# 기본 용어

### 마스터 master

- 서로 다른 노드들 사이의 통신을 총괄 관리
- 통상 ROS Core이라 부른다.

### 노드 node

- 실행가능한 최소의 단위, 프로세스로 이해할 수 있다.
- ROS에서 발생하는 통신(메시지 송/수신)의 주체
- HW장치에 대해 하나씩의 노드, SW모듈에 대해 하나씩의 노드 할당

### 토픽 Topics

- 노드와 노드가 주고받는 데이터
- 토픽 안에 들어 있는 실제 데이터를 메시지라 부른다.
- 예: 센서데이터, 카메라 이미지 ,액츄에이터 제어명령...

### 발행자 publisher

- 특정 토픽에 메시지를 담아 외부로 송신하는 노드
- 예 : 센서, 카메라, 모터제어 알고리즘...

### 구독자 subscribers

- 특정 토픽에 담겨진 메시지를 수신하는 노드
- 예 : 액츄에이터 제어기, 데이터 시각화 도구 ...
    
<img src="/assets/img/dev/week3/day1/turtlerqt.png">
    

### 패키지 packages

- 하나 이상의 노드와 노드의 실행을 위한 정보 등을 묶어 놓은 단위
- 노드, 라이브러리, 데이터, 파라미터 등을 포함

---

## ROS 응용 예

<img src="/assets\img\dev\week3\day1\sensor.png" caption = "sensor">

각각 어느 토픽에 어떤 형태의 메시지를 발행하는지가 정해져 있다.

<img src="/assets\img\dev\week3\day1\actuator.png" caption = "actuator">


어느 토픽에 어떤 메시지를 발행하면 어떻게 동작하는지가 정해져 있다.

제어 알고리즘 노드는 파이썬을 짤 것이다. ROS책을 사도 적혀있다. 

---

## ROS 노드간 통신 기본 과정

<img src="/assets\img\dev\week3\day1\master.png">

- 통신이 이루어지기 이전에 통신을 원하는 노드는 마스터에 의뢰하여 연결해야 하는 노드의 정보(주소)를 얻어오고, 서로 접속정보를 교환

<img src="/assets\img\dev\week3\day1\servernetwork.png">

- 통신환경 구축이 완료되고 나면 노드간 통신은 마스터를 거치지 않고 직접 이루어진다.

---

## ros노드간 통신의 두 가지 방식

- 토픽 방식의 통신
    - 일방적이고 지속적인 메시지 전송
    - 1:1 뿐만 아니라 1:N , N:N 통신도 가능
    - 자율주행시 이 방식을 사용함
    
    <img src="/assets\img\dev\week3\day1\topic.png">
    
- 서비스 방식의 통신
    - 서버가 제공하는 서비스에 클라이언트가 요청을 보내고 응답을 받는 방식
    - 양방향 통신, 일회성 메시지 송수신
    
    <img src="/assets\img\dev\week3\day1\service.png">
    

---

## 시나리오

1. 마스터 (roscore) 시동:
    - 통신이 이루어지려면 우선 roscore가 실행되고 있어야 한다.
    
<img src="/assets\img\dev\week3\day1\node1.png">
    
2. 구독자(subscriber) 노드 구동:
    - 특정 토픽(topic)에 발행되는 메시지를 수신하기를 요청
    
<img src="/assets\img\dev\week3\day1\node2.png">
    

3. 발행자(publisher)  노드 구동:
    - 특정 토픽(topic) 메시지를 발행하겠다는 의사를 전달
        
        이름 - 토픽 이름 - 숫자(토픽 이름의 숫자를 원하는 사람이 있으면 줘라) - 주소
        
<img src="/assets\img\dev\week3\day1\node3.png">
        

4. 노드 정보 전달:
    - 마스터가 발행자 정보를 구독자에게 전달
        
<img src="/assets\img\dev\week3\day1\node4.png">
        

5. 노드간 접속 요청:
    - 구독자 노드가 발행자 노드에 TCPROS 접속을 요청
        
<img src="/assets\img\dev\week3\day1\node5.png">
        

6. 노드간 접속 요청
    - 발행자 노드가 자신의 TCPROS URI(포트 포함)를 전송하여 응답
    
<img src="/assets\img\dev\week3\day1\node6.png">
    

7. TCPROS 접속:
    - 발행자 노드와 구독자 노드 사이에 소켓(socket) 연결이 이루어짐
    
<img src="/assets\img\dev\week3\day1\node7.png">
    

8. 메시지 전송:
    - 발행자 노드가 구독자 노드에게 메시지 전송(토픽)
    
<img src="/assets\img\dev\week3\day1\node8.png">
    

9. 메시지 전송 반복:
    - 접속이 한번 이루어진 뒤에는 별도 절차없이 지속적으로 메시지 송수신
    
<img src="/assets\img\dev\week3\day1\node9.png">
    

---

## 명령어

1. ros 셀 명령어
    - roscd: 지정한 ros패키지 폴더로 이동
    - rosls: ros 패키지 파일 목록 확인
    - rosed: ros 패키지 파일 편집
    - roscp: ros 패키지 파일 복사
2. **ros 실행 명령어**
    - **roscore:** **master**+rosout+parameter server
    - **rosrun: 패키지 노드 실행**
    - **roslaunch: 패키지 노드를 여러 개 실행**
    - rosclean: ros 로그 파일 검사 및 삭제
3. ros 정보 명령어
    - rostopic: 토픽 정보 확인
    - rosnode: 노드 정보 확인
    - rosbag: 메시지 기록, 재생
4. ros catkin 명령어
    - catkin_create_pkg: catkin 빌드 시스템으로 패키지 생성
    - catkin_make: catkin 빌드 시스템으로 빌드
5. ros package 명령어
    - rospack: 패키지와 관련된 정보 보기
    - rosinstall: 추가 패키지 설치
    - rosdep: 해당 패키지의 의존성 파일 설치

주요 명령어

```bash
ros 기본 시스템이 구동되기 위해 필요한 프로그램들 실행
$ roscore

rosrun [package name] [node_name]: 패키지에 있는 노드를 선택 실행
$ rosrun turtlesim turtlesim_node

rosnode [info...]: 노드의 정보를 표시
$ rosnode info node_name

rostopic [option]: 토픽의 정보를 출력 (메시지 타입, 노드 이름 등)
$ rostopic info /imu 

roslaunch [package_name] [file.launch]: 파라미터 값과 함께 노드를 실행
$ roslaunch usb_cam usb_cam-test.launch
```

---

---

# 2강

## ros 설치

- ubuntu 18.04가 설치된 노트북이 필요

1. ros를 제공하는 software repository 등록
    
    ```bash
    $ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    $ cat /etc/apt/sources.list.d/ros-latest.list
    ```
    
<img src="/assets\img\dev\week3\day1\repo.png">
    
2. apt key 셋업
    
    ```bash
    $ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
    ```
    
<img src="/assets\img\dev\week3\day1\aptkey.png">
    
3. 패키지 설치
    
    ```bash
    $ sudo apt update
    $ sudo apt install ros-melodic-desktop-full // 다양한 것들을 다 설치
    ```
    
4. 쉘 환경 설정
    
    ```bash
    $ echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
    $ source ~/.bashrc
    ```
    
5. 추가로 필요한 도구 설치
    
    ```bash
    $ sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
    ```
    
6. rosdep 초기화
    
    ```bash
    $ sudo rosdep init
    $ sudo rosdep update
    ```
    

[http://wiki.ros.org/melodic/Installation/Ubuntu](http://wiki.ros.org/melodic/Installation/Ubuntu)

---

## roscore 실습

1. roscore (나중에 <ctrl+c>를 누르면 종료
    
    ```bash
    $ roscore
    ```
    
2. (다른 터미널에서) resnode list
    
    ```bash
    $ rosnode list
    ```
    
    - 오류가 난다면
        
        ```bash
        $ apt update
        
        업그레이드할 것이 있다면
        $ apt upgrade
        ```
        

workspace 생성

```bash
$ mkdir -p ~/xycar_ws/src
$ cd xycar_ws
$ catkin_make
```

이 작업 공간이 ros 프로그래밍 공간으로 사용

`/src`: 소스코드는 여기에 넣음, CMakeList.txt

### catkin_make

- ros의 workspace에서 새로운 소스코드 파일이나 패키지가 만들어지면 catkin_make를 명령한다.
    - 이 명령을 통해 빌드(build)작업을 진행한다.
    - ROS 프로그래밍 작업과 관련 있는 모든 것들을 깔끔하게 정리해서 최신 상태로 만드는 작업
    - 환경 정리? 와 같음

## ros 작업환경 설정

ros 작업에 필요한 환경변수 설정

- 홈 디렉토리에 있는 .bashrc 파일을 수정

```bash
$ sudo gedit ~/.bashrc
```

아래 내용 작성

> alias cm='cd ~/xycar_ws && catkin_make' # 자주 쓰는 명령어를 정의해놓는 것
alias cs=’cd ~/xycar_ws/src’
source /opt/ros/melodic/setup.bash
source ~/xycar_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
> 

그다음 수정한 내용을 시스템에 반영하기 위해

```bash
$ source ~/.bashrc
$ printenv | grep ROS
ROS_ETC_DIR=/opt/ros/melodic/etc/ros
ROS_ROOT=/opt/ros/melodic/share/ros
...
```

---

## ROS예제 프로그램 구동 실습

<!-- <img src="/assets\img\dev\week3\day1\untitled.png"> -->

publisher

- 이름: teleop_turtle
- 보내는 토픽: turtle1/cmd_vel
- 담고 있는 메시지: geometry_msgs/Twist

subscriber

- 이름: turtlesim
- 받고자하는 토픽: turtle1/cmd_vel
- 토픽이 담고 있어야 하는 메시지: geometry_msgs/Twist

### 실행

총 4개의 터미널을 열어야 한다.

1. 마스터 : $ roscore
2. ros node의 확인 : $ rosnode list
3. rosrun turtlesim turtlesim_node
4. roscore와 turtlesim_node가 실행중인 상태일 때 : $ rosrun turtlesim turtle_teleop_key

4터미널에서 키보드 방향키를 누르면 움직임

이 때 rosnode list를 확인하면 3개가 보일 것이다.

```bash
$ rosnode list
/rosout
/teleop_turtle
/turtlesim
```

<img src="/assets\img\dev\week3\day1\rosnodelist.png">

또는 rqt_graph를 통해 시각화 할 수 있다.

```bash
$ rqt_graph
```

<img src="/assets\img\dev\week3\day1\turtlerqt.png">

어떤 토픽이 이동되는지 보려면 , 또는 그 토픽의 메시지를 보려면

```bash
$ rostopic list
/rosout
/rosout_agg
/turtle1/cmd_vel
/turtle1/color_sensor
/turtle1/pose

$ rostopic echo /turtle1/cmd_vel
linear: 
  x: 0.0
  y: 0.0
  z: 0.0
angular: 
  x: 0.0
  y: 0.0
  z: 2.0
---
linear: 
  x: 0.0
  y: 0.0
  z: 0.0
angular: 
  x: 0.0
  y: 0.0
  z: -2.0
---
```

![Untitled](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2019.png)

조금 더 자세히 보기 위해서는

토픽 - 메시지 내용 - 보내거나 받는 노드개수 - pub/sub

```bash
$ rostopic list -v

Published topics:
 * /turtle1/color_sensor [turtlesim/Color] 1 publisher
 * **/turtle1/cmd_vel [geometry_msgs/Twist] 1 publisher**
 * /rosout [rosgraph_msgs/Log] 2 publishers
 * /rosout_agg [rosgraph_msgs/Log] 1 publisher
 * /turtle1/pose [turtlesim/Pose] 1 publisher

Subscribed topics:
 * **/turtle1/cmd_vel [geometry_msgs/Twist] 1 subscriber**
 * /rosout [rosgraph_msgs/Log] 1 subscriber
```

메시지의 타입과 구성을 볼 수도 있다.

```bash
$ rostopic type /turtle1/cmd_vel
geometry_msgs/Twist

$ rosmsg show geometry_msgs/Twist
geometry_msgs/Vector3 linear
  float64 x
  float64 y
  float64 z
geometry_msgs/Vector3 angular
  float64 x
  float64 y
  float64 z
```

### 토픽 직접 발행하기

$ rostopic pub [발행 횟수] [토픽 이름] [메시지 타입] — ‘메시지 내용’

```bash
1번만 보내기
$ rostopic pub -1 /turtle1/cmd_vel geometry_msgs/Twist -- '[2.0,0.0,0.0]' '[0.0,0.0,1.8]'
publishing and latching message for 3.0 seconds
```

![실행 전](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2020.png)

실행 전

![실행 후](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2021.png)

실행 후

$ rostopic pub [토픽 이름] [메시지 타입] [-r #Hz] — ‘메시지 내용’

```bash
1초에 1번씩 보내기
$ rostopic pub /turtle1/cmd_vel geometry_msgs/Twist -r 1 -- '[2.0,0.0,0.0]' '[0.0,0.0,1.8]'
```

<ctrl+C> 누르면 종료된다.

---

---

# 3강

## ROS package

패키지 : ROS에서 개발되는 소프트웨어를 논리적 묶음으로 만든 것

![Untitled](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2022.png)

보행자 추적 패키지, 차선인식 패키지 등..

## ros가 제공하는 편리한 명령들

- rospack list : 어떤 패키지들이 있는지 나열
- rospack find [ package_name ] : 이름을 이용해서 패키지 검색
- roscd [location_name[/subdir] : ROS 패키지 디렉토리로 이동
- rosls [location_name[/subdir] : linux ls와 유사 (경로를 몰라도 이름 적용 가능)
- rosed [file_name] : 에디터로 파일을 편집

## ros 패키지 만들기

코드를 작성하는 명령어로는 gedit을 사용하면 된다.

```bash
$ gedit pub.py
```

1. 패키지 담을 디렉토리로 이동
    
    ```bash
    $ cd ~/xycar_ws/src
    ```
    
2. 패키지 새로 만들기
    
    catkin_create_pkg [패키지 이름] [이 패키지가 의존하고 있는 다른 패키지들 나열]
    
    ```bash
    $ catkin_create_pkg my_pkg1 std_msgs rospy
    ```
    
3. 빌드(갱신)
    
    ```bash
    $ catkin_make
    ```
    
4. 만들어진 패키지 확인
    
    ```bash
    $ rospack find my_pkg1
    /home/jaehoyoon/xycar_ws/src/my_pkg1
    
    의존하고 있는 패키지 출력
    $ rospack depends1 my_pkg1
    rospy
    std_msgs
    
    이동
    $ roscd my_pkg1
    ~/xycar_ws/src/my_pkg1$
    ```
    
5. 소스코드를 추가한 후 그것을 실행하려고 했지만 파일 타입이 실행 권한이 없는 것이다. 그래서 타입을 바꿔줘야 한다.
    
    ```bash
    $ ls -l
    합계 8
    -rw-rw-r-- 1 jaehoyoon jaehoyoon 397  2월 28 20:52 pub.py
    -rw-rw-r-- 1 jaehoyoon jaehoyoon 285  2월 28 20:54 sub.py
    
    $ chmod +x pub.py sub.py
    
    $ ls -l
    합계 8
    -rwxrwxr-x 1 jaehoyoon jaehoyoon 397  2월 28 20:52 pub.py
    -rwxrwxr-x 1 jaehoyoon jaehoyoon 285  2월 28 20:54 sub.py
    ```
    

---

## 패키지 실행

### publisher

터미널 4개로 해서

1. roscore
2. [pub.py](http://pub.py) 실행  : rosrun my_pkg1 pub.py (my_pkg1안에 있는 pub.py를 실행)
3. turtlesim 노드 실행 : rosrun turtlesim turtlesim_node
4. rqt_graph 실행

```bash
$ roscore

---
$ rosrun turtlesim turtlesim_node

---
$ rosrun my_pkg1 pub.py
```

노드 상태 확인

```bash
$ rqt_graph
```

![Untitled](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2023.png)

```bash
$ rosnode list
/my_node_59758_1646054585073
/rosout
/turtlesim
```

### subscriber

우선, turtle이 어떤 토픽에 어떤 메시지를 발행하고 있는지 알아봐야 한다.

```bash
이동되고 있는 전체 토픽
$ rostopic list
/rosout
/rosout_agg
/turtle1/cmd_vel
/turtle1/color_sensor
/turtle1/pose

어떤 타입이고 어떤 값을 가지는지 보기
$ rostopic type /turtle1/pose
turtlesim/Pose

어떤 메시지를 가지는가
$ rosmsg show turtlesim/Pose
float32 x
float32 y
float32 theta
float32 linear_velocity
float32 angular_velocity

토픽에 어떤 메시지가 발행되고 있는지 실제 값 보기
$ rostopic echo /turtle1/pose
x: 4.42990398407
y: 6.82239103317
theta: -1.73598301411
linear_velocity: 2.0
angular_velocity: 1.79999995232
---
x: 4.425552845
y: 6.79068851471
theta: -1.70718300343
linear_velocity: 2.0
angular_velocity: 1.79999995232
---
x: 4.42211675644
y: 6.75887346268
theta: -1.67838299274
linear_velocity: 2.0
angular_velocity: 1.79999995232
---
...
```

터미널 4개

1. roscore
2. turtlesim 노드 실행
3. pub.py
4. sub.py

```bash
$ roscore

---
$ rosrun turtlesim turtlesim_node

---
$ rosrun my_pkg1 pub.py

---
$ rosrun my_pkg1 sub.py
[INFO] [1646055414.320049]: /my_listener_59983_1646055411189Location: 4.47,6.996734
[INFO] [1646055414.336577]: /my_listener_59983_1646055411189Location: 4.46,6.966142
[INFO] [1646055414.351566]: /my_listener_59983_1646055411189Location: 4.45,6.935292
[INFO] [1646055414.370935]: /my_listener_59983_1646055411189Location: 4.45,6.904211
[INFO] [1646055414.383501]: /my_listener_59983_1646055411189Location: 4.44,6.872923
[INFO] [1646055414.400245]: /my_listener_59983_1646055411189Location: 4.43,6.841455
[INFO] [1646055414.415747]: /my_listener_59983_1646055411189Location: 4.43,6.809832
[INFO] [1646055414.431543]: /my_listener_59983_1646055411189Location: 4.42,6.778082
[INFO] [1646055414.448686]: /my_listener_59983_1646055411189Location: 4.42,6.746230
[INFO] [1646055414.463205]: /my_listener_59983_1646055411189Location: 4.42,6.714302
[INFO] [1646055414.480053]: /my_listener_59983_1646055411189Location: 4.42,6.682326
...
```

```bash
$ rqt_graph
```

![Untitled](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2024.png)

![rostopic echo를 하나 더 실행할 때](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2025.png)

rostopic echo를 하나 더 실행할 때

---

---

# 4강

roslaunch

*.launch 파일 내용에 따라 여러 노드들을 한꺼번에 실행시킬 수 있음, 파라미터 값을 노드에 전달할 수도 있다.

![Untitled](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2026.png)

roslaunch [패키지이름] [실행시킬 launch 파일 이름]

- roslaunch my_pkg1 aaa.launch

이 때 실행시킬 launch 파일은 반드시 패키지에 포함된 launch파일이어야 한다.

### *.launch 파일

실행시킬 노드들의 정보가 XML 형식으로 기록되어 있다. launch 파일은 my_pkg1/launch/*.launch 경로로 생성해야 한다. 

![Untitled](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2027.png)

- node 태그: 실행할 노드 정보를 입력할 때 사용되는 태그
    - <node pkg=”패키지 명” type=”노드가 포함된 소스파일 명” name=”노드 이름” />
    - 속성
        - pkg: 실행시킬 노드의 패키지 이름을 입력하는 속성 → 반드시 빌드된 패키지의 이름을 입력해야 함
        - type: 노드의 소스코드가 담긴 파이썬 파일의 이름을 입력하는 속성 → 이때 파이썬 py파일은 반드시 실행권한이 있어야 함
        
        > 실행권한이 없으면 ERROR: cannot launch node of type ~ 이라 나옴
        > 
        - name: 노드의 이름을 입력하는 속성 → 소스코드에서 지정된 노드의 이름을 무시하고 launch파일에 기록된 노드의 이름으로 노드가 실행된다.

- include 태그: 다른 launch 파일을 불러오고 싶을 때 사용하는 태그
    - <include file=”같이 실행할 *.launch 파일 경로” />
    - 이 때 <include file=”$(find usb_cam)/src/launch/aaa.launch” /> 라고 칠 수 있는데, $()는 함수를 실행하는 것으로 find usb_cam을 실행한 결과를 가져오는 것
    - 속성
        - file: 함께 실행시킬 *.launch 파일의 경로를 입력하는 속성

### launch 파일 실행

```bash
launch 파일 생성
my_pkg1$ mkdir launch

launch 파일 작성
$ cd launch
$ gedit pub-sub.launch

```

```xml
<launch>
    <node pkg="turtlesim" type="turtlesim_node" name="turtlesim_node" />
    <node pkg="my_pkg1" type="[pub.py](http://pub.py/)" name="pub_node" />
    <node pkg="my_pkg1" type="[sub.py](http://sub.py/)" name="sub_node" output="screen" />
</launch>
```

를 작성해준다.

```bash
launch 파일 실행
$ roslaunch my_pkg1 pub-sub.launch
```

<aside>
💡 **여기서 중요한 것은 roslaunch를 할 때는 roscore를 할 필요가 없다. 스스로 roscore가 작동되기 때문이다.**

</aside>

```bash
$ roslaunch my_pkg1 pub-sub.launch 
... logging to /home/jaehoyoon/.ros/log/2089d226-989f-11ec-8c28-000c2983f177/roslaunch-jaeho-vm-60582.log
Checking log directory for disk usage. This may take a while.
Press Ctrl-C to interrupt
Done checking log file disk usage. Usage is <1GB.

started roslaunch server http://localhost:44745/

SUMMARY
========

PARAMETERS
 * /rosdistro: melodic
 * /rosversion: 1.14.12

NODES
  /
    pub_node (my_pkg1/pub.py)
    sub_node (my_pkg1/sub.py)
    turtlesim_node (turtlesim/turtlesim_node)

auto-starting new master
process[master]: started with pid [60592]
ROS_MASTER_URI=http://localhost:11311

setting /run_id to 2089d226-989f-11ec-8c28-000c2983f177
process[rosout-1]: started with pid [60603]
started core service [/rosout]
process[turtlesim_node-2]: started with pid [60606]
process[pub_node-3]: started with pid [60611]
process[sub_node-4]: started with pid [60612]
[INFO] [1646056978.076219]: /sub_nodeLocation: 5.54,5.544445
[INFO] [1646056978.092649]: /sub_nodeLocation: 5.54,5.544445
[INFO] [1646056978.107839]: /sub_nodeLocation: 5.54,5.544445
[INFO] [1646056978.123153]: /sub_nodeLocation: 5.54,5.544445
[INFO] [1646056978.140403]: /sub_nodeLocation: 5.54,5.544445
[INFO] [1646056978.156125]: /sub_nodeLocation: 5.54,5.544445
[INFO] [1646056978.172139]: /sub_nodeLocation: 5.54,5.544445
[INFO] [1646056978.187652]: /sub_nodeLocation: 5.54,5.544445
...
```

![Untitled](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2028.png)

---

## launch 파일의 유용한 tag

1. USB 카메라 구동과 파라미터 세팅을 위한 launch 파일

패키지 이름: usb_cam

파라미터 5개

```xml
<launch>
	<node name="usb_cam" pkg="turtlesim" type="cam_node" output="screen">
		<param name="autoexposure" value="false"/>
		<param name="exposure" value="180"/>
		<param name="image_width" value="640"/>
		<param name="image_height" value="480"/>
    <param name="camera_frame_id" value="usb_cam"/>
	</node>
</launch>
```

type=”cam_node”에 py가 없는 걸로 보아 c나 c++로 되어 있을 것이다. 

autoexposure이라는 파라미터의 값을 false, exposure 파라미터를 180, 이미지 크기를 [640,480]으로 정하는 것이다.

즉, launch를 통해 다섯가지 파라미터의 값을 셋팅할 수 있다.

### param

ROS 파라미터 서버에 변수를 등록하고 그 변수에 값을 설정하기 위한 태그다.

<param name=”변수 이름” type=”변수 타입” value=”변수 값” />

속성

- name: 등록할 변수의 이름
- type(선택): 등록할 변수의 타입, 사용할 수 있는 타입의 종류는 str, int, double, bool, yaml
- value: 등록할 변수의 값

ROS파라미터 서버에 등록된 변수는 노드 코드(소스코드)에서 불러와 사용할 수 있다.

*.launch ⇒ <param name=”age” type=”int” value=”11” />

*.py ⇒ import rospy; rospy.init_node(’노드’); print(rospy.**get_param**(’~age’))

~은 private parameter은 앞에 ~를 붙여야 한다.

실습

1. launch 파일을 새로 만들자 

```bash
$ cp pub-sub.launch pub-sub-param.launch
```

패키지 이름 = my_pkg1

타입 (소스코드 파일) = pub_param.py

노드 이름 = node_param

파라미터 이름 = circle_size

파라미터 값 = 2

```xml
# pub-sub-param.launch
<launch>
	<node pkg="turtlesim" type="turtlesim_node" name="turtlesim_node" />
	<node pkg="my_pkg1" type="pub_param.py" name='node_param'>
		<param name="circle_size" value="2" />
	</node>
	<node pkg="my_pkg1" type="sub.py" name="sub_node" output="screen" />
</launch>
```

1. pub_param.py 도 만들기

```bash
$ cp pub.py pub_param.py
```

```python
#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

rospy.init_node('my_node', anonymous=True)
pub = rospy.Publisher('/turtle1/cmd_vel',Twist,queue_size=10)

msg = Twist()

# msg.linear.x = 2.0
# --- 추가 사항 --- #
linear_X = rospy.get_param('~circle_size')
msg.linear.x = linear_X
# ----------------- #

msg.linear.y = 0.0
msg.linear.z = 0.0
msg.angular.x = 0.0
msg.angular.y = 0.0
msg.angular.z = 1.8

rate = rospy.Rate(1)

while not rospy.is_shutdown():
	pub.publish(msg)
	rate.sleep()
```

msg.linear.x 대신 linear_X를 통해 즉, circle_size를 통해 linear.x를 받는다.

![value = 3](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2029.png)

value = 3

![value = 1](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2030.png)

value = 1

![Untitled](ROS%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%204ef87/Untitled%2031.png)

### 참고 자료

- [https://goldenboylife.com/elementor-2890/](https://goldenboylife.com/elementor-2890/)
- [ROS 공식 사이트]([http://wiki.ros.org/melodic/Installation/Ubuntu](http://wiki.ros.org/melodic/Installation/Ubuntu))