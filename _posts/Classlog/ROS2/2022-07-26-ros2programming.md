---
title:    "[ROS2 프로그래밍] ROS 프로그래밍 코드 스타일 및 프로그래밍 기초"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-07-26 00:35:00 +0800
categories: [Classlog, ROS2]
tags: [ROS2]
toc: True
comments: True
---

<https://cafe.naver.com/openrt/24436>

# ROS 프로그래밍 규칙 (코드 스타일)

협업 프로그래밍 작업 시 일관된 규칙을 만들고, 이를 준수하여야 코드가 꼬이거나 문제가 생기지 않는다.

ROS가 지원하느 프로그래밍 언어는 다양하지만, 가장 많이 사용되는 C++과 python만 살펴보자.

&nbsp;

## C++ style

ROS2 Developer Guide 및 ROS2 Code Style에서 다루는 C++ 코드 스타일은 오픈소스 커뮤니티에서 가장 많이 사용되는 Google C++ Style Guide를 사용한다.

- 이름 규칙
  - 타입, 클래스, 구조체, 열거형 : CamelCased
  - 파일, 패키지, 인터페이스, 네임스페이스, 변수, 함수, 메소드 : snake_case
  - 상수, 매크로 : ALL_CAPITALS
  - 전역 변수는 `g_` 접두어를 붙인다.
  - 클래스 멤버 변수는 마지막에 `_`를 추가한다.

- 공백 : 기본 들여쓰기는 space 2개를 사용한다.
- 함수, if, for 등의 괄호는 아래 줄로 넘겨서 시작한다.

```cpp
if (a == 0) 
{
    a = 1;
}
```

&nbsp;

## python style

ROS2 Developer Guide 및 ROS2 Code Style에서 다루는 python 코드 스타일은 pythib

- 이름 규칙
  - 타입, 클래스 : CamelCased
  - 파일, 패키지, 인터페이스, 변수, 함수, 메소드 : snake_case
  - 상수 : ALL_CAPITALS

- 공백 : 기본 들여쓰기는 space 4개를 사용한다.

&nbsp;

&nbsp;

# ROS 프로그래밍 기초 - python

ROS2에는 python에서 작업할 수 있도록 도와주는 rclpy가 있다.

## 패키지 생성

ROS2 패키지 생성 명령어는 다음과 같다.

- `ros2 pkg create [패키지이름] --build-type [빌드 타입] --dependencies [의존하는패키지1] [의존하는패키지2]..`

```bash
cd ~/robot_ws/src/
ros2 pkg create rclpy_tutorial --build-type ament_python --dependencies rclpy std_msgs
```

&nbsp;

이를 실행하고 나면 디렉토리는 다음과 같게 된다.

```markdown
src
    └── rclpy_tutorial
        ├── rclpy_tutorial
            └── \__init__.py
        ├── resource
            └── rclpy_tutorial
        ├── test
            └── test_
                ├── test_copyright.py
                ├── test_flake8.py
                └── test_pep257.py
        ├── package.xml
        ├── setup.cfg
        └── setup.py
```

이렇게 기본적으로 생성된 파일은 패키지 설정 파일인 package.xml, 파이썬 패키지 설정 파일인 setup.py, 파이썬 패키지 환경 설정 파일인 setup.cfg로 구성된다.

- package.xml

패키지 설정 파일은 사용할 RCL(ROS2 Client Libraries)에 따라 달라지는데, C++은 build type으로 ament_cmake가 사용되고, python은 ament_python으로 설정한다. 그 외에는 각기 다른 개발 환경에 맞춘 의존성 패키지를 설정해준다.

```xml
<!-- package.xml -->
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>rclpy_tutorial</name>
  <version>0.0.0</version>
  <description>TODO: Package description</description>
  <maintainer email="jhyoon@todo.todo">jhyoon</maintainer>
  <license>TODO: License declaration</license>

  <!-- packages -->
  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <!-- dependencies -->
  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

&nbsp;

- setup.py

기본적으로 구성되어 있는 코드는 아래와 같다.

```python
from setuptools import setup

package_name = 'rclpy_tutorial'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jhyoon',
    maintainer_email='jhyoon@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
```

&nbsp;

파이썬 패키지 설정 파일은 entry_points 옵션의 console_scripts를 사용하는 부분이 중요하다. 이 부분에서 helloworld_publisher와 helloworld_subscriber 콘솔 스크립트는 각각 rclpy_tutorial.heeloworld_publisher 모듈과 rclpy_tutorial.helloworld_subscriber 모듈의 main 함수가 호출되게 된다. 이를 통해 ros2 run 또는 ros2 launch 로 해당 스크립트를 실행시킬 수 있다.

```python
from setuptools import find_packages
from setuptools import setup

package_name = 'rclpy_tutorial'

setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Mikael Arguedas, Pyo',
    author_email='mikael@osrfoundation.org, pyo@robotis.com',
    maintainer='Pyo',
    maintainer_email='pyo@robotis.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='ROS 2 rclpy basic package for the ROS 2 seminar',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'helloworld_publisher = rclpy_tutorial.helloworld_publisher:main',
            'helloworld_subscriber = rclpy_tutorial.helloworld_subscriber:main',
        ],
    },
)
```

&nbsp;

- setup.cfg

파이썬 패키지 환경 설정 파일에서는 자신의 패키지 이름이 그대로 작성되어 있는지 확인해야 하며, 추후 colcon을 이용하여 빌드하게 되면 `/home/<username>/robot_ws/install/<package-name>/lib/<package-name>` 에 실행 파일이 생성된다.

```cfg
[develop]
script-dir=$base/lib/rclpy_tutorial
[install]
install-scripts=$base/lib/rclpy_tutorial
```

&nbsp;

## 노드 생성

- 퍼블리셔 노드 생성

퍼블리션 노드 소스 코드 파일은 `robot_ws/src/rclpy_tutorial/rclpy_tutorial/` 위치에 **helloworld_publisher.py** 파일을 생성한다. 아래의 코드는 **helloworld** 메시지를 보내주기 위한 코드이다.

```python
#!/usr/bin/env python
# -*- coding: utf8 -*-

import rclpy
from rclpy.node import Node 
from rclpy.qos import QoSProfile # 퍼블리셔의 QoS 설정을 위해 QoSProfile 클래스를 사용
from std_msgs.msg import String  # 퍼블리싱하는 메시지 타입은 std_msgs.msg의 String이므로 import

class HelloworldPublisher(Node): # rclpy의 Node 클래스를 상속하여 사용
    def __init__(self):
        super().__init__('helloworld_publisher') # Node 클래스의 이름을 helloworld_publisher라 지정
        qos_profile = QoSProfile(depth=10)       # 퍼블리시할 데이터 버퍼에 10개까지 저장
        self.helloworld_publisher = self.create_publisher(String, 'helloworld', qos_profile) # 퍼블리셔 노드 생성 (msg type, topic name, QoS)
        self.timer = self.create_timer(1, self.publish_helloworld_msg) # 콜백 함수 실행 시 사용되는 타이머로 지정한 값마다 콜백함수를 실행 (timer_period_sec, 발행을 실행할 함수)
        self.count = 0

    def publish_helloworld_msg(self):   # callback function, callback함수를 구현할 때는 멤버함수, lambda, 지역 함수 등으로 선언이 가능하다. 이 때는 멤버함수 사용
        msg = String()                  # 메시지 타입 - String
        msg.data = 'Hello World: {0}'.format(self.count) # 메시지의 data 입력
        self.helloworld_publisher.publish(msg) # 메시지 발행
        self.get_logger().info('Published message: {0}'.format(msg.data)) # print와 비슷한 함수로 기록용
        self.count += 1

def main(args=None):                    
    rclpy.init(args=args)               # 초기화 
    node = HelloworldPublisher()        # node라는 이름으로 클래스 생성
    try:
        rclpy.spin(node)                # 노드를 spin, 즉 지정된 콜백함수를 실행
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')   # ctrl+c와 같은 인터럽트 시그널을 받으면 반복을 끝냄
    finally:
        node.destroy_node() # 노드 소멸
        rclpy.shutdown()    # 노드 종료

if __name__ == '__main__':
    main()
```

&nbsp;

- 서브스크라이버 노드 작성

퍼블리셔 노드와 같은 디렉토리인 `/home/<username>/robot_ws/install/<package-name>/lib/<package-name>` 에 **helloworld_subscriber.py** 파일을 생성한다.

```python
#!/usr/bin/env python
# -*- coding: utf8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String


class HelloworldSubscriber(Node):

    def __init__(self):
        super().__init__('Helloworld_subscriber')
        qos_profile = QoSProfile(depth=10)          # 서브스크라이버 데이터를 버퍼에 10개까지 저장
        self.helloworld_subscriber = self.create_subscription(
            String,                         # 메시지 타입
            'helloworld',                   # 토픽 이름
            self.subscribe_topic_message,   # 콜백 함수
            qos_profile)                    # QoS

    def subscribe_topic_message(self, msg):
        self.get_logger().info('Received message: {0}'.format(msg.data)) # 데이터를 받으면 logging


def main(args=None):
    rclpy.init(args=args)           # 초기화
    node = HelloworldSubscriber()   # 클래스 생성
    try:
        rclpy.spin(node)            # 콜백함수 실행
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')   # 시그널 시 정지
    finally:
        node.destroy_node()         # 노드 소멸
        rclpy.shutdown()            # 노드 종료


if __name__ == '__main__':
    main()
```

서브스크라이버의 코드는 퍼블리셔와 거의 동일하다.

&nbsp;

## 빌드

ROS2에서는 colcon으로 빌드를 한다. 우선 소스 코드가 있는 workspace로 이동하고 colcon build 명령어를 통해 전체를 빌드한다. 여기에 빌드 옵션을 추가할 수 있는데, 특정 패키지만 선택하여 빌드하고자 할 때는 `--packages-select`, symlink를 사용하려면 `--symlink-install` 옵션을 사용한다.

```bash
(워크스페이스내의 모든 패키지 빌드하는 방법) 
$ cd ~/robot_ws && colcon build --symlink-install

(특정 패키지만 빌드하는 방법)
$ cd ~/robot_ws && colcon build --symlink-install --packages-select [패키지 이름1] [패키지 이름2] [패키지 이름N]

(특정 패키지 및 의존성 패키지를 함께 빌드하는 방법)
$ cd ~/robot_ws && colcon build --symlink-install --packages-up-to [패키지 이름]
```

따라서 위에 생성한 패키지만 빌드한다.

```bash
$ cd ~/robot_ws
$ colcon build --symlink-install --packages-select rclpy_tutorial
Starting >>> rclpy_tutorial
Finished <<< rclpy_tutorial [0.81s]

Summary: 1 package finished [1.00s]
```

특정 패키지의 첫 빌드 시에는 빌드 후 아래 명령어를 통해 환경 설정 파일을 불러와서 실행 가능한 패키지의 노드 설정들을 해줘야 빌드된 노드를 실행할 수 있다.

```bash
$ . ~/robot_ws/install/local_setup.bash
```

&nbsp;

## 실행

각 노드를 실행할 때는 `ros2 run`명령어를 통해 실행한다. 패키지 전체를 실행할 때는 `ros2 launch` 명령어를 사용할 수 있지만, 아직 launch를 생성하지 않았기에 우선 run으로 실행한다.

```bash
$ ros2 run rclpy_tutorial helloworld_subscriber
[INFO]: Received message: Hello World: 0
[INFO]: Received message: Hello World: 1
[INFO]: Received message: Hello World: 2
[INFO]: Received message: Hello World: 3
[INFO]: Received message: Hello World: 4
...

$ ros2 run rclpy_tutorial helloworld_publisher
[INFO]: Published message: Hello World: 0
[INFO]: Published message: Hello World: 1
[INFO]: Published message: Hello World: 2
[INFO]: Published message: Hello World: 3
[INFO]: Published message: Hello World: 4
...
```

&nbsp;

```bash
$ rqt_graph
```

- rqt_graph

<img src="/assets/img/ros2/rqt_graph.png">

&nbsp;

&nbsp;

# ROS 프로그래밍 기초 - C++

ROS2에는 python과 마찬가지로 c++을 위해 제작된 rclcpp가 있다. 코드도 앞서 배운 python과 거의 동일하다.

## 패키지 생성

```bash
$ cd ~/robot_ws/src/
$ ros2 pkg create rclcpp_tutorial --build-type ament_cmake --dependencies rclcpp std_msgs
```

ROS에서 C++을 사용하기 위한 클라이언트 라이브러리 rclcpp를 사용한다. 이렇게 하면 ROS1과 동일한 디렉토리 경로를 확인할 수 있다.

```markdown
src
    └── rclpy_tutorial
        ├── include
            └── rclcpp_tutorial
        ├── src
        ├── CMakeLists.txt
        └── package.xml
```

&nbsp;

생성된 package.xml, CMakeLists.txt은 각각 패키지 설정 파일과 빌드 설정 파일이다.

- package.xml

C++이라면 build type으로 ament_cmake를 사용한다.

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>rclcpp_tutorial</name>
  <version>0.0.1</version>
  <description>ROS 2 rclcpp basic package for the ROS 2 seminar</description>
  <maintainer email="jhyoon@todo.todo">jhyoon</maintainer>
  <license>Apache License 2.0</license>
  <buildtool_depend>ament_cmake</buildtool_depend>

  <!-- packages -->
  <depend>rclcpp</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

&nbsp;

- CMakeLists.txt

CMakeLists파일을 아래와 같이 수정한다.

```txt
cmake_minimum_required(VERSION 3.5)
project(rclcpp_tutorial)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Build
add_executable(helloworld_publisher src/helloworld_publisher.cpp)
ament_target_dependencies(helloworld_publisher rclcpp std_msgs)

add_executable(helloworld_subscriber src/helloworld_subscriber.cpp)
ament_target_dependencies(helloworld_subscriber rclcpp std_msgs)

# Install
install(TARGETS
  helloworld_publisher
  helloworld_subscriber
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Macro for ament package
ament_package()
```

&nbsp;

## 노드 생성

- 퍼블리셔 노드 생성

퍼블리셔 노드는 `~/robot_ws/src/rclcpp_tutorial/src/` 폴더에 **helloworld_publisher.cpp** 파일을 생성한다.

```cpp
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"            // Node 클래스를 사용하기 위한 rclcpp 헤더 파일
#include "std_msgs/msg/string.hpp"      // 메시지 타입인 String 선언

using namespace std::chrono_literals;   // 추후 500ms, 1s 와 같이 시간을 가시성 있게 문자로 표현하기 위한 namespace

class HelloworldPublisher : public rclcpp::Node // rclcpp의 Node 클래스를 상속하여 사용
{
public:
  HelloworldPublisher()
  : Node("helloworld_publisher"), count_(0)     // Node 클래스의 생성자를 호출, 노드 이름을 helloworld_publisher로 지정, count_는 0으로 초기화
  {
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));   //  QoS 설정을 위해 KeepLast 형태로 depth를 10으로 설정하여 퍼블리시할 데이터를 버퍼에 10개까지 저장
    helloworld_publisher_ = this->create_publisher<std_msgs::msg::String>( 
      "helloworld", qos_profile); // node클래스의 create_publisher함수를 이용하여 퍼블리셔 설정, 메시지 타입으로 String, 토픽 이름으로 helloworld, QoS
    timer_ = this->create_wall_timer(
      1s, std::bind(&HelloworldPublisher::publish_helloworld_msg, this)); // 콜백 함수를 수행, period=1초, 1초마다 지정한 콜백함수를 실행
  }

private:
  void publish_helloworld_msg()     // 콜백 함수
  {
    auto msg = std_msgs::msg::String(); // String 타입으로 msg 선언
    msg.data = "Hello World: " + std::to_string(count_++);  // 메시지 데이터를 입력
    RCLCPP_INFO(this->get_logger(), "Published message: '%s'", msg.data.c_str()); // logging, RCLCPP_XXX 계열의 함수는 print와 비슷
    helloworld_publisher_->publish(msg);    // publishing
  }
  rclcpp::TimerBase::SharedPtr timer_;      // private 변수
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr helloworld_publisher_;
  size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);     // rclcpp 초기화
  auto node = std::make_shared<HelloworldPublisher>(); // 클래스 생성
  rclcpp::spin(node);       // 콜백 함수 실행
  rclcpp::shutdown();       // ctrl+c와 같은 인터럽트 시그널 예외 상황에서 노드 종료
  return 0;
}
```

이때도, 멤버함수로 콜백함수를 선언했다. 만약 lambda로 구현하고자 한다면 publish_helloworld_msg 함수를 삭제하고 위의 HelloworldPublisher 클래스 생성자 구문의 timer_ = this->create_wall_timer() 함수에 람다 표현식을 추가하면 된다.

```cpp
    timer_ = this->create_wall_timer(
      1s,
      [this]() -> void
        {
          auto msg = std_msgs::msg::String();
          msg.data = "Hello World2: " + std::to_string(count_++);
          RCLCPP_INFO(this->get_logger(), "Published message: '%s'", msg.data.c_str());
          helloworld_publisher_->publish(msg);
        }
      );
```

&nbsp;

- 서브스크라이버 노드 생성

동일하게 `~/robot_ws/src/rclcpp_tutorial/src/`에 **helloworld_subscriber.cpp** 파일을 생성한다.

```cpp
#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using std::placeholders::_1; // bind 함수의 대체자 역할을 위해 _1로 선언

class HelloworldSubscriber : public rclcpp::Node // rclcpp의 Node클래스를 상속하여 사용
{
public:
  HelloworldSubscriber()
  : Node("Helloworld_subscriber")  // Node 클래스의 생성자를 호출하고 노드 이름을 helloworld_subscriber로 지정
  {
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));   // QoS depth 10으로 하여 버퍼에 10개 저장
    helloworld_subscriber_ = this->create_subscription<std_msgs::msg::String>(
      "helloworld",
      qos_profile,
      std::bind(&HelloworldSubscriber::subscribe_topic_message, this, _1)); // 구독할 토픽의 메시지 타입과 토픽의 이름, QoS 설정, 수신받은 메시지를 처리할 콜백함수를 기입
  }

private:
  void subscribe_topic_message(const std_msgs::msg::String::SharedPtr msg) const // 콜백함수
  {
    RCLCPP_INFO(this->get_logger(), "Received message: '%s'", msg->data.c_str()); // logging
  }
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr helloworld_subscriber_; // private 변수로 사용되는 helloworld_subscriber_ 선언
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HelloworldSubscriber>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
```

&nbsp;

## 빌드

rclpy를 빌드할 때와 동일하다.

```bash
$ cd ~/robot_ws
$ colcon build --symlink-install --packages-select rclcpp_tutorial
Starting >>> rclcpp_tutorial
Finished <<< rclcpp_tutorial [11.8s]

Summary: 1 package finished [12.0s]
```

마지막으로 동일하게 bash파일을 실행시켜준다.

```bash
$ . ~/robot_ws/install/local_setup.bash
```

&nbsp;

```bash
$ ros2 run rclcpp_tutorial helloworld_subscriber
[INFO]: Received message: 'Hello World: 0'
[INFO]: Received message: 'Hello World: 1'
[INFO]: Received message: 'Hello World: 2'
[INFO]: Received message: 'Hello World: 3'
[INFO]: Received message: 'Hello World: 4'
...
```

```bash
$ ros2 run rclcpp_tutorial helloworld_publisher
[INFO]: Published message: 'Hello World: 0'
[INFO]: Published message: 'Hello World: 1'
[INFO]: Published message: 'Hello World: 2'
[INFO]: Published message: 'Hello World: 3'
[INFO]: Published message: 'Hello World: 4'
...
```