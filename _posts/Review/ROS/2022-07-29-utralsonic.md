---
title:    "[ROS] ROS Ultrasonic Data processing "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-05-15 03:00:00 +0800
categories: [Review, ROS]
tags: [ROS, ultrasonic]
toc: True
comments: True
---

초음파 센서를 가지고, 직접 데이터를 받아보는 방법에 대해 설명하고자 한다. 먼저 초음파 센서 장비는 가지고 있다는 가정 하에 진행한다.

<img src="/assets/img/dev/week4/day3/wavesensor.jpg">

&nbsp;

기술 스택
- Arduino
- ROS
- python

&nbsp;

&nbsp;

# 1. 초음파 센서 데이터 받아오기 (Arduino)

아두이노를 실행하려면, IDE를 설치해야 한다.

[다운로드 링크](https://www.arduino.cc/en/software)

다운로드를 한 후, 압축을 풀면, 안에 `install.sh` 라느 파일이 있다. 이를 실행한다.

- 압축 풀기

```bash
tar xvf arduino-1.8.19-linux64.tar.xz && rm -rf arduino-1.8.19-linux64.tar.xz
```

&nbsp;

- install.sh 실행

```bash
$ cd arduino-1.8.19 && sudo sh install.sh
Adding desktop shortcut and menu item for Arduino IDE...

 done!
```

&nbsp;

- 아두이노 실행

```bash
sudo arduino
```

&nbsp;

- 초음파센서와 pc 연결 확인

```bash
$ lsusb
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
```

&nbsp;

여기서 wsl을 사용한다면 lsusb로 나오지 않을 수 있다. 따라서 usb연결을 진행해야 한다. 만약 네이티브 우분투를 사용할 경우는 그냥 넘어가도 좋다.

---

- [참고 자료](https://docs.microsoft.com/ko-kr/windows/wsl/connect-usb)

1. usbipd 설치

usb 디바이스 연결에 대한 지원은 wsl에서 기본적으로 사용할 수 없으므로 usbipd-win 을 설치해야 한다.

powershell을 켜서 명령어를 실행한다.

```bash
> winget install --interactive --exact dorssel.usbipd-win
'msstore' 원본을 사용하려면 다음 계약을 확인해야 합니다.
Terms of Transaction: https://aka.ms/microsoft-store-terms-of-transaction
원본이 제대로 작동하려면 현재 컴퓨터의 두 글자 지리적 지역을 백 엔드 서비스로 보내야 합니다(예: "미국").

모든 원본 사용 약관에 동의하십니까?
[Y] 예  [N] 아니요: y
찾음 usbipd-win [dorssel.usbipd-win] 버전 2.3.0
이 응용 프로그램의 라이선스는 그 소유자가 사용자에게 부여했습니다.
Microsoft는 타사 패키지에 대한 책임을 지지 않고 라이선스를 부여하지도 않습니다.
Downloading https://github.com/dorssel/usbipd-win/releases/download/v2.3.0/usbipd-win_2.3.0.msi
  ██████████████████████████████  10.4 MB / 10.4 MB
설치 관리자 해시를 확인했습니다.
패키지 설치를 시작하는 중...
설치 성공
```

&nbsp;

설치가 다 되면, 설치 창이 하나 뜬다. 설치하면 된다.

&nbsp;

2. usbip 도구 및 하드웨어 데이터베이스 설치

Linux에서 아래 명령어를 사용하여 USB 하드웨어 식별자 데이터베이스를 설치해야 한다.

```bash
$ sudo apt install linux-tools-5.4.0-77-generic hwdata
$ sudo update-alternatives --install /usr/local/bin/usbip usbip /usr/lib/linux-tools/5.4.0-77-generic/usbip 20
update-alternatives: using /usr/lib/linux-tools/5.4.0-77-generic/usbip to provide /usr/local/bin/usbip (usbip) in auto mode
```

이를 다 진행하고 나면, Windows와 Linux와의 디바이스 공유 도구가 설치되어 있을 것이다.

&nbsp;

3. usb 디바이스 연결

```bash
> usbipd wsl list
BUSID  VID:PID    DEVICE                                                        STATE
2-5    04e8:730b  CanvasBio Fingerprint Driver                                  Not attached
2-6    2b7e:0134  720p HD Camera                                                Not attached
2-7    1a86:7523  USB-SERIAL CH340(COM5)                                        Not attached
2-10   8087:0026  인텔(R) 무선 Bluetooth(R)                                     Not attached
2-13   0bc2:231a  UAS(USB Attached SCSI) 대용량 저장 장치                       Not attached
```

```bash
> usbipd wsl attach --busid 2-7
usbipd: info: Using default distribution 'Ubuntu-18.04'.
```

&nbsp;

이 후, ubuntu로 들어와, 다음 명령어를 사용한다.

```bash
$ lsusb
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 002: ID 1a86:7523 QinHeng Electronics HL-340 USB-Serial adapter
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
```

---

&nbsp;

&nbsp;

- arduino 소스코드

```arduino
/*
HL-340 초음파 센서 아두이노 펌웨어
*/

#define trig 2 // 트리거 핀 선언
#define echo 3 // 에코 핀 선언

void setup()
{
    Serial.begin(9600);     // 통신속도 9600bps로 시리얼 통신 시작
    pinMode(trig, OUTPUT);  // 트리거 핀을 출력으로 선언
    pinMode(echo, INPUT);   // 에코핀을 입력으로 선선
}

void loop() {
    long duration, distance;    // 거리 측정을 위한 변수 선언
    // 트리거 핀으로 10us 동안 펄스 출력
    digitalWrite(trig, LOW);    // Trig 핀 Low
    delayMicroseconds(2);       // 2us 딜레이
    digitalWrite(trig, HIGH);   // Trig 핀 High
    delayMicroseconds(10);      // 2us 딜레이
    digitalWrite(trig, LOW);    // Trig 핀 Low

    // pulseln() 함수는 핀에서 펄스 신호를 읽어서 마이크로초 단위로 반환
    duration = pulseIn(echo, HIGH);
    distance = duration * 170 / 1000;   // 왕복시간이므로 340/2=170 곱하는걸로 계산, 마이크로초를 mm로 변환하기 위해 1000나눔
    Serial.print("Distance(mm): ");
    Serial.println(distance);           // 거리정보를 시리얼 모니터에 출력
    delay(100);
}
```

&nbsp;

&nbsp;

# 2. 초음파 센서 데이터 받아오기 (ROS)

아두이노에서 보내주는 물체까지의 거리 정보를 받아와 토픽으로 Publishing하는 코드를 만든다.

- ultrasonic_pub.py

```python
#!/usr/bin/env python

import serial, time, rospy
from std_msgs.msg import Int32

ser_front = serial.Serial( 
    port='/dev/ttyUSB0', # 아두이노가 연결된 포트
    baudrate=9600,       # 아두이노에서 선언한 통신 속도
)

def read_sensor():
    serial_data = ser_front.readline() # 시리얼 포트로 들어온 데이터를 받아옴
    ser_front.flushInput()  # 중간에 버퍼들이 있어서 그를 삭제해주는 flush
    ser_front.flushOutput()
    ultrasonic_data = int(filter(str.isdigit, serial_data)) # string을 숫자로 변환
    msg.data = ultrasonic_data

if __name__ == '__main__':
    rospy.init_node('ultrasonic_pub', anonymous=False)
    pub = rospy.Publisher('ultrasonic', Int32, queue_size= 1)

    msg = Int32()
    
    while not rospy.is_shutdown():
        read_sensor() # 시리얼포트에서 센서가 보내준 문자열 읽엇거 거리 정보 추출
        pub.publish(msg)    
        time.sleep(0.2) # 토픽에 담아서 publish

    ser_front.close() # 끝나면 시리얼포트 닫기
```

&nbsp;

## 초음파 센서 데이터 viewer

초음파 데이터가 정확하게 잘 도착하는지, 데이터가 ROS로 잘 전달되고 있는지 확인하기 위한 검증 코드이다.

- ultrasonic_sub.py

```python
#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32

def callback(msg):
    print(msg.data)

rospy.init_node('ultrasonic_sub')
sub = rospy.Subscriber('ultrasonic', Int32, callback)

rospy.spin()
```

&nbsp;

&nbsp;

# 실행

실행을 위한 launch파일을 제작한다.

- ultra.launch

```xml

<launch>
    <node pkg="ultrasonic" type="ultrasonic_pub.py" name="ultrasonic_pub"/>
    <node pkg="ultrasonic" type="ultrasonic_sub.py" name="ultrasonic_sub" output="screen"/>
</launch>
```

&nbsp;

- 실행

```bash
roslaunch ultra ultra.launch
```


&nbsp;

