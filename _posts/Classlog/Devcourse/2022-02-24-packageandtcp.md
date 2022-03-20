---
title:    "[데브코스] 2주차 - linux 기초(File System)"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-02-24 12:00:00 +0800
categories: [Classlog, devcourse]
tags: [linux, tcp, devcourse]
toc: True
comments: True
---

<br>

[오답노트](https://www.notion.so/c4463b5ec83e4351b8eba4df97236b88)

package 관리
- redhat

```markdown
rpm database
*yum
```

- debian

```markdown
dpkg(옛날거)
apt-get
*apt
```

<br>

<br>

# package

package : 시스템을 구성하는 파일의 묶음

관리의 편리함을 제공한다.

- package name - version & release - architecture.확장자
- amd64는 x86 64bit

apt를 주로 사용하고, dpkg는 중요한 기능 몇가지만 알아두면 된다.

## dpkg 중요 기능

`strace, gcc` 패키지 리스트를 확인해보자.

- 옵션: -l (list)

설치되지않아도 찾아볼 수 있다.

```bash
$ dpkg -1 strace

$ dpkg -1 gcc
```

- 옵션: -s (status)

status에 install ok installed 이면 설치된 것, install ok unpacked는 설치된 것이 아니다.

```bash
# dpkg -s strace
status: install ok installed
```

- 옵션: -S (search)

패키지 검색, 해당 파일이 어디에 존재하는지 확인 가능

```bash
# dpkg -S '*trace'
linux-headers-3.2.0-4-common: /usr/...
.
.
.
```

에러가 나면 해결하는 방법으로 audit을 사용하기도 한다.

<br>

## apt

apt는 debian에서 사용하는 툴로 가장 많이 사용하는 명령어다.

dependency 탐색 및 설치 가능
- e.g. A를 설치하기 위해 B를 설치해야 하고, B를 위해 C를 설치해야 한다면 그것들을 탐색해준다.

예전에는 apt-get, apt-cache 등등이 있는데 요즘에는 이들을 통합한 apt를 사용한다. 

<br>

apt를 쓰기 위해서는 source list를 만들어야 한다.

apt를 자동으로 다운하는데 어디에 패키지가 있는지 알아야 하기 때문에 그것을 지정

```bash
$ vim /etc/apt/sources.list
```

직접 추가할 경우 etc/apt/sources.list.d/에 *.list 파일명으로 추가해도 되지만, **가장 좋은 방법은 apt edit-sources를 사용하여 수정하는 것이다.**

sources.list안에는 `deb`와 `deb-src`가 있다. **deb는 패키지를 받아오는 주소, deb-src는 소스코드를 받아오는 주소**이다.

```markdown
deb [option1=value1 option2=value2] uri suite [component1] [component2] ...
```

- uri : deb 패키지를 제공하는 사이트의 URI
- 옵션은 대부분 필요x
- suite : 코드네임 디렉코리 이름을 뜻하는데, 16.04 = xenial, 18.04 = bionic
- component : suite의 구성 요소 및 라이선스 종류별 분류, 최소 1개 이상의 컴포넌트를 지정해야 함
    - **main: 우분투에서 직접 패키징하는 것들**
    - **restricted: 제한된 무료 라이선스**
    - **universe: 대체로 절반 무료 라이선스, 대부분이 이에 해당됨**
    - security
    - updates

<br>

kakao uri의 apt를 참고하고자 한다. 그 이유는 kakao uri가 대체로 더 빠르다.

```bash
$ sudo apt edit-sources kakao.list
```

실행 한 후 아래 코드를 추가한다.

```bash
deb http://mirror.kakao.com/ubuntu/ bionic main restricted universe
deb http://mirror.kakao.com/ubuntu/ bionic-updates main restricted universe
deb http://mirror.kakao.com/ubuntu/ bionic-security main restricted universe
```

1. 편집할 파일은 /etc/apt/sources/list.d/kakao.list 이다.

> `# sudo select-editor vim` 을 실행하여 기본 에디터를 vim으로 변경핤 수 있다.

2. 먼저 [http://mirror.kakao.com/ubuntu/](http://mirror.kakao.com/ubuntu/에) 에 접속한다. 
3. 그러면 `dists/`폴더가 있다. 
4. 여기를 들어가서 리스트를 쭉 보고 설치할 패키지 이름을 본다. 
5. bionic, bionic-updates, 등이 있다.
6. `# sudo apt edit-sources kakao.list` 작성

<br>

### 패키지 목록 출력

- list

```bash
apt list [option] [package pattern]
```

```bash
모든 리스트 출력
$ apt list

설치된 것만
$ apt list --installed

업그레이드 가능한 패키지만
$ apt list --upgradable

모든 버전
$ apt list --all-versions

package pattern을 설정하여 그에 해당하는 것만 출력
$ apt list bash*
```

<br>

## search

```bash
apt search [-n] <regex>
```

```bash
name이 아닌 설명에 bash가 들어간 경우까지 검색
$ apt search bash

name 중간에 bash가 있어도 검색
$ apt search -n bash

시작 부분에 bash가 있는 경우만 검색
$ apt search -n '^bash'
```

<br>

## show

```bash
apt show <package name>[=version]
```

```bash
정보가 나옴
$ apt show bash

모든 버전 보기 위함
$ apt list --all-versions bash

한 개의 버전만
$ apt show bash=4.4.18-2ubuntu1
```

<br>

## apt remove, purge, autoremove

```bash
apt <remove|purge|autoremove> <package>[=version]
```

- remove: 패키지만 삭제 - (config파일은 남겨둠) 설정이 꼬여서 재설치를 할 때
- purge: 패키지 삭제 - 완전 삭제, 다시는 안쓸 것 같다.
- autoremove: 의존성이 깨지거나 버전 관리로 인해 쓰이지 않는 패키지 자동 제거

```bash
$ apt -y install htop

$ apt show htop

$ apt purge htop
```

패키지 이름만 보고 싶은 경우에는 apt-cache를 사용

```bash
pcp가 들어가는 패키지 다
$ apt search -n '^pcp*'

pcp가 들어가는 이름만
$ apt-cache pkgnames pcp

-y를 뒤에 적어도됨
$ apt install pcp -y
```

<br>

<br>

# 네트워크에 필요한 기초 용어

- hostname: primary hostname, FQDN
- TCP/IP : IP address(IPv4, IPv6), subnet mask, gateway
- NIC: Network Interface Care == 랜카드
- Wired Network (Wired connection) : 유선 네트워크 (유선 연결)
- Wireless Network (Wireless connection) : 무선 네트워크 (무선 연결)
- LAN : Local Area Network
- WAN : Wide Area Network

<br>

## hostname

`컴퓨터의 이름 : access.redhat.com`이라 하면, 호스트이름은 사람의 이름과 비슷하게 만들어져 있다. 컴퓨터의 성은 `redhat.com`이 된다. 도메인주소는 사람의 성에 해당하는 것으로 `redhat.com`이다.

<br>

## FQDN, hostname

hostname에는 두가지 중의적 의미가 있다.

1. domain을 제외한 호스트 이름
2. domain을 포함한 FQDN

FQDN: fully qualifed domain name
- 도메인 내에서 유일하게 구별 가능한 이름, 즉 겹치지 않게 구별해주는 이름
- e.g. [fedora.redhat.com](http://fedora.redhat.com) = hostname = FQDN

도메인 주소는 체계적인 구조를 가진다. 
- e.g. devel.fclinux.or.kr
    - kr: 한국의 주소
    - or: 단체
    - fclinux: 단체의 이름
    - devel: 단체 내에서 유일한 이름

hostname 중에서 special hostname라는 것이 있다.

> localhost
: 항상 자기 자신을 의미하는 주소와 맵핑된다.
    - IPv4 = 127.0.0.1
    - IPv6 = ::1
>

<br>

## IP주소

IPv4
- 32bit 주소 체계, 8bit씩 끊어서

IPv6
- 128bit 주소 체계

<br>

IPv6에는 IPv4를 포함한 주소 표기법이 있다. 이를 IPv4-mapped IPv6이라 한다.
- e.g. 58.232.1.100 ⇒ ::ffff:58.232.1.100

<br>

<br>

### CIDR(Classless Inter-Domain Routing)

CIDR : IP 클래스와 상관없이 **서브넷**을 지정하여 자르는 것을 의미
- xxx.xxx.xxx.xxx**/##**
- ##에는 서브넷 매스크의 on 배트의 개수를 표기
- 111.111.111.11/24

<br>

public IP/private IP
- public IP: 공인 주소(인터넷에서 유일한 주소)
- private IP: 사설 주소(인터넷에 직접 연결되지 않는 유일하지 않은 주소)

<br>

<br>

## SELinux

Securiy Enhanced Linux : 커널 레벨에서의 중앙 집중식 보안 기능

- 대부분의 리눅스는 기본으로 설치되지만, ubuntu는 설치되지 않는다

<br>

SELinux의 보안레벨
- enforcing (강제) : SELinux를 사용, 보안설정에 걸리면 강제로 막음
- permissive (허가) : SELinux를 사용, 보안설정에 걸리면 허용하되 로그 출력
- disabled (비활성) : SELinux를 사용하지 않음

서버 구성시에는 SELinux에 대한 이해가 필요하다. 실습을 할 때는 허가레벨이 적절하다.

<br>

<br>

## debian계열 Network Configuration

네트워크 설정은 2가지 방식이 있다.
1. legacy
2. networkmanager

여기서 네트워크 매니저만 사용한다. 사용하면 안되는 legacy부분을 간단히 볼 것이다.

- Debian

`/etc/network/interfaces` 에서 설정하고,ifdown,ifup,ifconfig, eth0를 사용하는 것은 다 옛날 버전이다.

- RedHat

/etc/sysconfig/netowkr-scripts/ifcfg-* 를 사용하는 것은 다 옛날 버전이다.

<br>

<br>

### networkManager

장점
- daemon으로 작동하면서 network cofiguration을 수행
- 자동으로 network connection을 관리
- Dbus 기반으로 동적 상태를 감지할 수 있기 때문에 다른 애플리케이션이나 daemon들에게 네트워크 정보를 제공하거나, 관리를 위한 권한을 줄 수 있다
- 통일된 명령어를 사용하여 **systemd** 기반의 다른 Linux distribution들에게도 동일한 방식의 경험을 제공할 수 있다.
- **Ethernet, wi-fi, moblie broadband** 등 다양한 기능들에게 플랫폼을 제공하므로, 네트워크 연결 관리가 좀 더 쉬워졌다.

<br>

<br>

# NMCLI (network manager CLI tool)

- 네트워크에 관련된 대부분의 기능을 가지고 있다.
- 조회 및 설정 가능 (root계정이거나 sudo 를 사용해야 함)

## nmcli g[eneral]

현재 상태 조회

```bash
# nmcli g
STATE   CONNECTIVITY  WIFI-HW  WIFI  WWAN-HW  WWAN 
연결됨  전체          사용     사용  사용     사용
```

state : connected / asleep

connectivity : full/none

## nmcli n[etworking]

네트워크 상태 조회

네트워크를 끊거나 연결할 때는

- nmcli n on
- nmcli n off

간혹 off되어 있어서 설정이 안되는 경우가 있다. 따라서 상태부터 조회해봐야 한다.

```bash
# nmcli n
enabled

# nmcli n connectivity
full
```

## nmcli d[evice]

장치 확인

```bash
# nmcli d
DEVICE  TYPE      STATE          CONNECTION  
ens33   ethernet  연결됨         유선 연결 1 
lo      loopback  관리되지 않음  --
```

<br>

> 추가 정보
    - en : ethernet
    - wl : wireless lan
    - ww : wireless wan
>
> 그 뒤에 붙는 것들이
o\<index> : on-board device index number 
s\<slot> : hotplug slot index number
p\<bus> : PCI location 
>

<br>

## nmcli r[adio]

무선 관련 설정

```bash
# nmcli r
WIFI-HW  WIFI     WWAN-HW  WWAN    
enabled  enabled  enabled  enabled
```

<br>

## nmcli c[onnection] s\[how]\(default)

디바이스와 이름의 연결 상태와 넘버,타입 등

```bash
# nmcli c
NAME         UUID                                  TYPE      DEVICE 
유선 연결 1  226b498c-6a18-3983-869d-665b9c680b39  ethernet  ens33

# nmcli c s

특정 연결 이름에 대한 설정된 속성
# nmcli c s ens33
```

이 때, cmcli c s ens33 을 했을 때 소문자로 된 것과 대문자로 된 것들이 있다.

소문자는 설정된 값(offline일때도 보임), 대문자는 할당된 값(online일때만 보임)

<br>

**주요 속성**
- ipv4.method
    - auto | manual
    - auto = dhcp
    - manual = static ip
        - ip주소에 대해서는 다음과같이 적어야 한다.
            - ipv4.addr = CIDR 표기법 : 192.168.110.50/24 라면 24이므로 앞에 192.168.110까지만
- 옵션
    - \+ : 기존의 것에 추가
    - \- : 기존의 것에서 삭제
    - none : 교체

<br>

```bash
# nmcli d
DEVICE  TYPE      STATE      CONNECTION  
ens33   ethernet  connected  유선 연결 1 
lo      loopback  unmanaged  --

# nmcli con down "유선 연결 1"
Connection '유선 연결 1' successfully deactivated (D-Bus active path: /org/freedesktop/NetworkManager/ActiveConnection/1)

# nmcli c s "유선 연결 1"
connection.id:                          유선 연결 1
connection.uuid:                        226b498c-6a18-3983-869d-665b9c680b39
connection.stable-id:                   --
connection.type:                        802-3-ethernet
connection.interface-name:              --
connection.autoconnect:                 yes
connection.autoconnect-priority:        -999
connection.autoconnect-retries:         -1 (default)
connection.auth-retries:                -1
...

# nmcli con up "유선 연결 1"
Connection '유선 연결 1' successfully activated (D-Bus active path: /org/freedesktop/NetworkManager/ActiveConnection/1)

IP4.* 부분 관찰
# nmcli c s "유선 연결 1"
```

<br>

## nmcli 속성 변경 - 이름 변경

한글인 이름을 변경시켜본다.

```bash
# nmcli con modify “변경전이름” 속성 변경후이름
```

<br>

```bash
# nmcli d
DEVICE  TYPE      STATE      CONNECTION
ens33   ethernet  connected  유선 연결 1
lo      loopback  unmanaged  --

# nmcli con modify "유선 연결 1" connection.id ens33
# nmcli d
DEVICE  TYPE      STATE      CONNECTION 
ens33   ethernet  connected  ens33      
lo      loopback  unmanaged  --
```

<br>

### nmcli 속성 변경 - ip주소 변경

ip 변경 전에 현재 시스템의 ip부터 메모해놓아야 한다.

```bash
# nmcli c s
NAME   UUID                                  TYPE      DEVICE 
ens33  226b498c-6a18-3983-869d-665b9c680b39  ethernet  ens33


# nmcli c s ens33
ipv4.method:                            auto
ipv4.dns:                               --
...
IP4.ADDRESS[1]:                         192.168.40.128/24
IP4.GATEWAY:                            192.168.40.2
IP4.ROUTE[1]:                           dst = 0.0.0.0/0, nh = 192.168.40.2, mt = 100
...
```

- auto = DHCP

<br>

```bash
# nmcli c mod ens33 ipv4.method manual ipv4.addresses 192.168.40.128/24 \
>ipv4.gateway 192.168.40.2 +ipv4.dns 8.8.8.8



대문자는 그대로지만, 소문자 속성만 바뀌어 있다.
# nmcli c s ens33
ipv4.method:                            manual
ipv4.dns:                               8.8.8.8
IP4.ADDRESS[1]:                         192.168.40.128/24
IP4.GATEWAY:                            192.168.40.2



다시 원상복구
이 때 &&은 앞에꺼가 True 이면 뒤에꺼도 실행
# nmcli c down ens33 && nmcli c up ens33
Connection 'ens33' successfully deactivated (D-Bus active path: /org/freedesktop/NetworkManager/ActiveConnection/2)
Connection successfully activated (D-Bus active path: /org/freedesktop/NetworkManager/ActiveConnection/3)
```

<br>

<br>

### virtual IP 추가/삭제

```bash
# nmcli c mod ens33 +ipv4.addr 192.168.40.181/24
# nmcli c up ens33

# nmcli c mod ens33 -ipv4.addr 192.168.40.181/24
```

<br>

### 기존의 설정을 삭제했다가 새로 생성

```bash
# nmcli n del ens33

# nmcli c s

device 정보는 나오지만 connection은 없음
# nmcli d s

ifname이 디바이스 이름
# nmcli c add con-name ens33 ifname ens33 type ethernet \
> ip4 192.168.50.128/24
```

<br>

+ 만약 디바이스 자체가 올라오지 않는 경우

```bash
status 확인
# nmcli dev s


disconnected라고 되어 있으면 connect로 바꿔야 함
# nmcli dev connect ens33


connect가 실패한다면 status 확인
# nmcli g


asleep이라면 이 명령은 되지 않을 수 있다. 되지 않는다면
# nmcli device connect ens33


# nmcli networking


disabled라면
# nmcli networking on


확인
# nmcli g
```

<br>

<br>

# ss (socket statistics)

> netstat은 구식 명령어

<br>

## 상태 확인

```bash
# ss -ntl
```

옵션
- -n : —numeric
- -a : —all
- -l : —listening
- -e : —extended
- -o : —options
- -m : —memory
- -p : —processes

<br>

filter로 state, address를 지정하여 추출할 수 있다.

filter = [state TCP-state] [EXPRESSION]

- TCP-STATE
    - established
    - fin-wait-2
    - close-wait

<br>

<br>

## TCP state

<img src="assets/img/dev/week2/day4/tcpstate.png">

- 실선: 클라이언트
- 점선: 서버

1. 서버 측에서 요청을 들을 수 있도록 listen을 하면 listen 상태가 된다.
2. 클라이언트에서도 connect를 해야하고, 하게 되면 SYN_SENT (싱크 보내기)를 한다.
3. 받은 싱크는 무조건 다시 echo, ack를 해줘야 한다. 싱크를 보낸측을active opne이라 하고, 받은 측을 passive open 이라 부른다. 
4. 싱크를 보내는 것을 SYN_SENT → 싱크를 잘 받았다고 다시 보내는 것을 SYN_RCVD → 또 그것을 받았다는 것을 ack해준다. 이를 three way handshaking
5. 3번 핸드쉐이킹이 끝나면 established 상태가 된다.
6. 데이터를 주고 받음
7. 종료할 때는 대체로 클라이언트가 종료하게 되므로 클라이언트가 active close, 서버가 passive close ,, 그러나 서버측이 먼저 끊을 수도 있다. 그러면 저 FIN부터 다 반대로 작동한다고 보면 된다.
8. close를 하게 되면 바로 ack와 echo를 돌려보내줘야 하는데, 먼저 ack를 주고받는다. 이를 기다리는 것이 FIN_WAIT1, FIN_WAIT2가 echo
9. echo를 잘 받았다는 ack를 기다리는 것이 LAST_ACK
10. 중복되어 받는 것들을 처리해주는 것을 TIME_WAIT

<br>

### three-way handshaking

TCP 접속을 만드는 과정

- 3번 왔다갔다 하기 때문
- 문제가 발생하기 드뭄

### Four-way handshaking

TCP 접속을 해제하는 과정

- 4번 왕복하지만 아주 드물게 동시에 접속이 해제되면 3번만에 끝난다.
- **문제가 종종 생긴다. 특히 passive close에서**

<br>

🎈 close-wait 와 fin-wait-2 는 심각한 문제다. 그 중 close-wait는 시한 폭탄과도 같다. 

<br>

위 2가지 TCP상태가 발생하는지 확인하기 위해서는 다음과 같이 명령

```bash
# ss -nt state close-wait state fin-wait-2

1초마다(-n 1) 확인(watch)하기 위는 방법
# watch -d -n 1 ss -nt state close-wait state fin-wait-2
```

## ss -nt

option : numeric, tcp

```bash
# ss -nt
State   Recv-Q    Send-Q        Local Address:Port        Peer Address:Port
ESTAB   123123    0             192.168.0.71:40632        42.123.16.377.364:12370
ESTAB   123123    0   [::ffff::192.168.0.71]:40632        42.123.16.377.364:12370
```

- ESTAB: established 
- Q: 소켓 버퍼를 의미, 저 숫자만큼 쌓여 있다는 것
- ffff가 없는 것은 IPv4, 있는 것은 IPv6-mapped ipv4

<br>

## ss state \<tcp state>

ESTAB 상태를 보려면

```bash
# ss state established

numeric과 ipv6만 보여달라는 옵션
# ss -n6 state established
```

<br>

## address filter

주소나 포트 번호를 통해 필터를 걸 수도 있다. 지정을 위해 2가지 방법이 있는데, symbolic보다는 literal을 더 많이 쓴다.

| symbolic | literal | description |
| --- | --- | --- |
| > | gt | greater than |
| < | ls | less than |
| >= | ge | greater than or equal |
| <= | le | less than or equal |
| == | eq | equal |
| != | ne | not equal |
|  | and | and |
|  | or | or |

```bash
dport가 22인거만 추출, 작은따옴표로 묶어주는 것이 좋다.
# ss -n 'dport = :22'
```

<br>

## ss -s

option: statistics

통계정보, 열고 있는 통계정보, 몇개를 열고 있는지 볼 수 있는 중요한 기능

<br>

## ss -utlp

프로세스의 정보를 보여준다. 할 때는 root계정으로 해야 다 볼 수 있다.

```bash
user에서 첫번째값은 실행파일의 이름, pid ,fd 값들임 
# ss -utlp
State    Recv-Q    Send-Q        Local Address:Port       Peer Address:Port                                                                                   
LISTEN   0         128           127.0.0.53%lo:53              0.0.0.0:*        users:(("systemd-resolve",pid=561,fd=13))                                     
LISTEN   0         5                 127.0.0.1:631             0.0.0.0:*        users:(("cupsd",pid=712,fd=7))                                                
LISTEN   0         5                     [::1]:631                [::]:*        users:(("cupsd",pid=712,fd=6))



# ss -n 'src 192.168.110.0/24'




dport가 ssh인거거나 sport가 ssh인것들을 보여달라
# ss -n 'dport = :ssh or sport = :ssh'




dport가 ssh이면서 sport가 ssh인 것(여기서 괄호 사이에 공백은 일부러 넣은것으로 안하면 오루가 날 수도 있다)
그리고 src번호가 이것인거만 추출
# ss -n '( dport = :ssh or sport = :ssh ) and src 192.168.110.134'
```

192.168.110.0/24 라는 필터를 걸면 24bit 즉, 앞에 8bit씩 끊어서 앞에 192.168.110인것들을 모두 보여준다는 의미다.

<br>

## ss와 같이 쓰이는 명령어

### lsof, fuser

열린 파일을 검색하거나 액션을 행할 수 있는 기능을 가진다. 특정 소켓 주소를 점유하고 있는 프로세스를 찾아내거나 할 때 사용한다.

<br>

<br>

# ping 또는 server 관련 명령어

## ping

상대 호스트의 응답을 확인함

```bash
ping [-c count] [-i interval] [-s size] [-t ttl] target
```

count를 지정하지 않으면 계속 보내질수도 있으며, interval을 지정하지 않으면 초단위로 보내진다. 다 하고 나면 통계정보(min,avg,max,mdev등)를 알려준다. 이 때 표준편차를 주의깊게 봐야 한다.

```bash
# ping -c 3 192.168.0.1
PING 192.168.0.1 (192.168.0.1) 56(84) bytes of data.

--- 192.168.0.1 ping statistics ---
3 packets transmitted, 0 received, 100% packet loss, time 2043ms

0.2sec 간격으로 2000번 테스트
# ping -c 2000 -i 0.2 -s 1000 102.168.1.1
# ping -c 2000 -i 0.2 -s 1000 192.168.1.1
PING 192.168.1.1 (192.168.1.1) 1000(1028) bytes of data.

--- 192.168.1.1 ping statistics ---
1064 packets transmitted, 0 received, 100% packet loss, time 216897ms
```

<br>

## arp

arp 테이블: IP와 MAC 주소를 매핑해주는 것

```bash
# arp
Address                  HWtype  HWaddress           Flags Mask            Iface
_gateway                 ether   00:50:56:f4:31:a8   C                     ens33
```

수동으로 매핑해줄 수도 있다.

> NIC 교체 후 통신이 실패한다면 → 고정 IP주소를 사용하는 기관이나 회사의 경우 보안이나 IP주소 관리를 위해 고정 ARP테이블을 사용한다. 
따라서 ARP 테이블을 확인하여 IP와 MAC주소가 제대로 매칭되는지 확인(nmcli , arp), 교체하면 네트워크 관리자에게 NIC의 MAC과 IP를 알려주어야 함
> 

<br>

## resolver : name service

IP address나 hostname을 해석해주는 것으로, 이를 이용해서 특정 이름을 지정해서 볼 때 `dig` 명령어를 사용한다.

```bash
# dig dkssud8150.github.io
.
.
.
;; ANSWER SECTION:
dkssud8150.github.io.	3600	IN	A	185.199.108.153
dkssud8150.github.io.	3600	IN	A	185.199.109.153
dkssud8150.github.io.	3600	IN	A	185.199.110.153
dkssud8150.github.io.	3600	IN	A	185.199.111.153

;; Query time: 49 msec
;; SERVER: 127.0.0.53#53(127.0.0.53)





nameserver를 직접 지정가능 dig [@server] target
# dig @8.8.8.8 dkssud8150.github.io
.
.
.
;; ANSWER SECTION:
dkssud8150.github.io.	3600	IN	A	185.199.111.153
dkssud8150.github.io.	3600	IN	A	185.199.108.153
dkssud8150.github.io.	3600	IN	A	185.199.109.153
dkssud8150.github.io.	3600	IN	A	185.199.110.153

;; Query time: 41 msec
;; SERVER: 8.8.8.8#53(8.8.8.8)





# dig @1.1.1.1 dkssud8150.github.io
;; ANSWER SECTION:
dkssud8150.github.io.	3600	IN	A	185.199.108.153
dkssud8150.github.io.	3600	IN	A	185.199.109.153
dkssud8150.github.io.	3600	IN	A	185.199.110.153
dkssud8150.github.io.	3600	IN	A	185.199.111.153

;; Query time: 44 msec
;; SERVER: 1.1.1.1#53(1.1.1.1)
```

nameserver == DNS 유명한 것은 알아두자.
- Cloudflare DNS : 1.1.1.1 , ipv6= 2606:4700:4700::1111
- CISCO openDNS : 208.67.222.222 , 208.67.200.200
- google DNS : 8.8.8.8

네임서버를 지정해주는 이유는 지정하는 것마다 속도가 다르다. 그렇기에 어떤 것이 제일빠른지 비교해보고 사용하면 된다. @server를 지정해주지 않을 경우 /etc/resolv.conf에 기록되어 있는 nameserver를 사용한다.

<br>

<br>

## ethtool

간혹 시스템이 느려지는 경우 duplex 문제일 가능성이 있다. 저전력, powersave모드 사용시 발생할 수 있다.

여기서 speed와 duplex를 봐야 하는데, 시스템에서 지원하는 속도와 duplex와 다를 경우 문제가 있다. 그래서 속도와 duplex를 지정해주면 원상복귀된다.

```bash
# ethtool enp0s3lf6
supported link modes : 1000baseT/full
.
.
.
speed : 100Mb /s
Duplex : Half


# ethtool -s enp0s3lf6 speed 1000 duplex full

# ethtool enp0s3lf6
...
speed : 1000Mb/s
duplex : full
```

<br>

<br>

# ssh server

## ssh(secure shell) 개념

ssh는 통신 구간을 암호화하는 서비스로 기본적으로 리눅스 서버들은 ssh서비스가 탑재되어 있다. 없으면 설치를 추천한다. 리눅스의 ssh는 **openssh**를 사용한다.

- sshd : ssh daemon, 즉 ssh server를 의미한다.

<br>

ssh : ssh client이며, ssh 명령어가 바로 ssh client CLI이다.
- ms윈도우에서 접속하려면 putty나 mobaxterm 툴을 사용한다.

## sshd 서버 준비

1. **sshd 서버의 설치 여부 확인**
2. **sshd 서비스가 실행 중인지 확인**
    - Listen 상태인지도 확인 : ss -nlt or ss -nltp (프로세스 이름까지 보려면)
3. ssh port(22/tcp)가 방화벽에 허용되어 있는지 확인 
    - ssh port는 22를 사용한다.

<br>

### sshd 서버 설치 여부 확인

- debian

```bash
# open list openssh*
.
.
.
openssh-server/bionic-updates 1:7.6p1-4ubuntu0.6 amd64

# apt openssh-server
```

[installed]되어 있어야 함

<br>

- RH계열

```bash
# rpm -qa openssh-server
```

<br>

### sshd 서비스가 실행중인지 확인

- systemd 기반이라면 systemctl로 확인

```bash
# systemctl status sshd
● ssh.service - OpenBSD Secure Shell server
   Loaded: loaded (/lib/systemd/system/ssh.service; enabled; vendor preset: ena
   Active: inactive (dead)
```

<br>

sshd 서비스가 정지된 경우

- systemd 기반이라면 `systemctl start sshd`

```bash
# systemctl start sshd

# systemctl status sshd
● ssh.service - OpenBSD Secure Shell server
   Loaded: loaded (/lib/systemd/system/ssh.service; enabled; vendor preset: ena
   Active: active (running) since Fri 2022-02-25 01:51:51 KST; 5s ago
  Process: 4494 ExecStartPre=/usr/sbin/sshd -t (code=exited, status=0/SUCCESS)
 Main PID: 4495 (sshd)
    Tasks: 1 (limit: 2292)
   CGroup: /system.slice/ssh.service
           └─4495 /usr/sbin/sshd -D

 2월 25 01:51:51 jaeho-vm systemd[1]: Starting OpenBSD Secure Shell server...
 2월 25 01:51:51 jaeho-vm sshd[4495]: Server listening on 0.0.0.0 port 22.
 2월 25 01:51:51 jaeho-vm sshd[4495]: Server listening on :: port 22.
 2월 25 01:51:51 jaeho-vm systemd[1]: Started OpenBSD Secure Shell server.
```

<br>

- 부팅할 때 sshd 서비스가 실행되도록 하고 싶다면 `systemctl enable sshd`
    - 여기서 enable에 —now 옵션을 사용하면 start기능까지 수행됨

<br>

### listen 상태 확인

sshd는 22번 포트이므로 22번 포트 확인

```bash
# ss -nlt
State    Recv-Q    Send-Q        Local Address:Port       Peer Address:Port    
LISTEN   0         128           127.0.0.53%lo:53              0.0.0.0:*       
**LISTEN   0         128                 0.0.0.0:22              0.0.0.0:***       
LISTEN   0         5                 127.0.0.1:631             0.0.0.0:*       
**LISTEN   0         128                    [::]:22                 [::]:***       
LISTEN   0         5                     [::1]:631                [::]:*
```

<br>

### 방화벽에 허용되어 있는지 확인

```bash
# iptables -nL
Chain INPUT (policy ACCEPT)
target     prot opt source               destination         

Chain FORWARD (policy ACCEPT)
target     prot opt source               destination         

Chain OUTPUT (policy ACCEPT)
target     prot opt source               destination

ubuntu의 경우
# ufw status
상태: 비활성
# ufw enable
방화벽이 활성 상태이며 시스템이 시작할 때 사용됩니다
# ufw status
상태: 활성

# ufw allow 22/tcp
규칙이 추가되었습니다
규칙이 추가되었습니다 (v6)
# ufw allow 80/tcp
규칙이 추가되었습니다
규칙이 추가되었습니다 (v6)
```

<br>

<img src="/assets/img/dev/week2/day4/sshd.png">

accept가 허용 즉, 잘 되어 있다는 것이다. 허용되어 있지 않다면 허용시켜라. 안그럼 다 해킹당할 수 있음 

<br>

## ssh client

```bash
ssh [-p port] [username@] <host address>
```

username을 적지않을 경우 지금 사용중인 유저네임으로 자동으로 선택됨, 다른 유저명을 사용하고 싶은 경우 적어주면된다.

<br>

## ssh-keygen

ssh를 사용하다보면 보안 문제로 인해 키 기반 통신을 하는 경우가 많다. 그래서 키를 생성하는 방법은 다음과 같다.

```bash
ssh-keygen -N “”
```

- -N: passphrase - 키를 풀때마다 키를 넣어야 하는 것을 “”(공백)으로, 즉 이를 사용하지 않겠다는 것을 의미한다.

```bash
키 생성
# ssh-keygen -N ""

접속할 서버를 카피
# ssh-copy-id sunyzero@192.168.52.110

카피 한 후에는 암호를 기입하지 않고도 바로 접속 가능
# ssh sunyzero@192.168.52.110
```

<br>

ssh는 접속 뿐만 아니라 어떤 명령어를 접속한 서버에서 실행한 결과를 볼 수도 있다.

```bash
w는 앞의 서버에서 실행된 결과값을 알려주는 것이다.
# ssh -t sunyzero@192.168.52.110 w
```

<br>

<br>

# curl [options] <URL>

URL기반의 통신하는 기능

```bash
# curl https://dkssud8150.github.io
<!DOCTYPE html><html lang="en" data-mode="light" ><head><meta http-equiv="Content-Type" 
content="text/html; charset=UTF-8"><meta name="viewport" content="width=device-width, 
initial-scale=1, shrink-to-fit=no"><meta name="day-prompt" content="days ago">
...



결과를 파일로 저장하고 싶은 경우, manual.html파일로 저장
# curl -O https://www.mycompany.com/docs/manual.html



뒤에 붙이지 않고 파일명으로 직접 지정
#curl -o myblog.html https://dkssud8150.github.io
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 22068  100 22068    0     0   468k      0 --:--:-- --:--:-- --:--:--  468k
```

<br>

## curl -C - -O <URL>

파일 다운로드에도 사용할 수 있다.

-C는 다운로드 중간에 끊기더라도 계속해서 다운로드 가능

```bash
# curl -C - -O http://blah.org/blah.iso
```

<br>

API 서버로 접속해서 값들을 받아오는 것도 가능하다.

```bash
날씨 api
# curl v2.wttr.in/Seoul

환율 api
# curl https://api.exchangeratesapi.io/lastest?base=USD
```

<img src="/assets\img\dev\week2\day4\weather.png">

<br>

<br>

# wget <url>

wget과 curl은 대부분 기능이 비슷하나 curl이 더 많은 기능을 가진다. 그러나 **wget은 파일 다운로드에 특화된 기능이 존재**

```bash
# wget https://.../a.png
```

<br>

<br>

## nc (netcat)

nc: network 기능이 가능한 cat을 뜻한다.

server, client의 양쪽 기능이 가능
- 간단한 간이 서버, 클라이언트로 사용 가능
- 바이너리 통신 가능

```bash
TCP 5000
# nc -k -l 5000 

뒤의 주소와 tcp에 접속해서 hello를 보내게 됨
# echo Hello | nc 127.0.0.1 5000

그냥 바로 접속
# nc 127.0.0.1 5000
... interactive mode
```

<br>

<br>

# wireless network

## nmcli r[adio]

```bash
radio wifi [on|off]
```

무선 네트워크 기능이 활성화되어 있는지 확인하는 코드로, on,off를 통해 키거나 끄기도 가능하다.

```bash
# nmcli radio
WIFI-HW  WIFI     WWAN-HW  WWAN    
enabled  enabled  enabled  enabled


# nmcli r wifi
enabled


# nmcli r wifi on
```

만약에라도 wifi가 disabled 되어 있다면 우선적으로 blocked되어 있는지 봐야한다.

```bash
# rfkill list
...
2: phy0: wireless LAN
	Soft blocked: no
	Hard blocked: no
```

soft blocked는 `rfkill unblock`으로 해제 가능하지만, `hard(ware) blocked`인 경우는 bios나 장치 fireware 기능이 막힌 것이기에 이것들에 접근해서 찾아야 한다.

<br>

> wicd는 옛날 것이므로 사용할 수 없다. 삭제해야 한다.

<br>

## nmcli dev [list]

여기서 와이파이가 보여야 한다., list는 보통은 생략하지만, 구버전은 꼭 명시해야 하는 경우도 있다.

```bash
현재 vmware에 와이파이 커넥터가 없어서 안뜸
# nmcli dev
DEVICE  TYPE      STATE      CONNECTION 
ens33   ethernet  connected  ens33      
lo      loopback  unmanaged  --



접속 가능한 모든 wifi에 대해 보기, channel, 속도, signal 등을 알 수 있음 ,signal이 높을수록 좋음, wpa1은 보안이 취약
# nmcli dev wifi



해당 와이파이에 접속
# nmcli dev wifi connect Dev_wifi4 password qwer01234



접속되었는지 확인
# nmcli d



접속 끊기
# nmcli d disconnect wlan0
```

<br>

<br>

## wifi hotspot ( ap mode )

공유기 역할을 하는 것으로 핫스팟처럼 하는 것으로, 이를 만드는 방법은 다음과 같다.

### nmcli c

```bash
ifname은 아무거나 ssid(와이파이 스캔할 때 보여질 부분)
# nmcli c add type wifi ifname '*' con-name syhotspot autoconnect no ssid syhotspot_rpi
```

<br>

알맞게 설정하기

```bash
# nmcli c mod syhotspot 802-11-wireless.mode ap 802-11-wireless.band bg ipv4.method shared \
> ipv4.addresses 192.168.100.1/24



키 생성
# nmcli c mod syhotspot 802-11-wireless-security.key-mgmt wpa-psk 802-11-wireless-security.psk suny1234



활성화 (psk가 너무 짧으면 에러가 발생)
# nmcli c up syhotspot



확인
# nmcli c s
NAME
...
syhotspot



hotspot 정보 출력
# nmcli -p c s syhotspot



정지
# nmcli c down syhotspot
```

mode값을 ap로 고정했다. 속도에 따라서 band가 달라지는데 bg는 2.4gb에서 사용하는 band 값이다.

<br>

<br>

## hostapd

네트워크매니저보다 더 많은 기능을 사용할 수 있고, AP로 구동을 위한 기능을 제공, daemon으로 작동한다.

5GHz를 사용할 때는 그 나라마다 전파 채널, 속도 등이 다르다(regulatory domain rule). 그래서 나라를 설정해줘야 한다. 그렇지 않으면 커널을 수정해야 하기 때문에 주의해야 한다.