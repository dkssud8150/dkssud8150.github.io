---
title:    "[linux] error 모음집 및 해결 "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-03-30 00:17:00 +0800
categories: [Review, linux]
tags: [linux, error]
toc: True
comments: True
math: true
mermaid: true
# image:
#   src: /assets/img/autodriving/yolo/main.png
#   width: 800
#   height: 500
---

# sof-audio-pci ~

https://bugzilla.kernel.org/show_bug.cgi?id=205959

여기서 

```bash
$ echo "options snd-intel-dspcfg dsp_driver=1" > /etc/modprobe.d/alsa.conf
```

그래도 안되면

https://askubuntu.com/questions/1243369/sound-card-not-detected-ubuntu-20-04-sof-audio-pci

```bash
$ sudo gedit /etc/default/grub

GRUB_CMDLINE_LINUX_DEFAULT="quiet splash snd_hda_intel.dmic_detect=0"

$ sudo update-grub
$ reboot
```

<br>

# hdaudio hdaudioC0D2: Unable to bind the codec

https://ubuntuforums.org/showthread.php?t=2437409

nomodeset을 삭제한다. nvidia드라이버를 설치하면 nomodeset이 필요없기 때문이다.

```bash
$ sudo gedit /etc/default/grub

GRUB_CMDLINE_LINUX_DEFAULT="snd_hda_intel.dmic_detect=0"

$ sudo update-grub
$ reboot
```

<br>

https://ubuntuforums.org/showthread.php?t=2444070

```bash
$ pacmd list-sources | grep input

위에서 나온 name을 지정
$ pacmd set-default-source alsa_input.usb-046d_HD_Pro_Webcam_C920_76B4D93F-02.analog-stereo.2
```

<br>

그래도 안된다면 

```bash
$ journalctl -b -p err
```

출력된 에러들을 다 처리해줘야 한다.



# Nvidia 그래픽 드라이버 설치

https://phoenixnap.com/kb/install-nvidia-drivers-ubuntu#ftoc-heading-17

