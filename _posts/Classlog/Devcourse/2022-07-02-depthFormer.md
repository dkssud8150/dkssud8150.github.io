---
title:    "[데브코스] 15주차 - DepthFormer 실행 "
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-07-02 22:20:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, DepthFormer]
toc: true
comments: true
math: true
---

&nbsp;

# Miniconda 설치 방법

설치 사이트 : [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

참고 사이트 : [https://digiconfactory.tistory.com/entry/미니콘다Miniconda-설치하기-파이썬-패키지관리](https://digiconfactory.tistory.com/entry/%EB%AF%B8%EB%8B%88%EC%BD%98%EB%8B%A4Miniconda-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%8C%A8%ED%82%A4%EC%A7%80%EA%B4%80%EB%A6%AC)

&nbsp;

python버전 별로 설치하거나 최신 버전을 다운받는다.

설치 exe파일을 열면 2개의 체크박스가 나오는데, 첫번째는 시스템 환경 변수에 추가할 건지에 대한 것이고, 2번째는 지금 설치하는 python을 시스템 전체 default로 설정할 것인지이다.

&nbsp;

&nbsp;

## WSL2 or Ubuntu miniconda install

참고 사이트 : [https://smoothiecoding.kr/미니콘다-wsl2-vsc-파이썬/](https://smoothiecoding.kr/%EB%AF%B8%EB%8B%88%EC%BD%98%EB%8B%A4-wsl2-vsc-%ED%8C%8C%EC%9D%B4%EC%8D%AC/)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

1. `ENTER`를 누르고, 사용 설명서가 뜨고, yes를 입력해야 설치가 진행된다.
2. 그러면 설치 경로를 변경하는 창이 뜬다. default값인 `/usr/bin/miniconda3` 로 정해준다. default값으로 사용하려면 그냥 ENTER를 누르면 된다.
3. `conda init`을 할거냐는 창이 뜨면, yes라고 친다. 그러면 설치 성공된다.

최종적으로 conda를 적용하기 위해 다음 명령어를 친다.

```bash
source ~/.bashrc
```

&nbsp;

&nbsp;

- 가상환경 interpreter

vscode에서 코드를 디버깅하기 위해서, 가상환경의 python을 인터프리터로 사용해야 가상환경 내에서 코드를 돌릴 수 있다. 그렇지 않으면 local에 있는 인터프리터로 잡혀서 가상환경 안에 깔았던 패키지들이 인식되지 않는다.

&nbsp;

그래서 가상환경에 존재하는 python으로 인터프리터를 바꿔줘야 한다.

1. 먼저 `CTRL+SHIFT+P`를 눌러 명령창을 켠다.
2. `python: select interpreter`를 선택한다.
3. 가상환경안에 존재하는 python을 선택한다. conda를 사용했다면, 자신이 설치한 anaconda의 경로를 알아야 한다. 보통은 `/usr/bin/anaconda3` 또는 `/home/<username>/anaconda3` 등에 있을 것이다. 그러면 그 안에 존재하는 인터프리터의 경로는 `/usr/bin/anaconda3/envs/<envname>/bin/python.exe`가 된다.

&nbsp;

&nbsp;

# 가상환경 구축

참고 사이트 : https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/docs/get_started.md#installation

&nbsp;

## Conda 사용하여 환경 구축

- 가상 환경 생성

```bash
conda create -n mde python=3.7
conda activate mde
```

&nbsp;

- install pytorch

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

&nbsp;

- MMCV 설치 및 toolbox 설치

```bash
pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

git clone https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox.git
cd Monocular-Depth-Estimation-Toolbox
pip install -e .
```

&nbsp;

- 추가 패키지 설치

```bash
pip install future tensorboard
```

&nbsp;

## Virtualenv로 가상환경 구축

anaconda를 설치하지 않았다면, 다른 패키지로 가상환경을 구축해야 한다. (그러나 이렇게 하면 conda에 대한 에러가 많이 떠서 비추천)

- virtualenv 패키지 설치

```bash
pip install virtualenv
```

&nbsp;

- 가상환경 구축

```bash
virtualenv MDE --python=python3.7
source MDE/bin/activate
```

ubuntu인 경우 bin폴더에 activate가 있을 것이고, window인 경우 bin대신 `Scripts` 에 있다.

&nbsp;

- pytorch, cudatoolkit 설치

cuda 11.6 version 설치

```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda-repo-wsl-ubuntu-11-6-local_11.6.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-6-local_11.6.2-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-6-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

&nbsp;

- conda대신 virtualenv를 사용하게 되면 train.sh 실행 시 바꿔줘야 할 부분이 있다.

[https://pytorch.org/docs/stable/elastic/run.html](https://pytorch.org/docs/stable/elastic/run.html)

```bash
$ python -m torch.distributed.launch ...
```

에서 `python -m torch.distributed.launch`대신 `torchrun`으로 변경한다.

```bash
$ torchrun ...
```

&nbsp;

- 가상환경 나가기

```bash
deactivate
```

&nbsp;

&nbsp;

# 데이터셋 다운로드

## 데이터셋 디렉토리 구조

```
monocular-depth-estimation-toolbox
├── depth
├── tools
├── configs
├── splits
├── data
│   ├── kitti
│   │   ├── input
│   │   │   ├── 2011_09_26
│   │   │   ├── 2011_09_28
│   │   │   ├── ...
│   │   │   ├── 2011_10_03
│   │   ├── gt_depth
│   │   │   ├── 2011_09_26_drive_0001_sync
│   │   │   ├── 2011_09_26_drive_0002_sync
│   │   │   ├── ...
│   │   │   ├── 2011_10_03_drive_0047_sync
|   |   ├── benchmark_test
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
│   │   │   ├── ...
│   │   │   ├── 0000000499.png
|   |   ├── benchmark_cam
│   │   │   ├── 0000000000.txt
│   │   │   ├── 0000000001.txt
│   │   │   ├── ...
│   │   │   ├── 0000000499.txt
│   │   ├── split_file.txt
│   ├── nyu
│   │   ├── basement_0001a
│   │   ├── basement_0001b
│   │   ├── ... (all scene names)
│   │   ├── split_file.txt
│   ├── SUNRGBD
│   │   ├── SUNRGBD
│   │   │   ├── kv1
│   │   │   ├── kv2
│   │   │   ├── realsense
│   │   │   ├── xtion
│   │   ├── split_file.txt
│   ├── cityscapes
│   │   ├── camera
│   │   │   ├── test
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── disparity_trainvaltest
│   │   │   ├── disparity
│   │   ├── leftImg8bit_trainvaltest
│   │   │   ├── leftImg8bit
│   │   ├── split_file.txt
```

&nbsp;

&nbsp;

## KITTI 데이터셋 설치

KITTI : [http://www.cvlibs.net/datasets/kitti/eval_depth_all.php](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php)

`Dataset annotated depth maps data set` 을 다운로드하여 데이터를 gt_depth에 넣어준다.

- curl로 gt_depth 데이터셋 다운로드

```bash
cd Monocular-Depth-Estimation-Toolbox
mkdir data
cd data

curl https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip -o ./kitti.zip
jar xvf ./kitti.zip

mkdir gt_depth
mv ./train/* ./gt_depth/
```

&nbsp;

- input 데이터셋 다운로드

```bash
mkdir input
```

[http://www.cvlibs.net/datasets/kitti/raw_data.php](http://www.cvlibs.net/datasets/kitti/raw_data.php)

위의 사이트에 raw data들을 synced data 형태로 다운로드 해서 input 폴더에 넣는다. 

&nbsp;

- benchmark_cam 다운로드

[http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)

vaildation and test set을 다운받고, prediction 폴더만 data폴더에 옮긴다.

```bash
data
	- kitti
		- test_depth_prediction_anonymous/intrinsics -> benchmark_cam
		- test_depth_prediction_anonymous/image -> benchmark_test
```

&nbsp;

- split_file download

[https://www.kaggle.com/datasets/qikangdeng/kitti-split-and-eigen-split?resource=download&select=eigen_train_files.txt](https://www.kaggle.com/datasets/qikangdeng/kitti-split-and-eigen-split?resource=download&select=eigen_train_files.txt)

&nbsp;

&nbsp;


## NYU 데이터셋 다운로드

NYU : [https://drive.google.com/u/0/uc?id=1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP&export=download](https://drive.google.com/u/0/uc?id=1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP&export=download)

```bash
cd data
mkdir nyu
```

nyu 폴더에 위의 zip파일을 다운로드한다.

```bash
unzip sync.zip
```

또는 

```bash
jar xvf sync.zip
```

그 후 test 데이터셋도 다운받아야 한다.

해당 주소 ([https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html))로 들어가 `labeled dataset`을 다운받는다.

이는 `.mat` 파일이므로 zip파일로 변환해야 한다. 변환하는 사이트는 다음과 같다. [https://m.ezyzip.com/kr-mat-zip.html](https://m.ezyzip.com/kr-mat-zip.html)

&nbsp;

&nbsp;

## SUN RGB-D 데이터셋 다운로드

SUN RGB-D : [https://rgbd.cs.princeton.edu/](https://rgbd.cs.princeton.edu/)

해당 사이트로 들어가 Data → SUNRGBD V1을 다운로드한다.

```bash
curl https://rgbd.cs.princeton.edu/data/SUNRGBD.zip -o sun_rgbd.zip
unzip sun_rgbd.zip

mkdir SUNRGBD
```

&nbsp;

&nbsp;

## Cityscapes 데이터셋 다운로드

cityscapes : [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/)

해당 사이트로 들어가 *[leftImg8bit_trainvaltest.zip (11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=3)*를 다운받는다.

```bash
mkdir cityscapes
```

&nbsp;

&nbsp;

# 실행

## pth 파일 다운로드

[https://drive.google.com/u/0/uc?id=1BpcY9tULBRTW-cG8EVHBAZBapAv3ide4&export=download](https://drive.google.com/u/0/uc?id=1BpcY9tULBRTW-cG8EVHBAZBapAv3ide4&export=download)

```bash
mkdir checkpoints
```

checkpoints안에 pth파일을 추가해준다.

&nbsp;

## DepthFormer Train

```bash
bash ./tools/dist_train.sh configs/depthformer/depthformer_swinl_22k_w7_kitti.py 1 --work-dir work_dirs/saves/depthformer/depthformer_swinl_22k_w7_kitti
```

- sh [./tools/dist_train.sh] ${CONFIG_FILE} ${GPU_NUM} —work_dir [work_dir]

&nbsp;

&nbsp;

## DepthFormer Test

- single gpus

```bash
python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_kitti.py checkpoints/depthformer_swinl_22k_kitti.pth --show
```

- no gpus

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_kitti.py checkpoints/depthformer_swinl_22k_kitti.pth --show
```

&nbsp;

&nbsp;

# Bug report

- TypeError: multi_scale_deformable_attn_pytorch() takes 4 positional arguments but 6 were given

`output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step)`

change to

`output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)`

[https://github.com/open-mmlab/mmdetection/issues/6261](https://github.com/open-mmlab/mmdetection/issues/6261)

&nbsp;

&nbsp;

- subprocess.CalledProcessError: Command '['/usr/local/bin/python', '-u', 'tools/train.py', '--local_rank=0', 'configs/depthformer/depthformer_swint_w7_nyu.py', '--launcher', 'pytorch']' returned non-zero exit status 1.

`./configs/depthformer/depthformer_swinl_22k_w7_kitti.py`에서 model - pretrained 부분을 수정

[https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/issues/21](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/issues/21)

[https://github.com/open-mmlab/mmdetection/issues/5177](https://github.com/open-mmlab/mmdetection/issues/5177)

&nbsp;

&nbsp;

- please usr “init_cfg” instead

&nbsp;

&nbsp;

- CUDA out of memory

```bash
$ CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_nyu.py path/to/your/model --eval x
```

로 실행한다.

&nbsp;

또는 다음 사이트에 들어가서 하나하나 실행해본다.

[https://discuss.pytorch.kr/t/cuda-out-of-memory/216/6](https://discuss.pytorch.kr/t/cuda-out-of-memory/216/6)