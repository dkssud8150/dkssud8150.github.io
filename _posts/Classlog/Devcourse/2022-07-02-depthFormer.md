---
title:    "[데브코스] 18주차 - DepthFormer 실행 "
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
│   │   │   │   ├── 2011_09_26_drive_0001_sync
│   │   │   │   │   ├── image_02
│   │   │   │   ├── 2011_09_26_drive_0002_sync
│   │   │   │   ├── ...
│   │   │   │   ├── 2011_10_03_drive_0047_sync
│   │   │   ├── 2011_09_28
│   │   │   ├── ...
│   │   │   ├── 2011_10_03
│   │   ├── gt_depth
│   │   │   ├── 2011_09_26_drive_0001_sync
│   │   │   │   ├── image_03
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

```markdown
# data directory format
input_image: 2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png 
gt_depth:    2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png 
```


&nbsp;

&nbsp;

## KITTI 데이터셋 설치

KITTI : [http://www.cvlibs.net/datasets/kitti/eval_depth_all.php](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php)

&nbsp;

- input 데이터셋 다운로드

```bash
mkdir input
```

[http://www.cvlibs.net/datasets/kitti/raw_data.php](http://www.cvlibs.net/datasets/kitti/raw_data.php)

위의 사이트에 raw data들을 synced data 형태로 다운로드 해서 2011_09_26과 같이 날짜에 대한 폴더를 생성하고, 그 안에 데이터들을 넣어준다.

&nbsp;

&nbsp;

추가적인 데이터를 다운받으려면 `Dataset annotated depth maps data set` 을 다운로드하여 데이터를 gt_depth에 넣어준다.

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

추후 코드에서 input 데이터와 gt_depth 데이터를 비교할 때 사용하는데, input데이터를 image_02, gt_depth 데이터는 image_03 데이터를 사용하므로 gt_depth를 다운하고 그 데이터를 분할해줘도 된다. 위의 gt_depth를 다운받아보고, 자신의 kitti_eigen_test.txt 파일에 맞게 분할해주면 된다. 나의 경우 txt파일에 0002_sync, 0013,0020에 대한 gt 파일이 지정되어 있었는데, gt_depth에는 해당 번호들이 없어서 input데이터에서 복사해왔다.

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

해당 파일을 다운받고, data/kitti/폴더에 넣는다. 그리고 자기가 가지고 있는 데이터에 맞게 파일을 수정해준다. 나는 02,09,13,20 에 대해서만 사용할 예정이므로 나머지는 다 지웠다.

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
python ./tools/test.py ./configs/depthformer/depthformer_swinl_22k_w7_kitti.py ./checkpoints/depthformer_swinl_22k_kitti.pth --show-dir depthformer_swinl_22k_w7_kitti_result
```

show-dir은 test 결과를 해당 디렉토리에 저장하겠다는 의미이다. 

<img src="/assets/img/dev/result_directory.png">

<img src="/assets/img/dev/result.png">

&nbsp;

- no gpus

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_kitti.py checkpoints/depthformer_swinl_22k_kitti.pth --show
```

&nbsp;

&nbsp;

# Bug report

1. TypeError: multi_scale_deformable_attn_pytorch() takes 4 positional arguments but 6 were given

`output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step)`

change to

`output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)`

[https://github.com/open-mmlab/mmdetection/issues/6261](https://github.com/open-mmlab/mmdetection/issues/6261)

&nbsp;

&nbsp;

2. please usr “init_cfg” instead

[https://github.com/open-mmlab/mmdetection/issues/5177](https://github.com/open-mmlab/mmdetection/issues/5177)

&nbsp;

&nbsp;

3. CUDA out of memory

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/depthformer/depthformer_swinl_22k_w7_nyu.py path/to/your/model --eval x
```

로 실행

[https://discuss.pytorch.kr/t/cuda-out-of-memory/216/6](https://discuss.pytorch.kr/t/cuda-out-of-memory/216/6)

&nbsp;

&nbsp;

4. cuDNN error: CUDNN_STATUS_NOT_INITIALIZED

[https://github.com/werner-duvaud/muzero-general/issues/139](https://github.com/werner-duvaud/muzero-general/issues/139)

torch==1.8.0+cu112 하지 않고, torch==1.8.0 만 설치하고, 실행했더니 해당 에러가 뜬다.

[https://github.com/pytorch/pytorch/issues/53336](https://github.com/pytorch/pytorch/issues/53336)

&nbsp;

- cuda버전에 맞는 pytorch 설치
    - torch==1.8.0+cu<cuda version>
    - 11.2에 맞는 torch버전이 없어서 11.3을 설치한다.

&nbsp;

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

그러나 이 에러가 1.8이상 버전에서 발생하는 에러일 수도 있음. 그래서 11.3으로 설치해도 에러가 난다면 11.0으로 설치한다.

```python
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

> 둘 다 동일하게 에러 발생


```bash
ImportError: /usr/local/lib/python3.7/dist-packages/mmcv/_ext.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZN2at5sliceERKNS_6TensorElN3c108optionalIlEES5_l
```

이는 pytorch버전이랑 mmcv 버전이 안맞아서 발생하는 것!

&nbsp;

- 11.0버전

```bash
!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
```

> 동일하게 1.5.3 버전이 설치된다.

&nbsp;

- 11.3버전

```bash
!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

> mmcv-full 1.5.3 버전을 설치하니 버전이 너무 안맞다고 함… 결국 깃허브에 나와있는 버전대로 11.1버전에 맞게 설치하고자 한다.

&nbsp;

&nbsp;

원래 설명서에 나와있는 버전은 cuda 11.1, torch 1.8.0 mmcv 1.3.13

```bash
pip uninstall torch torchvision torchaudio mmcv-full -y
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
```

&nbsp;

&nbsp;

5. import 상대경로 설정

```python
import sys
sys.path.append(path)
```

&nbsp;

&nbsp;

6. qt.qpa.xcb: could not connect to display

QT라는 GUI 라이브러리가 있는데, 이를 실행할 수 없다는 것. colab에서는 실행이 불가능하다고 한다.

[qt.qpa.xcb: could not connect to display :1.0 qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.](https://forum.qt.io/topic/132855/qt-qpa-xcb-could-not-connect-to-display-1-0-qt-qpa-plugin-could-not-load-the-qt-platform-plugin-xcb-in-even-though-it-was-found/13)

&nbsp;

- colab에서 cv2 함수 사용하는 방법

import cv2 → `from google.colab.patches import cv2_imshow`

[COLAB에서 OpenCV 함수 사용하기](https://overface.tistory.com/573)

> cv2가 아닌 mmcv를 사용하고 있어서 어디를 고쳐야 할지 모름. 실패

&nbsp;

- cv2를 pillow로 변경

mmcv.imfrombytes(backend=’cv2’) → mmcv.imfrombytes(backend=’pillow’)

[fileio - mmcv 1.5.3 documentation](https://mmcv.readthedocs.io/en/latest/api.html)

> 실패

&nbsp;

| 결론 : —show를 `—eval x`로 변경하여 GUI 사용하지 않기

&nbsp;

- 추가 : Xming 설치해서 WSL2에서 GUI 실행하기

[[WSL] Windows Subsystem for Linux - 디스플레이 서버 설정 및 GUI 사용하기 - ECE - TUWLAB](https://www.tuwlab.com/ece/29485)

1. xming 설치 : [https://sourceforge.net/projects/xming/](https://sourceforge.net/projects/xming/)
2. 실행 → 단축 아이콘 부분에 실행된 것을 확인할 수 있음
3. wsl 쉘에 /etc/machine-id 생성
    
    ```bash
    sudo systemd-machine-id-setup
    sudo dbus-uuidgen --ensure
    ```
    
4. 생성 확인
    
    ```bash
    $ cat /etc/machine-id
    7d7873cb6d4a46f1adcf99fba8b07d5a
    ```
    
5. X-window 구성 요소 설치
    
    ```bash
    sudo apt install x11-apps xfonts-base xfonts-100dpi xfonts-75dpi xfonts-cyrillic
    ```
    
6. ~/.bashrc에 추가
    
    ```bash
    export DISPLAY=:0
    ```
    
7. 적용
    
    ```bash
    source ~/.bashrc
    ```
    
8. 동작확인 → wsl쉘에 타이핑
    
    ```bash
    xeyes
    ```
    

&nbsp;

7. TypeError: multi_scale_deformable_attn_pytorch() takes 4 positional arguments but 6 were given

4개 args만 들어가야 하는데 6개 들어간다는 의미이다. 코드를 보니 실제로 6개가 들어가고 있었다.

[https://github.com/open-mmlab/mmcv/pull/1223](https://github.com/open-mmlab/mmcv/pull/1223)

[](https://programtalk.com/python-more-examples/mmcv.ops.multi_scale_deform_attn.multi_scale_deformable_attn_pytorch.detach/)

```python
# usr/local/lib/python3.7/dist-packages/mmcv/ops/multi_scale_deform_attn.py line.351
output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations,
                attention_weights)
```

&nbsp;

8. tcmalloc: large alloc

python이 파일이 너무 큰 파일을 다룰 때, 나오는 경고 메시지라고 한다. 일반적으로는 warning으로만 출력되지만, 실제로 할당 메모리보다 자원이 부족한 경우에는 에러가 난다. 해결하는 방법은 환경 변수 `export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD`를 더 크게 변경하면 된다.  약 10GB로 지정해준다.

```bash
!sudo sh -c 'echo "export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=107374182400" >> ~/.bashrc'
!source ~/.bashrc
```

[[Python / Linux] tcmalloc: large alloc 2219589632 bytes == 0x8fa7a000 @ 0x7fb7e4433680 해결법](https://cryptosalamander.tistory.com/151)

&nbsp;

9. AttributeError: 'ConfigDict' object has no attribute 'eval_pipeline’

`./configs/_base_/datasets/kitti.py`에`./configs/_base_/datasets/nyu.py`에 있는 eval_pipeline을 복사하여 추가한다.

[https://github.com/open-mmlab/mmdetection/issues/5739](https://github.com/open-mmlab/mmdetection/issues/5739)

&nbsp;

10. KeyError: 'cam_intrinsic’

`./depthdatasets/kitti.py`에 있는 cam_instrinsic_dict를 cam_intrinsic으로 수정한다.