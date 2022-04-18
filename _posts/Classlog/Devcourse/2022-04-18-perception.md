---
title:    "[데브코스] 7주차 - DeepLearning Perception"
author:
  name: JaeHo YooN
  link: https://github.com/dkssud8150
date: 2022-04-18 16:40:00 +0800
categories: [Classlog, devcourse]
tags: [devcourse, deeplearning]
toc: true
comments: true
math: true
#image:
#  src: /assets/img/dev/week7/day4/main.png
#  width: 500
#  height: 500
---

<br>

# 환경 설정

코딩에 앞서 환경을 설정해주어야 한다. anaconda를 통해 가상환경을 설정할 수 있지만, 컴퓨터의 용량 한계로 인해 anaconda를 사용하지 않고도 가상환경을 구축할 수 있는 방법을 소개한다.

## 가상환경 설정 (use virtualenv)

### 설치 및 가상환경 생성

```bash
pip install virtualenv

virtualenv dev --python=python3.8
```

이 때 3.8 interpreter가 없다고 뜬다면, 3.8 python이 안깔려 있는 것이므로 3.8버전을 깔거나 다른 버전으로 실행한다.

<br>

### 가상환경 활성화

```bash
source dev/Scripts/activate
```

<br>

### 필요한 패키지 설치

```bash
pip install numpy
```

<br>

### 가상환경 나가기

```bash
deactivate
```

<br>

### 자신이 설치한 패키지를 저장하기

```bash
pip freeze > requirements.txt
```

<br>

### 다시 설치

```bash
pip install -r requirements.txt
```

<br>

<br>

# tensor 기초

```python
import torch
import numpy as np
```

## tensor 만들기

```python
def make_tensor():
    # int
    a = torch.tensor([[1, 2],[3, 4]], dtype=torch.int16)
    # float
    b = torch.tensor([1.0], dtype=torch.float32)
    # double
    c = torch.tensor([3], dtype=torch.float64, device="cuda:0")
    print(a.dtype, b.dtype, c.dtype, c.device)
```

`torch.tensor(data, dtype, device)` 의 형태로 선언해준다. data는 `[]`를 통해 작성해야 하고, dtype의 경우 float64로 하게 되면 double형태로 만들어진다. 그 뒤에 device로 gpu를 사용할지 cpu를 사용할지 지정해준다. 

변수의 type을 보려면 `type(a)`와 `a.dtype`이 있는데, 전자의 경우 전체 변수의 type을 보는 것이고, dtype의 경우 그 안에 들어있는 데이터 타입을 확인하는 메서드다.

<br>

## tensor 더하기/빼기

```python
def sumsub_tensor():
    a = torch.tensor([3,2])
    b = torch.tensor([5,3])
    
    print("input : {}, {}\nsum : {}\tsub : {}".format(a,b, a+b, a-b))

    # each element sum
    sum_element_a = a.sum()
    print(sum_element_a)

# ------------------- #

input : tensor([3, 2]), tensor([5, 3])
sum : tensor([8, 5])    sub : tensor([-2, -1])
tensor(5)
```

생성된 tensor을 행렬로 더할 때는 `a+b`,`a-b`를 통해 진행하고, 각각의 요소를 더할 때는 `sum()` 메서드를 사용한다.

<br>

## tensor 곱하기/나누기

```python
def muldiv_tensor():
    a = torch.arange(0,9,1).view(3,3)
    b = torch.arange(0,9,1).view(3,3)

    # mat mul
    c = torch.matmul(a,b)
    print("input : {} \n{}\n\n mul : {}".format(a,b,c))

    # elementwise multiplication
    d = torch.mul(a,b)
    print(d)

# --------------- # 

input : tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
        tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])

mul : tensor([[ 15,  18,  21],
        [ 42,  54,  66],
        [ 69,  90, 111]])
      tensor([[ 0,  1,  4],
        [ 9, 16, 25],
        [36, 49, 64]])
```

`view`는 차원을 바꿔주는 함수이다. 즉 view(3,3)을 하게 되면 3x3 행렬로 만들어준다. 그 후 `matmul`을 통해 행렬 곱을 진행한다. 단지 요소 별 곱셈을 하고 싶다면 `mul`을 사용한다.

<br>

## tensor 차원 바꾸기

```python
def reshape_tensor():
    a = torch.tensor([2,4,5,6,7,8])
    # view
    b = a.view(2,3)

    # transpose
    b_t = b.t()
    print("input : {} ,\t transpose : {}".format(b, b_t))

# -------------- #


```

<br>

## tensor 접근

```python
def access_tensor():
    a = torch.arange(1,13).view(4,3)
    
    # first row (slicing)
    print(a[:,0])

    # first col
    print(a[0,:])
```

<br>

## transform to numpy

```python
def transform_numpy():
    a = torch.arange(1,13).view(4,3)

    # array to numpy
    a_np = a.numpy()
    print(a_np)

    # tensor to numpy
    b = np.array([1,2,3])
    b_ts = torch.from_numpy(b)
    print(b_ts)
```

<br>

## tensor 결합

```python
# 두 dimension이 동일해야 한다.
def concat_tensor():
    a = torch.arange(1,10).view(3,3)
    b = torch.arange(10,19).view(3,3)
    c = torch.arange(19,28).view(3,3)

    abc = torch.cat([a,b,c],dim=1)
    print(abc)

    abc2 = torch.cat([a,b,c], dim=0)
    print(abc2)

    print(abc.shape, abc2.shape)
```

<br>

## tensor 결합 2

```python
# 새로운 dimension을 만들면서 결합, 위는 2d -> 2d, 현재는 2d -> 3d
def stack_tensor():
    a = torch.arange(1,10).view(3,3)
    b = torch.arange(10,19).view(3,3)
    c = torch.arange(19,28).view(3,3)

    abc = torch.stack([a,b,c], dim=0) # 0index = a, 1index = b, 2index = c
    abc2 = torch.stack([a,b,c], dim=1) # 0index = a[0,:]+b[0,:]+c[0,:]
    abc3 = torch.stack([a,b,c], dim=2) # 0index = a[:,0]+b[:,0]+c[:,0]
    print(abc, '\n', abc2, "\n", abc3)
    print(abc.shape, abc2.shape)

```

<br>

## tensor transpose

```python
def transpose_tensor():
    a = torch.arange(1,10).view(3,3)
    at = torch.transpose(a,0,1)
    print(a,"\n",at,"\n\n")

    b = torch.arange(1,25).view(4,3,2)
    bt = torch.transpose(b, 0, 2)
    print(b,"\n", bt)

    # 여러 가지 dimension을 바꿀 떄
    bp = b.permute(2,0,1) # 0,1,2 -> 2,0,1
    print(b.shape, bp.shape)

```

<br>

```python
if __name__ == '__main__':
    #make_tensor()
    #sumsub_tensor()
    #muldiv_tensor()
    #reshape_tensor()
    #access_tensor()
    #transform_numpy()
    #concat_tensor()
    #stack_tensor()
    #transpose_tensor()
```

<br>

<br>

# image classification using the MNIST Benchmarks

```markdown
dev
  ⊢ datasets
  ⊢ dev
  ⊢ loss
    ⊢ \__init__.py
    ∟ loss.py
  ⊢ model
    ⊢ \__init__.py
    ⊢ lenet5.py
    ∟ models.py
  ⊢ util
    ⊢ \__init__.py
    ∟ tools.py
  ⊢ img_classify.py
  ∟ requirements.txt
```

`datasets`폴더에는 데이터셋을 저장하고, dev는 가상환경을 위한 폴더이다. loss는 loss function을 선언해주는 파일을 담기위한 폴더이고, model에는 모델들을 저장해놓은 폴더이다. 정의해놓은 모델을들 models.py 파일에서 불러온다. utils는 예측 결과물을 보기 위해 만든 폴더이다. tools.py에는 PIL의 ImageDraw 함수를 통해 결과물을 본다.

실제 실행하는 파일은 img_classify.py 이고, 필요한 패키지를 저장해놓은 파일이 requirements.txt이다.

<br>

## import

```python
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

import os
import argparse
import sys
from tqdm.notebook import trange

from model.models import *
from loss.loss import *
from util.tools import *
```

<br>

## parser

```python
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="MNIST")
    parser.add_argument("--mode", dest="mode", help="train or valid or test", 
                        default=None, type=str)
    parser.add_argument("--download", dest="download", help="dataset download", 
                        default=False, type=bool)
    parser.add_argument("--odir", dest="output_dir", help="output directory for train result",
                        default="./output", type=str)
    parser.add_argument("--checkpoint",dest="checkpoint", help="checkpoint for trained model file", 
                        default=None, type=str)
    parser.add_argument("--device",dest="device", help="use device cpu / gpu",
                        default="cpu", type=str)

    if len(sys.argv) == 1: # python main.py 하나만 했다는 뜻
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()
    return args
```

터미널에서 실행할 때 매 번 인자들을 바꾸기 힘들기 때문에, 지정을 해줄 수 있도록 argument를 지정해준다.

<br>

## get dataset

```python
def get_data():
    download_root = "./datasets"

    transform = {
        "train" : transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(1.0,))
        ]),
        "test" : transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(1.0,))
        ])
    }
    # MNIST(root, train=false,transform,...,download )
    train_dataset = MNIST(root=download_root, 
                            transform=transform["train"], 
                            train=True,
                            download=args.download)
    valid_dataset = MNIST(root=download_root, 
                            transform=transform["test"], 
                            train=False,
                            download=args.download)
    test_dataset = MNIST(root=download_root, 
                            transform=transform["test"], 
                            train=False,
                            download=args.download)
    
    return train_dataset, valid_dataset, test_dataset
```

pytorch 내에 데이터셋이 MNIST, ImageNet, caltech 등이 있다. 이를 다운로드하고, transform시켜주는 함수이다.

<br>

## main

- model output directory 생성, 사용할 device 지정

```python
def main():
    #print(torch.__version__)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir + "/model_epoch", exist_ok=True)
    
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            print("gpu")
            device = torch.device("cuda")
        else:
            print("can not use gpu!")
            device = torch.device("cpu")
```

<br>

- dataset, dataloader

불러온 데이터셋을 dataloader

```python
    train_dataset, valid_dataset, test_dataset = get_data()

    # Make dataloader
    # image dataset과 annotation을 쉽게 넣어주기 위해 준비된 타입으로 만듦
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=8,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=True,
                                    )

    valid_dataloader = DataLoader(valid_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=False,
                                    )

    test_dataloader = DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=False,
                                    )
```

`DataLoader`에 인자로 다양한 것들이 존재한다.
- batch_size : batch 크기 지정
- shuffle : 데이터의 순서를 섞을지 지정
- num_workers : 데이터 로더를 만들어주는 일꾼과 같이 CUDA를 사용할 때 학습과 추론에서 cpu와 gpu 사이에서 동기화를 하면서 전송될 때, 몇개를 쓰냐에 따라 성능이 차이난다. 대체로 cpu 코어수의 절반으로 지정
- pin memory : true일 경우, 커스텀 타입의 데이터 요소들을 사용할 때 이를 리턴하기 전에 cuda pinned memory로부터 텐서를 복사한다.
- drop_last : batch size를 8로 사용하는데, 마지막에 4장이 남는다면 이걸로 batch로 만들 수 없는데, 이를 사용할지 버릴지에 대해 지정

<br>


- model 불러오기

model폴더에서 선언해둔 함수를 통해 모델 불러오기

```python
_model = get_model("lenet5")
```

<br>

- train에 대한 과정

```python
    # MNIST 데이터 : [1, H, W]
    if args.mode == "train":
        model = _model(batch=8, n_classes=10, in_channel=1, in_width=32, in_height=32, is_train=True)
        model.to(device)
        model.train()
        
        # optimizer & scheduler
        optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        criterion = get_criterion(crit="mnist", device=device)

        epochs = 10

        for epoch in trange(epochs):
            total_loss = 0
            best_loss = 5
            best_epoch = 0
            
            for i, (image, label) in enumerate(train_dataloader):
                image = image.to(device)
                label = label.to(device)

                output = model(image)
                #print(output)

                loss_val = criterion(output, label)
                #print(loss_val)

                # backpropagation
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss_val.item()

                # if i % 1000 == 0:
                #     print("{} epoch {} iter loss : {}".format(epoch, i, loss_val.item()))

            total_loss = total_loss / i
            scheduler.step()

            print("->{} epoch mean loss : {}".format(epoch, total_loss))

            if total_loss < best_loss:
                best_loss = total_loss
                best_epoch = epoch

        torch.save(model.state_dict(), args.output_dir + "/model_epoch/"+"best_epoch.pt")
```

train의 경우
- batch size : 8로 설정 
- n_classes : 분류할 클래스의 수는 10
- in_channel : MNIST 데이터의 경우 [1,H,W]로 이루어진 흑백 이미지이므로 1
- in_width,in_height : MNIST 데이터의 경우 크기가 32x32이므로 32

- optimizer : 최적화에 사용될 optimizer 알고리즘은 Stochastic Gradient Descent 를 사용하고, 처음 learning rate를 지정한다. 또, SGD에서 사용될 얼마나 lr을 갱신할 지에 대한 momentum도 지정해준다.
- scheduler : 처음에는 0.01로 시작하지만, 너무 큰 lr이면 학습이 잘 안될 수 있으므로 줄여주기 위한 용도이다. steplr은 단계적으로 lr을 갱신하는 방법으로 step size 마다 gamma 값만큼 줄인다.

- get_criterion : 지정해둔 mnist에 대한 손실함수를 불러와 저장한다.

- epoch : 얼마나 반복할지에 대한 값이므로 알아서 설정하면 된다.
- trange : tqdm에서 지원하는 메서드로 `tqdm(range())`와 같다. 루프의 진행 정도를 볼 수 있다.

- to(device) : gpu를 사용하여 학습할 것이므로 `to(device)`를 사용하여 cuda.tensor로 변경시킨다. 그 후 모델에 집어넣어 결과를 얻는다. 얻어진 결과를 바탕으로 loss를 구하고, total_loss에 더한다. 다 더해진 total_loss를 batch_size로 나누어준다. 한 epoch이 끝나면 scheduler를 진행시킨다.

- best_loss, torch.save : 가장 낮은 loss를 얻은 epoch을 저장해놓고 나중에 모델로 save한다.

<br>

- eval 과정

```python
    elif args.mode == "eval":
        model = _model(batch=1, n_classes=10, in_channel=1, in_width=32, in_height=32)

        # load trained model
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        acc = 0
        num_eval = 0

        with torch.no_grad():
            for i, (image, label) in enumerate(valid_dataloader):
                image = image.to(device)
                output = model(image)
                output = output.cpu()

                if output == label:
                    acc += 1
                num_eval += 1

            print("evaluation score : {} / {}".format(acc, num_eval))
```

- batch : train처럼 batch_size 단위로 학습시키는 것이 아닌 이미지 1개씩 평가를 진행해야 하므로 batch를 1로 설정
- checkpoint : 저장해둔 모델을 불러온다.
- model.load_state_dict : 모델의 파라미터들을 불러와서 평가를 진행

정확도를 얻기 위해 acc과 num_eval을 구하여 출력한다.

<br>

- test 과정

```python
    elif args.mode == "test":
        model = _model(batch=1, n_classes=10, in_channel=1, in_width=1, in_height=1)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        with torch.no_grad():
            for i, (image, label) in enumerate(test_dataloader):
                image = image.to(device)
                output = model(image)
                output = output.cpu()
                #print(output)

                # show result
                show_img(image.detach().cpu().numpy(),str(output.item()))
```

eval과정과 동일하나, 결과물을 얻기 위해 show_img를 사용하여 image를 본다. 이때, numpy로 변환시키기 위해 gradient 연산과 분리시키고, cuda연산을 했던 결과를 cpu로 변환하고, numpy로 변환한다.

<br>

## 실행

```python
if __name__ == "__main__":
    args = parse_args()
    main()
```

<br>

## loss/loss.py

```python
import torch
import torch.nn as nn
import sys

class MNISTloss(nn.Module):
    def __init__(self, device = torch.device('cpu')):
        super(MNISTloss, self).__init__()
        self.loss = nn.CrossEntropyLoss().to(device)

    def forward(self, output, label):
        loss_val = self.loss(output, label)

        return loss_val

def get_criterion(crit = "mnist", device = torch.device('cpu')):
    if crit is "mnist":
        return MNISTloss(device = device)
    else:
        print("unknown criterion")
        sys.exit(1)
```

MNISTloss : MNIST에 대한 loss를 선언해둔다. crossentropyloss를 사용할 것이고, loss를 연산하여 리턴한다. 

get_criterion : 선언해둔 MNISTloss를 불러와 mnist라는 인자를 받을 경우 MNISTloss를 반환하고, mnsit가 아닐 경우 일단은 없다고 출력하고 끝낸다.

<br>

## model

### lenet5.py

lenet5에 대한 레이어들을 선언해둔 파일이다.

```python
import torch
import torch.nn as nn

# in == input
class Lenet5(nn.Module):
    def __init__(self, batch, n_classes, in_channel, in_width, in_height, is_train = False):
        super().__init__()
        self.batch = batch
        self.n_classes = n_classes
        self.in_channel = in_channel
        self.in_width = in_width
        self.in_height = in_height
        self.is_train = is_train

        # layer define
        # convoluation output size : [(W - K + 2P)/S] + 1
        # w : input size, k : kernel size, p : padding size, s : stride
        self.conv0 = nn.Conv2d(self.in_channel, 6, kernel_size=5, stride=1, padding=0)
        # [(32 - 5 + 2*0) / 1] + 1 = 28
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)

        self.pool0 = nn.AvgPool2d(2, stride=2)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        
        self.fc0 = nn.Linear(120, 84)
        self.fc1 = nn.Linear(84, self.n_classes)

    # 실제 layer 연산 define    
    def forward(self, x):
        # x' shape = [B, C, H, W]
        x = self.conv0(x)
        x = torch.tanh(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        # x = self.pool0(x)

        x = self.conv1(x)
        x = torch.tanh(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        # x = self.pool1(x)

        x = self.conv2(x)
        x = torch.tanh(x)
        
        # change format from 3D to 2D ([B, C, H, W] -> B,C*H*W) 
        x = torch.flatten(x, start_dim=1)
        x = self.fc0(x)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = x.view(self.batch, -1)
        x = nn.functional.softmax(x, dim=1) 
        
        # 학습할 때는 모든 결과를 받아야 함
        if self.is_train is False:
            x = torch.argmax(x, dim=1)

        return x
```

<br>

### models.py

위에서 선언해둔 lenet5 모델을 불러온다. 이를 생성한 이유는 lenet5 이외에 다른 모델도 사용할 때 불러오기 편하도록 하기 위해서이다.

```python
from model.lenet5 import Lenet5

def get_model(model_name):
    if(model_name == "lenet5"):
        return Lenet5
    else:
        print("not exist this model : {}".format(model_name))
```

선언해둔 lenet5.py 의 Lenet5를 불러와 모벨을 리턴한다. 만약 선언되어 있지 않은 모델을 인자로 받으면 없다고 출력한다.

<br>

## util/tools.py

예측된 결과를 출력하고, 시각화하기 위해 만든 파일이다. 출력을 위해 PIL의 ImageDraw를 사용했다. 입력 받는 데이터를 정규화를 해제하고, 

```python
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def show_img(img_data, text):
    # 입력된 값이 0~1로 정규화된 값이므로 255를 곱함
    _img_data = img_data * 255

    print(_img_data.shape) # (1, 1, 32, 32)

    # 4d tensor -> 2d
    _img_data = np.array(_img_data[0,0], dtype=np.uint8)

    img_data = Image.fromarray(_img_data)
    draw = ImageDraw.Draw(img_data)

    # 예측결과을 텍스트로 보기 위해 선언
    # draw text in img, center_x, center_y
    cx, cy = _img_data.shape[0] / 2, _img_data.shape[1] / 2
    if text is not None:
        draw.text((cx,cy), text)

    plt.imshow(img_data)
    plt.show()
```

입력된 값들은 pytorch에서의 4dimension이므로 이를 출력하기 위해 2dimension으로 변환한다. 그 후 Image 포맷으로 변경하고, 이미지를 그린다. 여기서 cx,cy는 정답 라벨을 화면 위에 표시하기 위한 것들이다.

<img src="/assets\img\dev\week10\day1\imagedraw.png">

<br>

# requirements.txt

```txt
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
asttokens==2.0.5
attrs==21.4.0
backcall==0.2.0
beautifulsoup4==4.11.1
bleach==5.0.0
certifi==2021.10.8
cffi==1.15.0
charset-normalizer==2.0.12
colorama==0.4.4
cycler==0.11.0
debugpy==1.6.0
decorator==5.1.1
defusedxml==0.7.1
entrypoints==0.4
executing==0.8.3
fastjsonschema==2.15.3
fonttools==4.32.0
idna==3.3
ipykernel==6.13.0
ipython==8.2.0
ipython-genutils==0.2.0
ipywidgets==7.7.0
jedi==0.18.1
Jinja2==3.1.1
jsonschema==4.4.0
jupyter==1.0.0
jupyter-client==7.2.2
jupyter-console==6.4.3
jupyter-core==4.9.2
jupyterlab-pygments==0.2.2
jupyterlab-widgets==1.1.0
kiwisolver==1.4.2
MarkupSafe==2.1.1
matplotlib==3.5.1
matplotlib-inline==0.1.3
mistune==0.8.4
nbclient==0.6.0
nbconvert==6.5.0
nbformat==5.3.0
nest-asyncio==1.5.5
notebook==6.4.10
numpy==1.22.3
packaging==21.3
pandocfilters==1.5.0
parso==0.8.3
pickleshare==0.7.5
Pillow==9.1.0
prometheus-client==0.14.1
prompt-toolkit==3.0.29
psutil==5.9.0
pure-eval==0.2.2
pycparser==2.21
Pygments==2.11.2
pyparsing==3.0.8
pyrsistent==0.18.1
python-dateutil==2.8.2
pywin32==303
pywinpty==2.0.5
pyzmq==22.3.0
qtconsole==5.3.0
QtPy==2.0.1
requests==2.27.1
Send2Trash==1.8.0
six==1.16.0
soupsieve==2.3.2.post1
stack-data==0.2.0
terminado==0.13.3
tinycss2==1.1.1
torch==1.11.0+cu113
torchaudio==0.11.0+cu113
torchvision==0.12.0+cu113
tornado==6.1
tqdm==4.64.0
traitlets==5.1.1
typing_extensions==4.2.0
urllib3==1.26.9
wcwidth==0.2.5
webencodings==0.5.1
widgetsnbextension==3.6.0
```



