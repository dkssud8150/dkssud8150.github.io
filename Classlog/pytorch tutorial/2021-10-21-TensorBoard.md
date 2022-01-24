---
title:    "TensorBoard 사용해보기"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2021-10-21 12:00:00 +0800
categories: [Classlog,Pytorch Tutorial]
tags: [pytorch tutorial,TensorBoard]
toc: True
comments: True
math: true
mermaid: true
#image:
#  src: /commons/devices-mockup.png
#  width: 800
#  height: 500
---

* 기본 모델 구조

```python
# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=0)

# 분류 결과를 위한 상수
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

# 1. TensorBoard 설정

```python
from torch.utils.tensorboard import SummaryWriter

# 기본 'log_dir'은 "runs"이며, 여기서는 더 구체적으로 지정한다.
writer = SummaryWriter('runs/fashion_mnist_experiment_1') # run/fashion_mnist_experiment_1 폴더를 생성한다.
```
<br>

<br>

# 2. TensorBoard에 기록

이제 TensorBoard에 이미지(구체적으로는 make_grid를 사용하여 그리드)를 나타내보자.

```python
# 이미지를 보여주기 위한 함수
# (아래 `plot_classes_preds` 함수에서 사용)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 임의의 학습 이미지를 가져온다.
dataiter = iter(trainloader) # iter = for 반복과 비슷한 것
images, labels = dataiter.next() # 반복가능한 객체의 다음 요소를 반환

# 이미지 그리드를 만든다
img_grid = torchvision.utils.make_grid(images)

# 이미지를 보여준다.
matplotlib_imshow(img_grid, one_channel = True)

# tensorboard에 기록
writer.add_image('four_fashion_mnist_images', img_grid)
```

```shell
%load_ext tensorboard
%tensorboard --logdir=runs
```

<br>

<br>

# 3. TensorBoard를 사용하여 모델 살표보기(inspect)

복잡한 모델 구조를 시각화하는 기능

<br>

Net을 더블클릭하여 펼쳐보면 모델을 구성하는 개별 연산들에 대해 자세히 볼 수 있다.

tensorboard는 이미지 데이터와 같은 고차원 데이터를 저차원 공간에 시각화하는데 매우 편리한 기능을 제공한다.

```python
writer.add_graph(net, images)
writer.close()
```

<br>

<br>

# 4. TensorBoard에 Projector 추가하기

add_embedding 메소드를 통해 고차원 데이터의 저차원 표현을 시각화할 수 있다.

```python
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def select_n_random(data, labels, n=100):
    '''
    데이터셋에서 n개의 임의의 데이터포인트(datapoint)와 그에 해당하는 라벨을 선택합니다
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# 임의의 이미지들과 정답(target) 인덱스를 선택합니다
images, labels = select_n_random(trainset.data, trainset.targets)

# 각 이미지의 분류 라벨(class label)을 가져옵니다
class_labels = [classes[lab] for lab in labels]

# 임베딩(embedding) 내역을 기록합니다
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()
```

<br>

이를 통해 projector 탭에서 각각은 784차원인 100개의 이미지가 3차원 공간에 투사된 것을 볼 수 있다.

또한, 이것은 대화식으로 클릭하고 드래그하여 3차원으로 투영된 것을 회전할 수 있다.

마지막으로 시각화를 더 편히 볼 수 있는 팁으로는 좌측 상단에서 "color by:label"을 선택하고, 야간모드를 활성화하면 이미지 배경이 흰색이 되어 더 편하게 볼 수 있다.

<br>

<br>

# 4. TensorBoard로 모델 학습 추적

데이터를 충분히 살펴보았으므로 이제 학습 과정부터 시작하여 TensorBoard가 어떻게 모델 학습과 평가를 더 명확히 추적할 수 있는지 살펴보고자 한다.

<br>

이전 예제에서는 단순히 모델 학습 중 손실(running loss)을 2000번 반복할 때마다 출력하기만 했다. 이제는 tensorboard에 학습 중 손실을 기록하는 것 대신에 plot_classes_predcs함수를 통해 모델의 예측 결과를 함께 볼 수 있도록 하겠다.

<br>

```python
def images_to_probs(net, images):
    '''
    학습된 신경망과 이미지 목록으로부터 예측 결과 및 확률을 생성합니다
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    학습된 신경망과 배치로부터 가져온 이미지 / 라벨을 사용하여 matplotlib
    Figure를 생성합니다. 이는 신경망의 예측 결과 / 확률과 함께 정답을 보여주며,
    예측 결과가 맞았는지 여부에 따라 색을 다르게 표시합니다. "images_to_probs"
    함수를 사용합니다.
    '''
    preds, probs = images_to_probs(net, images)
    # 배치에서 이미지를 가져와 예측 결과 / 정답과 함께 표시(plot)합니다
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
```

<br>

마지막으로 이전 튜토리얼과 동일한 모델 학습 코드에서 1000배치마다 콘솔에 출력하는 대신에 tensorboard에 결과를 기록하도록 하여 학습해보자. 이는 add_scalar함수를 사용한다.

또한, 학습을 진행하면서 배치에 포함된 4개의 이미지에 대한 모델의 예측 결과와 정답을 비교(versus)하여 보여주는 이미지를 생성하도록 한다.

<br>

```python
running_loss = 0.0
for epoch in range(1):

    for i, data in enumerate(trainloader):

        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # 매 1000 미니배치마다...

            # ...학습 중 손실(running loss)을 기록하고
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i)

            # ...무작위 미니배치(mini-batch)에 대한 모델의 예측 결과를 보여주도록
            # Matplotlib Figure를 기록합니다
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Training')
```

<br>

이렇게 하면 scalars 탭에서 15000번 반복할 때의 손실을 확인할 수 있다.

또한, 학습 과정 전반에 걸쳐 임의의 배치에 대한 모델의 예측 결과를 확인할 수 있다. "images"탭에서 스크롤을 내려 "예측 vs 정답(predictions Vs actuals)" 시각화 부분에서 이 내용을 볼 수 있다.

<br>

<br>

# 5. TensorBoard로 학습된 모델 평가

이전에는 모델이 학습 완료된 후에 각 분류별 정확도를 살펴보았다. 이제느 tensorboard를 사용하여 각 분류별 정밀도-재현율 곡선을 그릴 수 있다.

<br>

```python
# 1. 예측 확률을 test_size x num_classes 텐서로 가져옵니다
# 2. 예측 결과를 test_size 텐서로 가져옵니다
# 실행하는데 10초 이하 소요
class_probs = []
class_label = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]

        class_probs.append(class_probs_batch)
        class_label.append(labels)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_label = torch.cat(class_label)

# 헬퍼 함수
def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
    '''
    0부터 9까지의 "class_index"를 가져온 후 해당 정밀도-재현율(precision-recall)
    곡선을 그립니다
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# 모든 정밀도-재현율(precision-recall; pr) 곡선을 그립니다
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)
```

<br>

PR Curves 탭에서 각 분류별 정밀도-재현율 곡선을 볼 수 있다.

<br>

<br>

# Reference
- [https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html](https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html)