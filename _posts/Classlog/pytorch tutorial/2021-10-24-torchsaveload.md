---
title:    "Torch Save and Load, data를 plot해보기"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2021-10-24 12:00:00 +0800
categories: [Classlog,Pytorch Tutorial]
tags: [pytorch tutorial,Torch save,plot]
toc: True
comments: True
math: true
mermaid: true
#image:
#  src: /commons/devices-mockup.png
#  width: 800
#  height: 500
---

# 기본 구조

기본 구조는 다음과 같다. 자세한 내용은 [참고 사이트](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)를 참고하길 바란다.

## Save

기본적으로 모델을 저장하는 일반적인 방법은 내부 상태 사전(internal state dictionary)를 직렬화(serialize)하는 것이다.

```python
torch.save(model.state_dic(), "model.pth")
print("save pytorch model state to model.pth")
```


## Load

모델을 불러오는 과정은 모델 구조를 다시 만들고 상태 사전을 모델에 불러와야 한다.

```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```

이 모델을 통해 예측을 진행한다.

```python
model.eval()

x, y = test_data[0],test_data[0]
with torch.no_grad():
  pred = model(x)
  predicted, actual = classes[pred[0].argmax(0), classes[y]]
  print("predicted: {}, actual: {}".format(predicted, actual))
```


# 실제 코드에 적용해보기

image classification에서의 best loss와 best accuracy에 대한 model을 저장하고 평가하고자 한다.

```python
#%env CUDA_VISIBLE_DEVICES=2

import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# Applying Transforms to the Data - normalization
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), # 랜덤 크기로 자른 후 잘라낸 이미지의 크기는 256x256로 조정
        transforms.RandomRotation(degrees=15),                    # 영상을 -15~15도 범위에서 랜덤 각도로 회전
        transforms.RandomHorizontalFlip(),                        # 50% 확률로 이미지를 수평으로 랜덤하게 뒤집는다.
        transforms.CenterCrop(size=224),                          # 중앙에서 224x224 이미지를 자른다.
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])               # 3채널 tensor을 취하며 각 채널의 입력 평균과 표준 편차로 각 채널을 정규화
    ]),
    'valid': transforms.Compose([                                 # 검증 및 테스트 데이터의 경우 resizedcrop과 rotalrotaion, horizontalfilp을 수행하지 않는다.
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the Data

# Set train and valid directory paths

dataset = '/content/drive/MyDrive/data/caltech_10/'

train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'valid')

# Batch size
BATCH_SIZE = 32

# Number of classes
num_classes = len(os.listdir(valid_directory))  #10#2#257 bear, chimp...
print('num_class :', num_classes)

# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']))
}

# Get a mapping of the indices to the class names, in order to see the output classes of the test images.
idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()} # class_to_idx 는 데이터 셋의 클래스 매핑 레이블을 반환
print(idx_to_class)

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
print(train_data_size, valid_data_size)

# Create iterators for the Data loaded using DataLoader module
train_data_loader = DataLoader(data['train'], batch_size=BATCH_SIZE, shuffle=True)
valid_data_loader = DataLoader(data['valid'], batch_size=BATCH_SIZE, shuffle=True)
```

```python
def train_and_val(model, criterion, optimizer, epochs=25):

  best_loss = 5
  best_epoch = None

  history = []



  for epoch in tqdm(range(epochs)):
    epoch_start = time.time()
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    valid_loss = 0.0
    valid_acc = 0.0

    correct = 0

    for i, (images, labels) in enumerate(train_data_loader):
      images = images.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()

      predictions = outputs.max(1, keepdim = True)[1]
      correct += predictions.eq(labels.data.view_as(predictions)).sum().item()
      
    train_loss /= (len(train_data_loader.dataset) / BATCH_SIZE)
    train_acc = 100. * correct / len(train_data_loader.dataset)
    

    correct = 0 
    
    with torch.no_grad():
      model.eval()

      for i, (images, labels) in enumerate(valid_data_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        valid_loss += criterion(outputs, labels).item()
        predictions = outputs.max(1, keepdim = True)[1]
        correct += predictions.eq(labels.data.view_as(predictions)).sum().item()

    valid_loss /= (len(valid_data_loader.dataset) / BATCH_SIZE)
    valid_acc = 100. * correct / len(valid_data_loader.dataset)

    if valid_loss < best_loss:
      best_loss = valid_loss
      best_epoch = epoch
      print("best_loss: {:.4f} \n best_epoch: {}".format(best_loss, best_epoch))

    history.append([train_loss,valid_loss, train_acc,valid_acc])

    epoch_end = time.time()

    print("Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.2f}%, \n\t\tValidation : Loss - {:.4f}, Accuracy - {:.2f}%, Time: {:.4f}s".format(epoch, train_loss, train_acc, valid_loss, valid_acc, epoch_end-epoch_start)) 

  return model, history, best_epoch
```

```python
''' pretrained model '''
import torchvision.models as models
model = models.resnet18(pretrained = False).to(device)           
num_ftrs = model.fc.in_features                                                            
model.fc = nn.Linear(num_ftrs, num_classes)      
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

train_model, history, best_epoch = train_and_val(model, criterion, optimizer)

torch.save(model, dataset+'_model_'+str(best_epoch)+'.pt')
```

## 가져온 history를 통해 그래프 그리기

```python
''' loss 그래프 '''
history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
#plt.ylim(0,1)
plt.savefig(dataset+'_loss_curve.png')
plt.show()
```

```python
''' accuracy 그래프 '''
plt.plot(history[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
#plt.ylim(0,1)
plt.savefig(dataset+'_accuracy_curve.png')
plt.show()
```


## 단일 이미지를 best model로 예측해보기

아래는 topk를 사용해보았다. 이는 topk에 대한 결과값을 받을 수 있다. 아래의 경우 k를 3으로 지정하였기에 확률이 가장 높은 3개에 대한 라벨 확률을 확인 할 수 있다.


```python
def predict(model, test_image_name):
  test_image = Image.open(test_image_name)
  plt.imshow(test_image)

  transform = image_transforms['test']
  test_image_tensor = transform(test_image)
  test_image_tensor = test_image_tensor.view(1,3,224,224).cuda()

  with torch.no_grad():
    model.eval()
    out = model(test_image_tensor)
    ps = torch.exp(out)

    topk, topclass = ps.topk(3, dim=1)  # argmax와 비슷하게 top-k에 대한 결과 값을 받는다.
    score = topk.cpu().numpy()[0][0]

    for i in range(3):
      print("Prediction", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])
```

```python
import wget

!wget https://cdn.pixabay.com/photo/2018/10/01/12/28/skunk-3716043_1280.jpg -O skunk.jpg
model = torch.load("{}_model_{}.pt".format(dataset, best_epoch))
predict(model, 'skunk.jpg')
```

# Reference
- [https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)
