---
title:    "[detection] 얼굴 이미지를 통해 동물상 테스트"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-02-12 18:02:00 +0800
categories: [Projects, detection]
tags: [projects, face detection]
toc: True
comments: True
math: true
mermaid: true
image:
  src: /assets/img/facedet/main.jpg
  width: 500
  height: 500
---

[깃허브 주소](https://github.com/dkssud8150/Animalface-detector)

# Abstract

업로드된 이미지에 대해 어떤 동물을 닮았는지 분류하는 테스트를 만들었다. 

<img src="/assets/img/facedet/upimg.jpg" title="before upload image" width="43%">
*before upload image*
<img src="/assets/img/facedet/upedimg.jpg" title="after upload image" width="57%">
*prediction result about upload image*

python을 통해 딥러닝 모델을 구축하고, flask로 서버와 html을 연결했다. 

* directory

```markdown
face
  └── data
      ├── train
          ├── cat
              ├── 0.jpg
              ├── 1.jpg
              .
              .
          ├── dinosasur
              ├── 0.jpg
              ├── 1.jpg
              .
              .
          .
          .
      ├── test
          ├── Angelina
              ├── 0.jpg
              ├── 1.jpg
              .
              .
          ├── cha
              ├── 0.jpg
              ├── 1.jpg
              .
              .
          .
          .
      └── google.py
  ├── static
      ├── cat.jpg
      ├── dinosaur.jpg
      └── img.jpg
      .
      .
  ├── templates
      ├── index.html
      ├── test.html
      └── upload.html
  ├── weight
      └── model_best_epoch.pt
  ├── chromedriver.exe
  ├── app.py
  ├── train.py
  ├── ngrok.yml
  .
  .
  └── README.md
```

<br>

* INDEX
1. 이미지 크롤링 (data/, google.py, chromedriver.exe)
2. 훈련/테스트 (weight/, train.py)
3. 서버와 연결 (html, templates/, app.py)
4. 구동 테스트 (static/)

<br>

---

<br>

# 이미지 크롤링

훈련, 테스트 셋을 다운받기 위해 이미지 크롤링을 진행했다.

이미지 크롤링 코드와 방법은 [조코딩 유튜버님의 영상](https://www.youtube.com/watch?v=1b7pXC1-IbE)을 참고했다.

1. chromedriver.exe

<img src="/assets/img/facedet/chromedriver.jpg">

일단 우선 chromedriver를 다운해서 해당 폴더에 chromedriver.exd를 넣어야 한다. 위의 사이트로 들어가서 다운로드 한 후 exe파일을 옮겨준다.

2. google.py

이미지 크롤링을 하기 위해서는 일단 파일명을 google.py로 지정을 해야한다고 한다. 파일을 생성한 후 영상을 참고하여 코드를 구성했다. 

```python
# import library
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

import time
import urllib.request
import os
```

크롤링을 위한 라이브러리를 임포트했다.

<br>

```python
# name list for downloading
train_namespace = ["dog","cat","dinosaur","rabbit","fox"]
test_namespace = ["song min ho","Angelina Jolie", "Keanu Reeves", "Mark Ruffalo", "Elon Musk", "Robert Pattinson","IU ","cha eun woo","lee dong wook","han hyo joo"]

# your base url
baseurl = "C:/Users/dkssu/Github/Animalface-detector/"
```

데이터셋 라벨로 사용할 이름과 나의 workspace를 지정해주었다.

<br>

```python
for name in train_namespace:
    # make fold
    k = os.path.join(baseurl,"data/train",name.split(" ")[0])        
    os.makedirs(k, exist_ok=True)

    # solving chrome error
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # chrome excute
    driver = webdriver.Chrome(options=options)
    
    driver.get("https://www.google.co.kr/imghp?hl=ko&authuser=0&ogbl")
    elem = driver.find_element(By.NAME, "q")
    elem.send_keys(name)
    elem.send_keys(Keys.RETURN)
```

이미지를 저장할 폴더를 생성하고, 자동화 크롤링을 위한 webdriver를 실행하는 코드다. headless는 코드를 실행할동안 chrome 화면이 뜨지 않도록 하는 옵션이다.

그리고 driver를 실행하면 다음과 같은 화면이 뜬다.

<img src="/assets/img/facedet/main_page.png">

여기서 내가 원하는 이름의 이미지를 검색해서 다운받아야 하기 떄문에, 검색창의 요소를 찾아야 한다. 그 이름이 `q`였다. 그래서 find_element를 한 후에 name, 즉 내가 검색할 이름을 key로 보낸다.

<br>


```python
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(1)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element(By.CSS_SELECTOR, ".mye4qd").click()
            except:
                break
        last_height = new_height
```

이 코드는 검색을 하면 이미지가 20장 정도 한정되어 나올 것이다. 그러나 내가 딥러닝 모델을 돌리기 위해서는 더 많은 데이터가 필요하기 때문에 스크롤을 해주는 코드다. 스크롤을 하는 이유는 스크롤을 하지 않으면 출력되는 이미지가 한정적인데, 스크롤을 하게 되면 더 많은 이미지가 검색되고 출력되기 때문이다. 그 후, 100장 정도 출력되면 다음과 같은 버튼이 보인다.

<img src="/assets/img/facedet/button.png">

`결과 더보기` 버튼을 눌러야 더 많은 이미지가 나오기 때문에 이를 클릭해주는 코드다. 모든 이미지가 다 나오면 결과 더보기가 안 뜨므로 try & except 코드를 통해 결과 더보기 버튼이 있으면 누르고 없으면 루프를 빠져나오도록 했다.

```python
    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")

    # number of images
    cnt = 0

    for img in images:
        if not os.path.isfile(k + "/" + str(cnt) + ".png"):
            try:
                start = time.time()
                img.click()
                cnt+=1
                time.sleep(2)
                src = driver.find_element(By.XPATH, "/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img").get_attribute("src")
                if start > 1*10^-5: pass
                urllib.request.urlretrieve(src, k + "/" + str(cnt) + ".png")
                if cnt == 500:
                    print(name," Finish!")
                    break
            except:
                print("Do Not this")
                pass    
        else:
            cnt+=1
            print("Exists already")
            pass

driver.quit()
```

이미지들의 주소는 images 변수에 저장된다. 저장된 주소를 통해 데이터 폴더에 저장한다. 시간이 너무 오래 걸리거나 사진을 찾을 수 없으면 지나가도록 했으며, 500개를 저장하면 루프를 빠져나온다. 또는 동일한 이름의 이미지가 이미 존재하면 바로 다음 이미지 주소로 넘어간다.

중요한 것은 마지막에 driver를 종료해주어야 한다. 아니면 창이 백그라운드에 계속 남아 있다.

<br>

```python
'''
print("\nTest image download\n")

for name in test_namespace:
    # make fold
    k = os.path.join(baseurl,"data/test",name.split(" ")[0])        
    os.makedirs(k, exist_ok=True)

    # solving chrome error
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # chrome excute
    driver = webdriver.Chrome(options=options)
    
    driver.get("https://www.google.co.kr/imghp?hl=ko&authuser=0&ogbl")
    elem = driver.find_element(By.NAME, "q")
    elem.send_keys(name)
    elem.send_keys(Keys.RETURN)


    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(1)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element(By.CSS_SELECTOR, ".mye4qd").click()
            except:
                break
        last_height = new_height

    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")

    # number of images
    cnt = 0

    for img in images:
        if not os.path.isfile(k + "/" + str(cnt) + ".jpg"):
            try:
                start = time.time()
                img.click()
                time.sleep(2)
                src = driver.find_element(By.XPATH, "/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img").get_attribute("src")
                if start > 1*10^-5: pass
                urllib.request.urlretrieve(src, k + "/" + str(cnt) + ".jpg")
                cnt+=1
                if cnt == 20:
                    print(name," Finish!")
                    break
            except:
                print("Do Not this")
                pass    
        else: 
            print("Exists already")
            pass

driver.quit()'''
```

이는 테스트 데이터셋을 생성하기 위한 코드다. 위와 동일한 구성이다.

<br>

<br>

* 전체 코드

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request
import os

# name list for downloading
train_namespace = ["dog","cat","dinosaur","rabbit","fox"]
test_namespace = ["song min ho","Angelina Jolie", "Keanu Reeves", "Mark Ruffalo", "Elon Musk", "Robert Pattinson","IU ","cha eun woo","lee dong wook","han hyo joo"]

# your base url
baseurl = "C:/Users/dkssu/Github/face"

for name in train_namespace:
    # make fold
    k = os.path.join(baseurl,"data/train",name.split(" ")[0])        
    os.makedirs(k, exist_ok=True)

    # solving chrome error
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # chrome excute
    driver = webdriver.Chrome(options=options)
    
    driver.get("https://www.google.co.kr/imghp?hl=ko&authuser=0&ogbl")
    elem = driver.find_element(By.NAME, "q")
    elem.send_keys(name)
    elem.send_keys(Keys.RETURN)


    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(1)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element(By.CSS_SELECTOR, ".mye4qd").click()
            except:
                break
        last_height = new_height

    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")

    # number of images
    cnt = 0

    for img in images:
        if not os.path.isfile(k + "/" + str(cnt) + ".jpg"):
            try:
                start = time.time()
                img.click()
                cnt+=1
                time.sleep(2)
                src = driver.find_element(By.XPATH, "/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img").get_attribute("src")
                if start > 1*10^-5: pass
                urllib.request.urlretrieve(src, k + "/" + str(cnt) + ".jpg")
                if cnt == 500:
                    print(name," Finish!")
                    break
            except:
                print("Do Not this")
                pass    
        else:
            cnt+=1
            print("Exists already")
            pass

driver.quit()
```

<br>

<br>

# 훈련/테스트

1. train.py

모델 훈련을 위한 파일이다.

* 전체 코드

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))

# 한글 폰트 설정하기
fontpath = 'C:/Windows/Fonts/NanumGothicLight.ttf'
font = fm.FontProperties(fname=fontpath, size=10).get_name()
plt.rc('font', family=font)


# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#CUDA_LAUNCH_BLOCKING=1


# 데이터셋을 불러올 때 사용할 변형(transformation) 객체 정의
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 데이터 증진(augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = './data'

train_datasets = datasets.ImageFolder(os.path.join(data_dir,'train'), transforms_train)

train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=4, shuffle=True, num_workers=0)
#valid_dataloader = torch.utils.data.DataLoader(valid_datasets, batch_size=4, shuffle=False, num_workers=0)

print('학습 데이터셋 크기:', len(train_datasets))
#print('테스트 데이터셋 크기:', len(valid_datasets))

class_names = train_datasets.classes
print('학습 클래스:', class_names)



def imshow(input, title):
    # torch.Tensor를 numpy 객체로 변환
    input = input.numpy().transpose((1, 2, 0))
    # 이미지 정규화 해제하기
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # 이미지 출력
    plt.imshow(input)
    plt.title(title)
    plt.show()



# 학습 데이터를 배치 단위로 불러오기
iterator = iter(train_dataloader)
# 현재 배치를 이용해 격자 형태의 이미지를 만들어 시각화
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])


# implement model
model = models.resnet34(pretrained=True)
num_features = model.fc.in_features

# transfer learning
model.fc = nn.Sequential(     
    nn.Linear(num_features, 256),        # 마지막 완전히 연결된 계층에 대한 입력은 선형 계층, 256개의 출력값을 가짐
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_features),      # Since 10 possible outputs = 10 classes
    nn.LogSoftmax(dim=1)              # For using NLLLoss()
)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

model = model.to(device)

num_epochs = 30

best_epoch = None
best_loss = 5

''' Train '''
# 전체 반복(epoch) 수 만큼 반복하며
for epoch in range(num_epochs):
    model.train()
    start_time = time.time()
    
    running_loss = 0.
    running_corrects = 0

    # 배치 단위로 학습 데이터 불러오기
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 모델에 입력(forward)하고 결과 계산
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # 역전파를 통해 기울기(gradient) 계산 및 학습 진행
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_datasets)
    epoch_acc = running_corrects / len(train_datasets) * 100.

    # 학습 과정 중에 결과 출력
    print('#{} Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch = epoch
        print("best_loss: {:.4f} \t best_epoch: {}".format(best_loss, best_epoch))

os.makedirs('./weight',exist_ok=True)
torch.save(model, './weight/model_best_epoch.pt')

''' Valid 
with torch.no_grad():
    model.eval()
    start_time = time.time()
    
    running_loss = 0.
    running_corrects = 0

    for inputs, labels in valid_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # 한 배치의 첫 번째 이미지에 대하여 결과 시각화
        print(f'[예측 결과: {class_names[preds[0]]}] (실제 정답: {class_names[labels.data[0]]})')
        imshow(inputs.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])

    epoch_loss = running_loss / len(valid_datasets)
    epoch_acc = running_corrects / len(valid_datasets) * 100.
    print('[valid Phase] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch_loss, epoch_acc, time.time() - start_time)) '''




''' Test '''

valid_images = []
valid_dir = data_dir + '/test'

val_folders = glob(valid_dir + '/*')
for val_folder in val_folders:
    image_paths = glob(val_folder + '/*')
    for image_path in image_paths: valid_images.append(image_path)


import random
num = random.randint(0,len(valid_images)-1)
valid_image = valid_images[num]


from PIL import Image
image = Image.open(valid_image)
image = transforms_test(image).unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    imshow(image.cpu().data[0], title=' 학습 결과 : ' + class_names[preds[0]])
```

<br>

<br>

코드를 상세히 설명하자면 먼저 필요한 라이브러리들을 임포트한다.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
```

<br>

```python
fontpath = 'C:/Windows/Fonts/NanumGothicLight.ttf'
font = fm.FontProperties(fname=fontpath, size=10).get_name()
plt.rc('font', family=font)
```

그 후 출력을 한글로 해야 하는데, matplotlib에는 한글 패치가 적용되어 있지 않다. 설정하지 않은 채로 한글을 출력하면 ▯형태로 출력된다. 그래서 직접 설정해주었다.

<br>

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

gpu를 사용하기 위해 설정해주었다.

<br>

```python
# 데이터셋을 불러올 때 사용할 변형(transformation) 객체 정의
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 데이터 증진(augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

데이터 증강을 위해 transform을 해야 한다. 이를 선언해주었다. 이미지 크기를 맞춰서 넣기 위해 resize를 하고, augmentation을 위해 flip(뒤집기)를 해준다. pytorch를 사용할 예정이므로 ToTensor를 넣어줘야 한다. 마지막으로 이미지 정규화를 위해 normalize를 추가했다. test에서는 데이터 증강을 하면 안되기 때문에 flip부분을 삭제했다.

<br>

```python
data_dir = './data'
train_datasets = datasets.ImageFolder(os.path.join(data_dir,'train'), transforms_train)

train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=4, shuffle=True, num_workers=0)
#valid_dataloader = torch.utils.data.DataLoader(valid_datasets, batch_size=4, shuffle=False, num_workers=0)

print('학습 데이터셋 크기:', len(train_datasets))
#print('테스트 데이터셋 크기:', len(valid_datasets))

class_names = train_datasets.classes
print('학습 클래스:', class_names)
```

robust하지는 않지만, 간단하게 학습하기 위해 customdataset이 아닌 iamgefolder과 dataloader 메서드를 사용하여 데이터셋을 생성했다. 컴퓨터 성능을 고려해서 batch_size를 작세 설정했다.

추후 손실 함수에 사용될 class name도 변수에 저장해놓는다.

<br>

```python
def imshow(input, title):
    # torch.Tensor를 numpy 객체로 변환
    input = input.numpy().transpose((1, 2, 0))
    # 이미지 정규화 해제하기
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # 이미지 출력
    plt.imshow(input)
    plt.title(title)
    plt.show()
```

validation에서 사용할 imshow함수다. 단순하게 검증한 이미지를 예측값과 함께 출력한다.

<br>

```python
# implement model
model = models.resnet34(pretrained=True)
num_features = model.fc.in_features

# transfer learning
model.fc = nn.Sequential(     
    nn.Linear(num_features, 256),        # 마지막 완전히 연결된 계층에 대한 입력은 선형 계층, 256개의 출력값을 가짐
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_features),      # Since 10 possible outputs = 10 classes
    nn.LogSoftmax(dim=1)              # For using NLLLoss()
)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

model = model.to(device)

num_epochs = 30

best_epoch = None
best_loss = 5
```

커스터마이징을 하지 않았기 때문에 pretrained model을 사용했다. 여기서 마지막 fc layer는 반드시 수정해줘야 한다. 원래의 resnet34의 출력 개수와 내가 출력하고자 하는 출력의 크기가 다르기 때문이다. NLLLoss를 사용하기 위해 logsoftmax 층을 추가했다.

손실함수는 NLLLoss, 최적화 함수는 SGD를 사용했다. learing rate와 weight decay를 설정했다.

gpu를 사용할 수 있다면 gpu를 사용하여 연산하기 위해 to(device)를 했다.

반복할 횟수를 지정해주고, 최고의 성능을 저장할 것이기 때문에 그것을 저장할 변수도 생성한다.

<br>

```python
''' Train '''
# 전체 반복(epoch) 수 만큼 반복하며
for epoch in range(num_epochs):
    model.train()
    start_time = time.time()
    
    running_loss = 0.
    running_corrects = 0

    # 배치 단위로 학습 데이터 불러오기
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 모델에 입력(forward)하고 결과 계산
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # 역전파를 통해 기울기(gradient) 계산 및 학습 진행
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_datasets)
    epoch_acc = running_corrects / len(train_datasets) * 100.

    # 학습 과정 중에 결과 출력
    print('#{} Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))
```

모델을 훈련시키는 과정은 다음과 같다. 분류 모델이므로 가장 score가 높은 값을 추출하면 되므로 max를 사용했다. 만약 1위,2위,3위를 출력하기 위해서는 topk를 사용하면 된다.

<br>

```python
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch = epoch
        print("best_loss: {:.4f} \t best_epoch: {}".format(best_loss, best_epoch))

os.makedirs('./weight',exist_ok=True)
torch.save(model, './weight/model_best_epoch.pt')
```

가장 성능이 좋았던 epoch을 저장하여 pt파일로 저장한다.

<br>

- 간단하게 모델 테스트 해보기

```python
valid_images = []
valid_dir = data_dir + '/test'

val_folders = glob(valid_dir + '/*')
for val_folder in val_folders:
    image_paths = glob(val_folder + '/*')
    for image_path in image_paths: valid_images.append(image_path)


import random
num = random.randint(0,len(valid_images)-1)
valid_image = valid_images[num]


from PIL import Image
image = Image.open(valid_image)
image = transforms_test(image).unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    imshow(image.cpu().data[0], title=' 학습 결과 : ' + class_names[preds[0]])
```

<br>

# 서버와 연결

1. app.py

파이썬과 flask를 연동해서 사용했다.

* 전체 코드

```python
''' 분류 모델 API 
학습된 모델을 다른 사람들이 사용할 수 있도록 api를 만들어 배포 '''

from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import os
import matplotlib
import matplotlib.font_manager as fm
from glob import glob

# print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))

# 한글 폰트 설정하기
fontpath = 'C:/Windows/Fonts/NanumGothicLight.ttf'
font = fm.FontProperties(fname=fontpath, size=10).get_name()
plt.rc('font', family=font)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 데이터 증진(augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = './data'
train_datasets = datasets.ImageFolder(os.path.join(data_dir,'train'), transforms_train)
class_names = train_datasets.classes

model = torch.load("./weight/model_best_epoch.pt")

'''웹 API 개방을 위해 ngrok 서비스 이용
    API 기능 제공을 위해 Flask 프레임워크 사용 '''

# 필요한 라이브러리 설치하기
import io


# 이미지를 읽어 결과를 반환하는 함수
def get_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(image)
        outputs = torch.exp(outputs)
        topk, topclass = outputs.topk(3, dim=1) # argmax와 비슷하게 top-k에 대한 결과 값을 받는다.
    
        classes = [class_names[i] for i in topclass.cpu().numpy()[0]]
        scores = [round(i*100,2) for i in topk.cpu().numpy()[0]]

        print(classes, scores)

    return classes,scores



from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request, render_template
from flask import jsonify

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def hello():
    return render_template('index.html')



@app.route('/upload_image', methods=['POST','GET'])
def upload_image_file():
    if request.method == 'POST':
        file = request.files['uploaded_image']
        if not file: return "No Files"
        image_bytes = file.read()
        

        up_image = Image.open(io.BytesIO(image_bytes))
        up_image.save("./static/img.jpg","jpeg")


        # 분류 결과 확인 및 클라이언트에게 결과 반환
        classes,scores = get_prediction(image_bytes=image_bytes)
        class_name = classes[0]


        ''' 예측된 클래스에 대한 무작위 사진 가져오기
        class_images = []
        class_dir = './data/train/' + class_name
        train_paths = sorted(glob(class_dir + '/*'),key= lambda x: x.split("\\")[-1].split('.')[0])
        for train_path in train_paths: class_images.append(train_path)

        import random
        num = random.randint(0,10)
        img_path = class_images[num]

        pr_image = Image.open(img_path)
        pr_image.save("./static/" + class_name + ".jpg","jpeg") '''


        return render_template('upload.html', classes = classes, scores = scores, label = class_name, upload_img = 'img.jpg', predict_img = class_name + '.jpg') 
    else:
        return jsonify({"Methods == ":request.method})
    


@app.route('/test', methods=['POST','GET'])
def testing():
    if request.method == 'POST':
        result = request.form
        return render_template('test.html',label = result)
    else:
        return render_template('test.html',label = request.method)


if __name__ == "__main__":
    app.debug = True
    app.run()


## 사용 방식
# curl -X POST -F file=@{이미지 파일명} {Ngrok 서버 주소}

## 사용 예시
# curl -X POST -F file=@dongseok.jpg http://c4cdb8de3a35.ngrok.io/

# 참고
# https://velog.io/@qsdcfd/%EC%9B%B9-%ED%8E%98%EC%9D%B4%EC%A7%80-%EB%A7%8C%EB%93%A4%EA%B8%B0
```

<br>

```python
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import os
import matplotlib
import matplotlib.font_manager as fm
from glob import glob
import io

fontpath = 'C:/Windows/Fonts/NanumGothicLight.ttf'
font = fm.FontProperties(fname=fontpath, size=10).get_name()
plt.rc('font', family=font)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 데이터 증진(augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = './data'
train_datasets = datasets.ImageFolder(os.path.join(data_dir,'train'), transforms_train)
class_names = train_datasets.classes
```

test 결과를 웹에서 출력하기 위해서 test에 필요한 변수들을 지정해준다.

<br>

```python
model = torch.load("./weight/model_best_epoch.pt")
```

저장했던 모델을 불러온다.

<br>

```python
# 이미지를 읽어 결과를 반환하는 함수
def get_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(image)
        outputs = torch.exp(outputs)
        topk, topclass = outputs.topk(3, dim=1) # argmax와 비슷하게 top-k에 대한 결과 값을 받는다.
    
        classes = [class_names[i] for i in topclass.cpu().numpy()[0]]
        scores = [round(i*100,2) for i in topk.cpu().numpy()[0]]

        print(classes, scores)

    return classes,scores
```

이미지를 인자로 받아 이미지를 열고, 모델에 집어넣어 예측한다. 이 때, 1,2,3 순위를 보기 위해 topk를 사용했다. 예측한 클래스들과 점수들을 리턴한다.

<br>

```python
from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request, render_template
from flask import jsonify

app = Flask(__name__)
run_with_ngrok(app)
```

flask를 위한 초기 세팅을 한다. 이 때, ngrok.yml 파일이 필요하다.

<br>

```python
@app.route('/')
def hello():
    return render_template('index.html')
```

웹사이트를 접속했을 때 가장 먼저 나오는 화면에 대한 코드다. index.html은 다음과 같다.

- templates/index.html

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset = "UTF-8">
  <title> main page </title>
<body>
  <!--
  <h1>Site URL</h1>
  <form action = "/test" method = "POST">
    <a href="../test">GET test</a> 
    
    <button>POST test</button><br>
  </form>
  <a href="../upload_image">upload_image</a>
    
  <br>
  -->
    
  <form action = "/upload_image" method = "POST" enctype = "multipart/form-data">
    <h2>이미지 업로드 하기</h2>
    
    <div>
    
    <label for="file">Choose file to upload</label>
    <input type="file" name="uploaded_image" id="uploaded_image" accept="image/*" required="True" value="업로드">
    <br>
    <img src="no" id="img_section" style="width: 400px; height: 400px;">
    
    <script>
        const reader = new FileReader();

        reader.onload = (readerEvent) => {
            document.querySelector("#img_section").setAttribute("src", readerEvent.target.result);
            //파일을 읽는 이벤트가 발생하면 img_section의 src 속성을 readerEvent의 결과물로 대체함
        };

        document.querySelector("#uploaded_image").addEventListener("change", (changeEvent) => {
            //upload_file 에 이벤트리스너를 장착

            const imgFile = changeEvent.target.files[0];
            reader.readAsDataURL(imgFile);
            //업로드한 이미지의 URL을 reader에 등록
        })
    </script>
    
    <button>이미지 업로드</button>

  </div>
</form>
</body>
</head>
</html>
```

<br>

```python
@app.route('/upload_image', methods=['POST','GET'])
def upload_image_file():
    if request.method == 'POST':
        file = request.files['uploaded_image']
        if not file: return "No Files"
        image_bytes = file.read()
        

        up_image = Image.open(io.BytesIO(image_bytes))
        up_image.save("./static/img.jpg","jpeg")


        # 분류 결과 확인 및 클라이언트에게 결과 반환
        classes,scores = get_prediction(image_bytes=image_bytes)
        class_name = classes[0]


        ''' 예측된 클래스에 대한 무작위 사진 가져오기
        class_images = []
        class_dir = './data/train/' + class_name
        train_paths = sorted(glob(class_dir + '/*'),key= lambda x: x.split("\\")[-1].split('.')[0])
        for train_path in train_paths: class_images.append(train_path)

        import random
        num = random.randint(0,10)
        img_path = class_images[num]

        pr_image = Image.open(img_path)
        pr_image.save("./static/" + class_name + ".jpg","jpeg") '''


        return render_template('upload.html', classes = classes, scores = scores, label = class_name, upload_img = 'img.jpg', predict_img = class_name + '.jpg') 
    else:
        return jsonify({"Methods == ":request.method})
```

이미지가 입력되면 그 이미지를 받아 읽어서 모델에 집어넣는다. 그 후 출력된 클래스 상위 3개를 웹페이지로 전송시킨다. 이 때, label을 설정해준 이유는 예측이 되었는지를 확인하기 위한 장치로 설정했다. 

- templates/upload.html

```html
<!DOCTYPE html>
<html>
  <head>
    <title> upload page </title>
  </head>
  <body>

    <label for="label">예측 결과 : </label>
    {% if label %} 
      <span>
        {{ label }}
      </span> 
      <table border="1">
      <th>업로드 이미지</th>
      <th>예측 결과</th>
      <tr><!-- 첫번째 줄 시작 -->
        <td>
          <img src="{{ url_for('static',filename = upload_img)}}" alt="업로드 이미지" width="400px" height="400px">
        </td>
        <td>
          <img src="{{ url_for('static',filename = predict_img)}}" alt="예측 이미지" width="400px" height="400px">
        </td>
      </tr>
      <tr>
        <td></td>
        <td>
          {% if classes %}
            <a>
              <fir>1위: {{ classes[0] }} : {{ scores[0] }}% </fir><br>
              <sec>2위: {{ classes[1] }} : {{ scores[1] }}% </sec><br>
              <thi>3위: {{ classes[2] }} : {{ scores[2] }}% </thi><br>
            </a>
          {% endif %}
        </td>
      </tr>
      </table>

    {% endif %}
    <br>
  </body>
</html>

```

<br>

```python
# flask 실행
if __name__ == "__main__":
    app.debug = True
    app.run()
```



# Summary

모델 자체를 커스터마이징 하거나, 웹페이지를 더 예쁘게 꾸몄다면 사람들에게도 테스트를 해보고자 했지만, 웹페이지를 꾸밀 시간도 없었고, 1달이라는 짧은 시간안에 제작을 목표로 하여 pretrained model을 사용했기에 혼자만의 프로젝트로만 남겨주고자 한다.

다음번에는 객체 검출을 통해 바운딩 박스 표기도 하고 더 정확도 높고, robust한 모델을 사용하여 만들어보고자 한다.

데이터도 너무 한계가 있었다. 구글링하여 이미지를 다운받았기 때문에 얼굴 사진이 아닌 그 사람이 썼던, 글귀나 많은 사람들과 찍은 사진 등등 이상한 사진이 많았다. 특히 공룡의 경우 사진이 너무 없어서 2~300개의 데이터만을 사용했다.