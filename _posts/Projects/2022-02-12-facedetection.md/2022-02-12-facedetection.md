---
title:    "[detection] 얼굴 이미지를 통해 동물상 테스트"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2022-02-12 18:02:00 +0800
categories: [Projects, face detection]
tags: [projects, Detection]
toc: True
comments: True
math: true
mermaid: true
image:
  src: /assets/img/facedet/main.jpg
  width: 500
  height: 500
---

[깃허브 주소](https://github.com/dkssud8150/face)

# Abstract

업로드된 이미지에 대해 어떤 동물을 닮았는지 분류하는 테스트를 만들었다. 

![image](/assets/img/facedet/upimg.jpg "before upload image")
![image](/assets/img/facedet/upedimg.jpg "prediction result about upload image")

`python + flask + html + css`

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
  ├── chromedriver.exe
      └── model_best_epoch.pt
  ├── train.py
  .
  .
  └── README.md
```

<br>

INDEX
1. 이미지 크롤링 (data/, google.py, chromedriver.exe)
2. 훈련/테스트 (weight/, train.py)
3. 서버와 연결 (html, templates/, app.py)
4. 구동 테스트 (static/)
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
# install library
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request
import os


# name list for downloading
train_namespace = ["dog","cat","dinosaur","rabbit","fox"]


# set my baseurl
baseurl = "C:/Users/dkssu/Github/face"


# make directory to save image 
k = os.path.join(baseurl,"data/train",name.split(" ")[0])        
os.makedirs(k, exist_ok=True)
```

- 먼저 필요한 라이브러리를 설치해준다.
- 다운로드해줄 동물 이름을 설정한다.
- 기본 폴더 경로인 `baseurl`을 설정해준다.


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







# Summary


