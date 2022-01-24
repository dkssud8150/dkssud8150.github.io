---
title:    "Motion Keypoint Detection AI Contest with HRNet & YOLOv5"
author:
  name: JaeHo-YooN
  link: https://github.com/dkssud8150
date: 2021-09-16 12:00:00 +0800
categories: [Review, DACON]
tags: [DACON,keypoint, Detection]
toc: True
comments: True
math: true
mermaid: true
---

일단 yolo를 중점적으로 볼 예정이므로 test에 쓰이는 yolo만 보고 나중에 keypoint 를 공부한 후 hrnet 공부 후 볼 예정

1. this ordered seed list will be replaced by the toc
{:toc}

**Yolov5 + HRNet**
[에르모팀 DACON 코드](https://dacon.io/competitions/official/235701/codeshare/2478?page=1&dtype=recent)
[github](https://github.com/MaiHon/dacon-motion-keypoints)

## 정리

[학습]
* HRNet 기반의 pose_HRNet model
* Learning rate = 1e-3, Optimizer = Adam
* train set : valid set = 0.8 : 0.2
* Epoch = 20 , 14일 떄가 가장 결과가 좋았지만, mse loss가 7.4e-5보다 낮아지기 시작하면 score가 올라가기 시작해서 해당 지점에서 멈춤
* Loss = gaussian -> heatmap mse
* input size = [640,480] , 사이즈가 클수록 성능이 좋음
* sigma = 3.0
* batch size = 8
* shifting = True
* aid = false , 팔이 가려져서 예측을 못하는 경우가 있음
* startify = True with direction , 각 사람별로 A,B,C,D,E 카메라로 촬영함. 그래서 카메라 별로 나누어 생각
* use different joints weights = False , 손목 눈코입 발목이 가려지는 경우가 많아서 해당 케이스에 강한 weight를 줘봤는데 똑같음

[전처리]
* 강한 augmentation
* 움직임에 대한 흐려짐 현상 해결을 위해 motionblur, blur, imagecompression, gaussianblur 사용
* 여러 색상에 대한 강인함을 위해 channelshuffle, huesaturation, rgbshift 사용
* 테스트 데이터의 경우 yolov5와 각 자세에 따라서 다른 가로 세로 비율을 주어 affine transformation 수행

[후처리]
* Dark-pose 사용

## 코드

```shell
!pip install -U https://github.com/albu/albumentations
```



### Test with YOLOv5

이 코드의 경우 HRNet을 위주로 사용하고, Test의 경우에 yolov5 모델을 써서 사람을 detection한 다음에 해당 bbox의 중점을 활용해서 추론해줬음
* yolov5를 통해서 사람의 위치를 뽑아내고, 해당 중점을 기준으로 900x900을 뽑아내었음
*  해당 bbox를 바로 사용하여 test했을때 결과가 그렇게 좋지 않아서 해당 bbox(900x900)의 중점만 활용해서 누워 있는 경우, 서 있는 경우, 앉아 있는 경우를 구분하여 따로 잘라내었음(test dataset에서 설명)

```shell
!pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
```

```python
test_df = pd.read_csv(os.path.join(main_dir,'sample_submission.csv'))
yolov5 = torch.hub.load('ultralytics/yolov5','yolov5x',pretrained = True)
yolov5.eval()
test_data = {'path':[],'x1':[],'y1':[],'x2':[],'y2':[]}

total_test_imgs = []
for i in range(len(test_df)):
  total_test_imgs.append(os.path.join(test_img_path, test_df.iloc[i, 0]))


for idx, path in tqdm(enumerate(total_test_imgs)):
  w, h = 900, 900
  offset = np.array([w//2, h//2])

  img = cv2.imread(path)[:, :, ::-1] # img 읽어오기
  centre = np.array(img.shape[:-1])//2 # center 좌표
  x1,y1 = centre - offset
  x2,y2 = centre + offset

  with torch.no_grad():
    cropped_img = img[x1:x2, y1:y2, :]
    results = yolo_v5([cropped_img])

  cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
  try:
    for i in range(len(results.xyxy[0])):
      xyxy = results.xyxy[0][i].detach().cpu().numpy()
      cropped_centre = np.array([(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2], dtype=np.float32)
      box_w = (xyxy[2]-xyxy[0])/2 * 1.2
      box_h = (xyxy[3]-xyxy[1])/2 * 1.2

      new_x1 = np.clip(int(cropped_centre[0] - box_w), 0, img.shape[1])
      new_x2 = np.clip(int(cropped_centre[0] + box_w), 0, img.shape[1])
      new_y1 = np.clip(int(cropped_centre[1] - box_h), 0, img.shape[0])
      new_y2 = np.clip(int(cropped_centre[1] + box_h), 0, img.shape[0])

      if int(xyxy[-1]) == 0:
        new_x1 += y1
        new_x2 += y1
        new_y1 += x1
        new_y2 += x1

        test_data['path'].append(path)
        test_data['x1'].append(new_x1)
        test_data['y1'].append(new_y1)
        test_data['x2'].append(new_x2)
        test_data['y2'].append(new_y2)
        
  except Exception as e:
    print("Skip")


test_df = pd.DataFrame(data=test_data)
test_df.to_csv(os.path.join(main_dir, 'test_bbox.csv'), index=False)

cfg = SingleModelTestConfig(input_size=[640, 480], target_type='gaussian')
predictions = bbox_test(cfg, yaml_name='for_test.yaml', filp_test=True, debug=False)

preds = []
for prediction in predictions:
    row = []
    for x, y in zip(prediction[:, 0], prediction[:, 1]):
        row.append(x)
        row.append(y)
    preds.append(row)
preds = np.array(preds)

# submission ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

submission_path = os.path.join(data_dir, 'sample_submission.csv')
submission = pd.read_csv(submission_path)

save_dir = os.path.join(main_dir, "submissions")
save_path = os.path.join(save_dir, 'submission.csv')
submission.iloc[:, 1:] = preds
submission.to_csv(save_path, index=False)

```


































































































https://dacon.io/competitions/official/235701/codeshare/2490?page=1&dtype=recent

HRNet + Detectron2
https://dacon.io/competitions/official/235701/codeshare/2500?page=1&dtype=recent

