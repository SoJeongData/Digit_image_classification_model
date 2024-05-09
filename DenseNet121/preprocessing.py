!pip install datasets==2.16.1 -q
!pip install Pillow -q

from datasets import load_dataset

train_dataset = load_dataset('svhn', 'full_numbers', split='train')
test_dataset = load_dataset('svhn', 'full_numbers', split='test')

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 이미지 자르기
image_list = []
label_list = []

for x in train_dataset:
    for i in range(len(x['digits']['bbox'])):
        x_min, y_min, width, height = x['digits']['bbox'][i]  # bounding box의 좌표를 가져옵니다.
        right = x_min + width
        lower = y_min + height
        image_list.append(x['image'].crop((x_min, y_min, right, lower)))
        label_list.append(x['digits']['label'][i])

# 이미지 저장 경로
save_train_dir = '/content/save/train-images'

for i, img in enumerate(image_list):
    # 라벨에 해당하는 디렉토리 경로
    label_dir = os.path.join(save_train_dir, str(label_list[i]))

    # 라벨 디렉토리가 없으면 생성
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # 이미지 파일 경로
    img_path = os.path.join(label_dir, f'image_{i}.jpg')

    # 이미지 파일 저장
    img.save(img_path)
  
# 테스트 이미지 자르기
image_test_list = []
label_test_list = []

for x in test_dataset:
    for i in range(len(x['digits']['bbox'])):
        x_min, y_min, width, height = x['digits']['bbox'][i]  # bounding box의 좌표를 가져옵니다.
        right = x_min + width
        lower = y_min + height
        image_test_list.append(x['image'].crop((x_min, y_min, right, lower)))
        label_test_list.append(x['digits']['label'][i])


# 이미지 저장 경로
save_test_dir = '/content/save/test-images'

for i, img in enumerate(image_test_list):
    # 라벨에 해당하는 디렉토리 경로
    label_dir = os.path.join(save_test_dir, str(label_test_list[i]))

    # 라벨 디렉토리가 없으면 생성
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # 이미지 파일 경로
    img_path = os.path.join(label_dir, f'image_{i}.jpg')

    # 이미지 파일 저장
    img.save(img_path)


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator를 사용하여 데이터 증강 설정
train_libdatagen = ImageDataGenerator(rescale=1/255,
                                      )

# 데이터 증강을 적용하고 배치 단위로 제공하는 제너레이터 생성
train_generator = train_libdatagen.flow_from_directory(directory=save_train_dir,
                                                       classes=[str(i) for i in range(10)],
                                                       target_size=(64, 64),
                                                       batch_size=256,
                                                       class_mode='categorical'
                                                       )

test_libdatagen = ImageDataGenerator(rescale=1/255,
                                     validation_split=0.1
                                     )

test_generator = test_libdatagen.flow_from_directory(directory=save_test_dir,
                                                     classes=[str(i) for i in range(10)],
                                                     target_size=(64, 64),
                                                     batch_size=256,
                                                     class_mode='categorical',
                                                     subset='validation'
                                                     )

