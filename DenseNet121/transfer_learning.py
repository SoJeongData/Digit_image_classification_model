# 학습된 모델 로드
model = load_model('/content/drive/MyDrive/save-model/02.15_model.h5')

# 설정 값 변경

train_libdatagen = ImageDataGenerator(rescale=1/255,
                                      shear_range=20,
                                      # rotation_range=20,
                                      # width_shift_range=0.1,
                                      # height_shift_range=0.1,
                                      # shear_range=0.1,
                                      # zoom_range=0.1,
                                      fill_mode='nearest'
                                      )

train_generator = train_libdatagen.flow_from_directory(directory=save_train_dir,
                                                       classes=[str(i) for i in range(10)],
                                                       target_size=(64, 64),
                                                       batch_size=256,
                                                       class_mode='categorical'
                                                       )
model_base.trainable = True

# 특정 레이어 이후에 동결 풀기
set_trainable = False # 플래그 생성
for layer in model_base.layers:
    if layer.name == 'conv5_block1_0_bn': # 이 레이어 이후의 레이어에 대해 동결풂
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
