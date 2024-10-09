import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Input, Add
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
 
# 파일을 읽어서 클래스 딕셔너리 생성
def create_class_dict(file_path):
    class_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("class number"):
                parts = line.strip().split(", ")
                class_number = int(parts[0].split(" = ")[1])
                class_name = parts[1].split(" = ")[1]
                class_dict[class_name] = class_number
    print(class_dict)
    return class_dict
 
# 이미지 파일 로드 및 데이터셋 준비
def load_images_from_folder(folder, class_dict):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                img = img.resize((32, 32))
                img_array = np.array(img)
                class_name = filename.split("_")[1].split(".")[0]
                class_number = class_dict.get(class_name)
                if class_number is not None:
                    images.append(img_array)
                    labels.append(class_number)
    return np.array(images), np.array(labels)
 
# 클래스 딕셔너리 생성
file_path = "C:\\Users\\11\\OneDrive - 충북대학교\\바탕 화면\\학부 자료\\딥러닝이론과 실습\\텀프로젝트\\CIFAR100\\fine_label_names_shuffle.txt"
class_dict = create_class_dict(file_path)
 
# CIFAR-100 이미지 데이터 로드
train_folder = "C:\\Users\\11\\OneDrive - 충북대학교\\바탕 화면\\학부 자료\\딥러닝이론과 실습\\텀프로젝트\\CIFAR100\\train"
 
train_x, train_y = load_images_from_folder(train_folder, class_dict)
train_x = train_x.reshape(train_x.shape[0], 32, 32, 3).astype('float64') / 255
 
# 훈련 데이터와 검증 데이터로 분할
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
 
# 데이터 증강 설정
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
 
datagen.fit(train_x)
 
# Residual Block 정의
def residual_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding='same', strides=strides, activation='relu')(shortcut)
    x = Add()([x, shortcut])
    return x
 
# CNN 모델 정의
def create_model(lr, dropout_rate):
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)
 
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
 
    # 첫 번째 Residual Block
    x = residual_block(x, 32)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
 
    # 두 번째 Residual Block
    x = residual_block(x, 64)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
 
    # 세 번째 Residual Block
    x = residual_block(x, 128)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
 
    # 네 번째 Residual Block
    x = residual_block(x, 256)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
 
    # 전결합 레이어
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
 
    # 출력 레이어
    outputs = Dense(len(class_dict), activation='softmax')(x)
 
    # 모델 정의
    model = Model(inputs, outputs)
 
    # 모델 컴파일
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 모델 학습 함수 정의
def train_model(lr, dropout_rate, batch_size):
    model = create_model(lr, dropout_rate)
    checkpoint = ModelCheckpoint('./best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
 
    history = model.fit(
        datagen.flow(train_x, train_y, batch_size=int(batch_size)), 
        epochs=100, 
        validation_data=(val_x, val_y), 
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    val_accuracy = max(history.history['val_accuracy'])
    return val_accuracy
 
# Bayesian Optimization
def optimize_model():
    pbounds = {
        'lr': (1e-5, 1e-2),
        'dropout_rate': (0.1, 0.5),
        'batch_size': (32, 256)
    }
 
    optimizer = BayesianOptimization(
        f=train_model,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
 
    optimizer.maximize(init_points=2, n_iter=10)
    return optimizer
 
optimizer = optimize_model()
 
# 최적의 하이퍼파라미터 출력
print("Best hyperparameters found were:")
print(optimizer.max)
 
# 최적의 하이퍼파라미터로 모델 재학습
best_params = optimizer.max['params']
best_lr = best_params['lr']
best_dropout_rate = best_params['dropout_rate']
best_batch_size = int(best_params['batch_size'])
 
model = create_model(best_lr, best_dropout_rate)
checkpoint = ModelCheckpoint('./best_model_final.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
 
history = model.fit(
    datagen.flow(train_x, train_y, batch_size=best_batch_size), 
    epochs=100, 
    validation_data=(val_x, val_y), 
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1
)
 
# 테스트 데이터 로드 및 전처리
test_folder = "C:\\Users\\11\\OneDrive - 충북대학교\\바탕 화면\\학부 자료\\딥러닝이론과 실습\\텀프로젝트\\CIFAR100\\test"
test_x, test_y = load_images_from_folder(test_folder, class_dict)
test_x = test_x.reshape(test_x.shape[0], 32, 32, 3).astype('float64') / 255
 
# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(test_x, test_y)[1]))
 
# 검증셋과 학습셋의 오차를 저장
y_vloss = history.history['val_loss']
y_loss = history.history['loss']
 
# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')
 
# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()