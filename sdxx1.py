import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

def unet(input_shape):
    inputs = Input(input_shape)

    # 下采样路径
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # 上采样路径
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)

    # 输出层
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    dice_score = (2.0 * intersection + 1e-5) / (2.0 *np.sum(y_true) + np.sum(y_pred) + 1e-5)
    return dice_score

# 读取图像
img = cv2.imread('d:/Dataset/road_dataset/JPEGImages/0011.jpg')
# 调整图像大小
img = cv2.resize(img, (256, 256))

# 读取标签
label = Image.open('d:/Dataset/road_dataset/SegmentationClassPNG/0011.png')
label = np.array(label)
label = cv2.resize(label, (256, 256))

# 构建训练数据集和测试数据集
train_data = []
test_data = []
for j in range(50000):
    train_data.append((img, label))  # 将图像和标签作为一组数据放入训练数据集
    test_data.append((img, label))  # 将图像和标签作为一组数据放入测试数据集

# 随机选择训练数据和测试数据
np.random.shuffle(train_data)
np.random.shuffle(test_data)

# 提取训练数据和测试数据
x_train = np.array([data[0] for data in train_data[:1000]])
y_train = np.array([data[1] for data in train_data[:1000]])

x_test = np.array([data[0] for data in test_data[:1000]])
y_test = np.array([data[1] for data in test_data[:1000]])

# 定义输入的图像尺寸
input_shape = (256, 256, 3)

# 创建输入层
inputs = Input(input_shape)
# 创建UNet模型
with tf.device('/device:GPU:0'):
    model = unet(input_shape)

# 编译模型
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # 训练模型
# model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test))
# # 在测试集上评估模型
# score = model.evaluate(x_test, y_test)
with tf.device('/device:GPU:0'):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=16, epochs=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)
# 可视化分割结果
predictions = model.predict(x_test)
