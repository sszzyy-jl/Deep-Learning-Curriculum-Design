# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# images = []
# labels = []
# for i in range(3):
#     img = cv2.imread('d:/Dataset/road_dataset/JPEGImages/00'+str(i)+'1.jpg')
#     img = img.reshape(img.shape[0]*img.shape[1],3)
#     # for j in range(len(img)):
#     for j in range(50000):
#         images.append(img[j])
#
# for i in range(3):
#     label = Image.open('d:/Dataset/road_dataset/SegmentationClassPNG/00'+str(i)+'1.png')
#     label = np.array(label)
#     label = label.reshape(label.shape[0] * label.shape[1])
#     # for j in range(len(label)):
#     for j in range(50000):
#         labels.append(label[j])
from sklearn.metrics import precision_score



images = []
labels = []

img = cv2.imread('d:/Dataset/road_dataset/JPEGImages/0011.jpg')
img = img.reshape(img.shape[0] * img.shape[1], 3)
for j in range(50000):
    images.append(img[j])
label = Image.open('d:/Dataset/road_dataset/SegmentationClassPNG/0011.png')
label = np.array(label)
label = label.reshape(label.shape[0] * label.shape[1])
for j in range(50000):
    labels.append(label[j])

train_indices = np.random.choice(len(images), size=1000, replace=False)
test_indices = np.random.choice(len(labels), size=1000, replace=False)

x_train = np.array(images)[train_indices]
y_train = np.array(labels)[train_indices]

x_test = np.array(images)[test_indices]
y_test = np.array(labels)[test_indices]

# 训练模型
kernel_type = 'linear'  # 选择核函数
clf = svm.SVC(kernel=kernel_type, probability=True)
clf.fit(x_train, y_train)

# 在测试集上进行预测和评估
predictions = clf.predict(x_test)

print("SVM的accuracy为：" + str(metrics.accuracy_score(y_test, predictions)))
print("SVM的查准率为：" + str(metrics.precision_score(y_test, predictions, average='weighted', zero_division=1)))
print("SVM的查全率为：" + str(metrics.recall_score(y_test, predictions, average='weighted')))
print("SVM的f1为：" + str(metrics.f1_score(y_test, predictions, average='weighted')))

# 绘制pr曲线
scores = clf.predict_proba(x_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, scores)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# 绘制roc曲线
fpr, tpr, thresholds = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.show()