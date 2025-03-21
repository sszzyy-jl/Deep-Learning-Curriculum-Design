# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,precision_recall_curve,roc_curve,auc
import cv2
from PIL import Image
import matplotlib.pyplot as plt

images = []
labels = []
for i in range(1,4):
    img = cv2.imread('./data_dataset_voc/JPEGImages/'+str(i)+'.jpg')
    img = img.reshape(img.shape[0]*img.shape[1],3)
    for j in range(len(img)):
        images.append(img[j])

for i in range(1,4):
    label = Image.open('./data_dataset_voc/SegmentationClassPNG/'+str(i)+'.png')
    label = np.array(label)
    label = label.reshape(label.shape[0] * label.shape[1])
    for j in range(len(label)):
        labels.append(label[j])
clf = LogisticRegression(random_state=0).fit(images, labels)
x_test = cv2.imread('./data_dataset_voc/JPEGImages/4.jpg')
x_test = x_test.reshape(x_test.shape[0]*x_test.shape[1],3)
y_test = Image.open('./data_dataset_voc/SegmentationClassPNG/4.png')
y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1])
predictions = clf.predict(x_test)

print("logistic的accuracy为："+ str(accuracy_score(y_test,predictions)))
print("logistic的查准率为："+ str(precision_score(y_test,predictions)))
print("logistic的查全率为："+ str(recall_score(y_test,predictions)))
print("logistic的f1为："+ str(f1_score(y_test,predictions)))
print(predictions)
# 绘制pr曲线
scores = clf.predict_proba(x_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, scores)
plt.plot(recall, precision, label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# 绘制roc曲线
fpr, tpr, thresholds = roc_curve(y_test, scores)
# 计算曲线下面积（AUC）
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.show()