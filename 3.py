from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

X, y = load_iris(return_X_y=True)

img=cv2.imread('d:/Dataset/road_dataset/images/0011.jpg')
shape=img.shape
img=img.reshape(img.shape[0]*img.shape[1],3)
#label=cv2.imread('data_dataset_voc/SegmentationClassPNG/18.png')
label=Image.open('d:/Dataset/road_dataset/SegmentationClassPNG/0011.png')
label=np.array(label)
label=label.reshape(label.shape[0]*label.shape[1])

clf = LogisticRegression(random_state=0,class_weight={0:1,1:4}).fit(img,label)
pred=clf.predict(img)
pred1=pred
pred=pred.reshape(shape[0],shape[1])
pred=pred*255
pred=cv2.resize(pred,(shape[1]//2,shape[0]//2))
cv2.imshow('pred',pred)
cv2.waitKey()
#print(clf.predict_proba(img[:10, :]))
#print(clf.score(X[:100,:], y[:100]))

# Step 4: Evaluate Performance
acc = metrics.accuracy_score(label, pred1)
precision = metrics.precision_score(label, pred1)
recall = metrics.recall_score(label, pred1)
f1 = metrics.f1_score(label, pred1)

print(f"Accuracy: {acc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Step 5: Plot PR Curve
precision, recall, thresholds = metrics.precision_recall_curve(label, clf.decision_function(img))
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Step 6: Plot ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(label, clf.decision_function(img))
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()