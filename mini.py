import numpy as np
import cv2
import sklearn.ensemble
import os
import sklearn
import sklearn.linear_model
import sklearn.tree
ori="C:\\Users\Ritvik\\Desktop\\minidataset"
x=[]
y=[]
x_test=[]
y_test=[]
os.chdir(ori)
dirs=os.listdir()
for dir in dirs:
    for file in os.listdir(dir):
        f=os.fsdecode(file)
        img = cv2.imread(dir+'\\'+file,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(480,360))
        img=img.reshape(1,-1)
        x.append(img)
        y.append(dir)
x = np.array(x).reshape(len(y),-1)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
clf=sklearn.ensemble.RandomForestClassifier(random_state=0)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
a=cv2.imread('C:\\Users\Ritvik\\Desktop\\b212.jpg',cv2.IMREAD_GRAYSCALE)
a = cv2.resize(a,(480,360))
cv2.imshow('image',a)
cv2.waitKey(0)
cv2.destroyAllWindows()

a=a.reshape(1,-1)
print(clf.predict(a))
