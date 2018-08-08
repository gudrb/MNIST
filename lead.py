import numpy as np
import sys
import os
from array import array

from struct import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fp_image = open('train-images.idx3-ubyte','rb')
fp_label = open('train-labels.idx1-ubyte','rb')

img = np.zeros((28,28))
lb1=[[],[],[],[],[],[],[],[],[],[]]
d=0
l=0
index =0

s = fp_image.read(16)
l = fp_label.read(8)

k=0
while True:
    s = fp_image.read(784)
    l = fp_label.read(1)

    if not s:
        break
    if not l:
        bread;
    index = int(l[0])
    #print(k,":",index)

    #unpack

    img = np.reshape( unpack(len(s)*'B',s), (28,28)) # 바이너리 파일을 풀어줌
    lb1[index].append(img) # 각 숫자영역 별로 해당 이미지를 추가
    k = k+1

plt.imshow(img,cmap = cm.binary) #img may be an array or a PIL image
#아마 color image 를 binary로 함으로써 흑색 백색으로 이미지를 표현하는거 같음
#plt.show()

#print(lb1[1]) 1에 해당하는 img 들이 잘 들어가 있나 확인

