import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
from sigmoid import sigmoid
from softmax import softmax

def img_show(img):
    pil_img =Image.fromarray(np.uint8(img))
    pil_img.show()
def get_data():
    (x_train, t_train), (x_test,t_test)=load_mnist(flatten=True,normalize=False)
    return x_test,t_test
def init_network():
    with open("sample_weight.pkl","rb") as f:
        network = pickle.load(f)


    return network

def predict(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax(a3)

    return y
#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=False)
# flatten : 일차원 배열로 펴기
# x_train 일차원 으로 편 픽셀(784) 몇만개
# t_train : 정답지
data=get_data()
img = data[0][1]
label=data[1][1]
print(img.shape)
print(label)
img_show(img.reshape(28,28))