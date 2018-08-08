import numpy as np
import sigmoid
'''
def init_network():
    network={}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network
    
def identity(x):
    return x

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1)+b1
    z1 = sigmoid.sigmoid(a1)
    a2 = np.dot(z1, w2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)+b3
    y = identity(a3)
    return y
network =init_network()
x=np.array([1.0,0.5])
y=forward(network, x)
print(y)
'''
'''
X = np.array([1.0, 0.5]) # 입력
W1 = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) +B1
Z1 = sigmoid.sigmoid(A1)
#print(A1)
#print(Z1)

W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

A2 = np.dot(Z1,W2)+B2
Z2=sigmoid.sigmoid(A2)

print(A2)
print(Z2)
'''
def identity(x):
    return (x)

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.2],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def forward(network , x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1)+b1
    z1 = sigmoid.sigmoid(a1)
    a2 = np.dot(z1, W2)+b2
    z2 = sigmoid.sigmoid(a2)
    a3 = np.dot(z2, W3)+b3
    y = identity(a3)
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)