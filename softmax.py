import numpy as np
def softmax(x):
    c = np.max(x)
    exp_x=np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x/sum_exp_a
    return y