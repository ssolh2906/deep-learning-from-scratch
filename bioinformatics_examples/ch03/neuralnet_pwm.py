# coding: utf-8
import sys, os
import pickle

import numpy as np

from bioinformatics_examples.ch03.splice_junction_pwm import load_pwm
from common.functions import softmax


def get_pwm():
    (x_train, t_train), (x_test, t_test) = load_pwm()
    return x_test, t_test


def init_network():
    with open(os.path.dirname(__file__) + "/pwm_weights.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']

    a1 = np.dot(x, W1) + b1
    z1 = np.maximum(0, a1)  # ReLU activation
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    return y


x, t = get_pwm()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
