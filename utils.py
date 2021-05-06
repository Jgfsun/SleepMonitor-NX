# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from scipy import signal


class EpochBuffer:
    def __init__(self, fixed_length=400):
        self.buffer = []  # filter raw data
        self.labels = []
        self.tmp_data = []  # 3000
        self.buffer_2 = []  # down sample data
        self.sample_rate = 100
        self.fixed_list = [0 for i in range(fixed_length + 10)]
        self.blank = [0 for i in range(10)]

    def set(self, data, label, sample_rate):
        if len(self.buffer) == 0:
            self.buffer = self.buffer + data
        else:
            self.buffer = self.buffer + data[-100:]
        self.labels = self.labels + label
        self.tmp_data = data
        self.sample_rate = sample_rate
        return None

    def set_raw_data(self, data):
        self.buffer = self.buffer + data
        return None

    def set_data(self, data):
        self.buffer_2 = self.buffer_2 + data
        return None

    def set_label(self, label):
        self.labels = self.labels + label

    def get_raw_data(self, index):
        # 过滤前10s，防止初始干扰
        return self.buffer[index * 250 + 2500: index * 250 + 10000]

    def get_raw_data_state(self, index):
        if len(self.buffer) >= 2500 + index * 250 + 7500:
            return True
        else:
            return False

    def get_data(self, fixed_length, moved_length, f_m_ratio, i):
        f_m_ratio = int(f_m_ratio)
        a = i % f_m_ratio
        b = i // f_m_ratio * fixed_length
        y = self.buffer_2[b + moved_length * a: moved_length * (1 + a) + b] + self.blank
        self.fixed_list[moved_length * a:moved_length * (1 + a) + 10] = y
        return self.fixed_list, self.sample_rate

    # 如果buffer_2的数据 ≥ index * 100就返回True
    def get_data_state(self, index, moved_length):
        if len(self.buffer_2) <= 0 + int(index * moved_length):
            return False
        else:
            return True

    def get_label(self, index):
        return self.labels[index]

    def get_label_state(self, index):
        if len(self.labels) > index:
            return True
        else:
            return False

    def print_information(self):
        print(len(self.buffer), len(self.labels), self.sample_rate)

    def get_filter_rhythm(self, l_f=4, h_f=8, ):  # 1/2 * sample_rate = 50
        b, a = signal.butter(8, [l_f / 50.0, h_f / 50.0], btype='bandpass', analog=False)  # 4Hz-8Hz
        filter_data = signal.filtfilt(b, a, np.array(self.fixed_list))  # numpy.ndarray
        return filter_data.tolist()


epoch_buffer = EpochBuffer(400)


# 作用：输出以batch_size大小为一组的(x,y)输出结果是很多个batch
def batch_data(x, batch_size):
    print("the length of x and batch_size: ", len(x), batch_size)
    shuffle = np.random.permutation(len(x))  # 打乱排序，0,1,2……,x
    start = 0
    #     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start + batch_size]
        start += batch_size


def flatten(name, input_var):
    dim = 1
    for d in input_var.get_shape()[1:].as_list():
        dim *= d
    # print("this is dim", dim, name)
    output_var = tf.reshape(input_var, shape=[-1, dim], name=name)
    # shape(-1,dim)这里-1代表Numpy会根据dim维度(有多少列)自动求出-1这个维度(有多少行)的数量
    # 这个shape=[-1, dim]得到的是个一维的行向量dim列
    # print("look here *********************", input_var.get_shape(), input_var.get_shape()[1:])
    # print("look here #####################", output_var.get_shape(), output_var.get_shape()[1:])
    return output_var


def sample_arr(sample, channel=1):
    tmp = []
    for i in range(len(sample)):
        tmp.append(sample[i][channel - 1])
    return tmp


def filter(data):
    data = np.array(data)
    # filter_detrend = signal.detrend(data)   # baseline drift
    notch_b, notch_a = signal.iirnotch(0.4, 30.0)
    filter_data_1 = signal.filtfilt(notch_b, notch_a, data)
    b, a = signal.butter(8, [0.008, 0.8], btype='bandpass', analog=False)  # 1Hz-100Hz
    filter_data_2 = signal.filtfilt(b, a, filter_data_1)  # numpy.ndarray
    data_list = filter_data_2.tolist()  # list
    return data_list


def down_sample(data_list):
    """the data_list length is 7500(30s epoch)"""
    data = np.zeros(3000)
    matrix = np.array([[1.0 / 3, 0], [1.0 / 3, 0], [1.0 / 6, 1.0 / 6], [0, 1.0 / 3], [0, 1.0 / 3]])

    if len(data_list) < 7500:  # 7500
        data_list = data_list + [data_list[-1] for index in range(7500 - len(data_list))]

    data_arr = np.array(data_list)
    for i in range(1500):
        data[2 * i: 2 * i + 2] = np.around(np.dot(data_arr[5 * i: 5 * i + 5], matrix),
                                           3)  # Keep three significant digits
    data = np.reshape(data, (1, 3000))
    return data  # arrary


def print_n_samples_each_class(labels, classes):
    class_dict = dict(zip(range(len(classes)), classes))
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print ("{}: {}".format(class_dict[c], n_samples))
