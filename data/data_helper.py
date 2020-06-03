# -*- coding:utf-8 -*-
# author:chenmeng
# datetime:2020/3/25 17:21
# software: PyCharm
import random
import numpy as np


def data_split(data_dir, data_out):
    with open(data_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        train_set = []
        test_set = []
        val_set = []
        for i, line in enumerate(lines):
            if i < 2000:
                train_set.append(line)
            elif 2000 <= i < 2500:
                test_set.append(line)
            elif 2500 <= i < 2700:
                val_set.append(line)
            else:
                break
    with open(data_out + '/train.txt', 'w', encoding='utf-8') as f:
        for i in train_set:
            f.write(i)
    with open(data_out + '/test.txt', 'w', encoding='utf-8') as f:
        for i in test_set:
            f.write(i)
    with open(data_out + '/val.txt', 'w', encoding='utf-8') as f:
        for i in val_set:
            f.write(i)



def data_analysis(data_dir):
    with open(data_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        len_ = []
        dict_ = {}
        for line in lines:
            label = line.split('\t')[0]
            if label not in dict_:
                dict_[label] = 1
            else:
                dict_[label] += 1
            content = line.split('\t')[1]
            len_.append(len(content))
        temp = np.array(len_)
        print(dict_)
        print('AVG:', np.average(temp))
        print('MEDIAN:', np.median(temp))
        print('MAX:', np.max(temp))
        print('MIN:', np.min(temp))


if __name__ == '__main__':
    # data_split('./cnews/cnews.train.txt', './small_cnews/')
    data_analysis('./small_cnews/train.txt')
