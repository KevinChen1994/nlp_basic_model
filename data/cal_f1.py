# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/9/16 14:13

'''
微平均和宏平均的计算方法，每一种提供了两个计算方法。
微平均就是先计算出全部的TP/TN/FN，然后统一计算precision和recall，最后算f1值。
宏平均就是计算每一个数据的precision和recall，最后计算平均的f1。
'''

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# ys1：多提一个；ys2：漏提一个，多提一个；ys3：全对；ys4：漏提两个，多提两个
pred = {'ys1': ['ys1_n1', 'ys1_n2', 'ys1_n3', 'ys1_n5'], 'ys2': ['ys2_n1', 'ys2_n2', 'ys2_n4'],
        'ys3': ['ys3_n1', 'ys3_n2', 'ys3_n3', 'ys3_n4'], 'ys4': ['ys4_n1', 'ys4_n5', 'ys4_n6']}

true = {'ys1': ['ys1_n1', 'ys1_n2', 'ys1_n3'], 'ys2': ['ys2_n1', 'ys2_n2', 'ys2_n3'],
        'ys3': ['ys3_n1', 'ys3_n2', 'ys3_n3', 'ys3_n4'], 'ys4': ['ys4_n1', 'ys4_n2', 'ys4_n3']}

cand = {'ys1': ['ys1_n1', 'ys1_n2', 'ys1_n3', 'ys1_n4', 'ys1_n5'], 'ys2': ['ys2_n1', 'ys2_n2', 'ys2_n3', 'ys2_n4'],
        'ys3': ['ys3_n1', 'ys3_n2', 'ys3_n3', 'ys3_n4'],
        'ys4': ['ys4_n1', 'ys4_n2', 'ys4_n3', 'ys4_n4', 'ys4_n5', 'ys4_n6']}


def cal_micro_f1(pred, true):
    TP_ALL = 0
    FP_ALL = 0
    FN_ALL = 0
    for key in pred:
        TP = 0
        FP = 0
        FN = 0
        for i in pred[key]:
            if i in true[key]:
                TP += 1
            else:
                FP += 1
        for i in true[key]:
            if i not in pred[key]:
                FN += 1
        TP_ALL += TP
        FP_ALL += FP
        FN_ALL += FN
    micro_p = TP_ALL / (TP_ALL + FP_ALL)
    micro_r = TP_ALL / (TP_ALL + FN_ALL)
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r)
    print('micro_f1: {}'.format(micro_f1))


def cal_micro_f1_(pred, true):
    p_sum = 0
    r_sum = 0
    hits = 0
    for key in pred:
        p_sum += len(pred[key])
        r_sum += len(true[key])
        for i in pred[key]:
            if i in true[key]:
                hits += 1
    p = hits / p_sum if p_sum > 0 else 0
    r = hits / r_sum if r_sum > 0 else 0
    micro_f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
    print('micro_f1 {}'.format(micro_f1))


def cal_PR(ytrue, ypred):
    return precision_score(ytrue, ypred), recall_score(ytrue, ypred)


def cal_macro_f1(pred, true, cand):
    result = {}
    for key in pred:
        dic = {}
        for i in cand[key]:
            dic[i] = 0
        for i in true[key]:
            dic[i] = 1

        y_true = (list(dic.values()))

        for i in cand[key]:
            dic[i] = 0

        for i in pred[key]:
            dic[i] = 1
        y_pred = (list(dic.values()))
        result[key] = cal_PR(y_true, y_pred)

    # for ys in result:
    #     print(str(ys) + ':precision:' + str(result[ys][0]))
    #     print(str(ys) + ':recall:' + str(result[ys][1]))
    # print(result)¬
    p_sum = 0
    r_sum = 0
    for key in result:
        p_sum += result[key][0]
        r_sum += result[key][1]

    macro_f1 = (2 * p_sum * r_sum) / (len(result) * (p_sum + r_sum))

    print('macro_f1: {}'.format(macro_f1))


def cal_macro_f1_(pred, true):
    p_sum = 0
    r_sum = 0
    for key in pred:
        TP = 0
        FP = 0
        FN = 0
        for i in pred[key]:
            if i in true[key]:
                TP += 1
            else:
                FP += 1
        for i in true[key]:
            if i not in pred[key]:
                FN += 1
        p_sum += TP / (TP + FP) if (TP + FP) > 0 else 0
        r_sum += TP / (TP + FN) if (TP + FN) > 0 else 0
    macro_f1 = (2 * p_sum * r_sum) / ((p_sum + r_sum) * len(pred))
    print('macro_f1_: {}'.format(macro_f1))


if __name__ == '__main__':
    # cal_micro_f1(pred, true)
    # cal_micro_f1_(pred, true)
    # cal_macro_f1(pred, true, cand)
    cal_macro_f1_(pred, true)
