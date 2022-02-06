# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:31:49 2021
Precision and recall graphics
@author: o.patsiuk
"""
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from scipy.stats.mstats import hmean

def PrecisionRecall(actual_train, predict_train, actual_test, predict_test, path):
    precision, recall, threshold = precision_recall_curve(actual_train, predict_train)
    precision2, recall2, threshold2 = precision_recall_curve(actual_test, predict_test)
    colors = ['red', 'darkorange', 'gold', 'darkgreen', 'blue']
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(16, 7))
    fig.suptitle('Графіки Precision-Recall',fontsize = 18, fontweight='bold')
    ax[0].plot(recall, precision, color='navy')
    for i in range(0, 5):
        ind = list(abs(threshold - (i+5)/10)).index(min(list(abs(threshold - (i+5)/10))))
        ax[0].scatter(recall[ind], precision[ind], color=colors[i], s=70, 
            label='t = {0}, precision = {1}, recall = {2}, f-score = {3}'.format(
            round(10*threshold[ind], 0)/10, round(precision[ind], 3), 
            round(recall[ind], 3), round(hmean([precision[ind], recall[ind]]), 3)))
    ax[0].set(title='Трейн', xlabel='Recall', ylabel='Precision')
    ax[1].plot(recall2, precision2, color='navy')
    for i in range(0, 5):
        ind2 = list(abs(threshold2 - (i+5)/10)).index(min(list(abs(threshold2 - (i+5)/10))))
        ax[1].scatter(recall2[ind2], precision2[ind2], color=colors[i], s=70, 
            label='t = {0}, precision = {1}, recall = {2}, f-score = {3}'.format(
            round(10*threshold2[ind2], 0)/10, round(precision2[ind2], 3), 
            round(recall2[ind2], 3), round(hmean([precision2[ind2], recall2[ind2]]), 3)))
    ax[1].set(title='Валідація', xlabel='Recall', ylabel='Precision')
    for i in [0, 1]:
        ax[i].legend(loc='lower left')
    fig.savefig(path)