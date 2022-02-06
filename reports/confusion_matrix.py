# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:54:58 2021
Confusion Matrix
@author: o.patsiuk
"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def ConfusionMatrix(actual_train, predict_train, actual_test, predict_test, path):
    conf_matr = confusion_matrix(actual_train, predict_train)
    conf_matr_2 = confusion_matrix(actual_test, predict_test)
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(16, 7))
    fig.suptitle('Confusion Matrices',fontsize = 18, fontweight='bold')
    sns.heatmap(conf_matr, annot=True, square=True, 
                cmap='Greens', cbar=False, fmt="d", ax=ax[0])
    sns.heatmap(conf_matr_2, annot=True, square=True, 
                cmap='Blues', cbar=False, fmt="d", ax=ax[1])
    ax[0].set(title='Трейн', xlabel='predict label', ylabel='true label')
    ax[1].set(title='Валідація', xlabel='predict label', ylabel='true label')
    fig.savefig(path)