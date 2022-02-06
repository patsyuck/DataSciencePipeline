# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:03:59 2021
ROC AUC and Gini graphics
@author: o.patsiuk
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
from sklearn.metrics import roc_auc_score, roc_curve

def RocAucGini(actual_train, predict_train, actual_test, predict_test, path):
    # train
    aucroc = roc_auc_score(actual_train, predict_train)
    gini = 2 * roc_auc_score(actual_train, predict_train) - 1
    fpr, tpr, t = roc_curve(actual_train, predict_train)
    data = zip(actual_train, predict_train)
    sorted_data = sorted(data, key=lambda d: d[1], reverse=True)
    sorted_actual = [d[0] for d in sorted_data]
    cumulative_actual = np.cumsum(sorted_actual) / sum(actual_train)
    cumulative_index = np.arange(1, len(cumulative_actual)+1) / len(predict_train)
    cumulative_actual_perfect = np.cumsum(sorted(actual_train, reverse=True)) / sum(actual_train)
    x_values = [0] + list(cumulative_index)
    y_values = [0] + list(cumulative_actual)
    y_values_perfect = [0] + list(cumulative_actual_perfect)
    S_pred = quad(interp1d(x_values, y_values), 0, 1, points=None)[0] - 0.5
    S_actual = quad(interp1d(x_values, y_values_perfect), 0, 1, points=None)[0] - 0.5
    # validation
    aucroc2 = roc_auc_score(actual_test, predict_test)
    gini2 = 2 * roc_auc_score(actual_test, predict_test) - 1
    fpr2, tpr2, t2 = roc_curve(actual_test, predict_test)
    data2 = zip(actual_test, predict_test)
    sorted_data2 = sorted(data2, key=lambda d: d[1], reverse=True)
    sorted_actual2 = [d[0] for d in sorted_data2]
    cumulative_actual2 = np.cumsum(sorted_actual2) / sum(actual_test)
    cumulative_index2 = np.arange(1, len(cumulative_actual2)+1) / len(predict_test)
    cumulative_actual_perfect2 = np.cumsum(sorted(actual_test, reverse=True)) / sum(actual_test)
    x_values2 = [0] + list(cumulative_index2)
    y_values2 = [0] + list(cumulative_actual2)
    y_values_perfect2 = [0] + list(cumulative_actual_perfect2)
    S_pred2 = quad(interp1d(x_values2, y_values2), 0, 1, points=None)[0] - 0.5
    S_actual2 = quad(interp1d(x_values2, y_values_perfect2), 0, 1, points=None)[0] - 0.5
    # graphics
    fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(15, 5))
    fig.suptitle("Графіки ROC та Lift-кривих",fontsize = 18, fontweight='bold')
    ax[0].plot([0]+fpr2.tolist(), [0]+tpr2.tolist(), lw = 2, color = 'orange', 
               label='ROC-крива валідації (AUC = %0.3f)' % aucroc2)
    ax[0].plot([0]+fpr.tolist(), [0]+tpr.tolist(), lw = 2, color = 'red', 
               label='ROC-крива трейну (AUC = %0.3f)' % aucroc)
    ax[0].fill_between([0]+fpr.tolist(), [0]+tpr.tolist(), color = 'red', alpha=0.1)
    ax[0].fill_between([0]+fpr2.tolist(), [0]+tpr2.tolist(), color = 'orange', alpha=0.1)
    ax[0].set(title='ROC-криві', xlabel='False Positive Rate', 
              ylabel='True Positive Rate', xlim=(-0.01, 1), ylim=(0, 1.01))
    ax[1].plot(x_values, y_values, lw = 2, color = 'blue', 
               label='навчена модель (Gini_m = %0.3f)' % S_pred)
    ax[1].plot(x_values, y_values_perfect, lw = 2, color = 'green', 
               label='ідеальна модель (Gini_p = %0.3f)' % S_actual)
    ax[1].fill_between(x_values, x_values, y_values_perfect, color = 'green', alpha=0.1)
    ax[1].fill_between(x_values, x_values, y_values, color = 'blue', alpha=0.1)
    ax[1].text(0.4,0.2,'Gini = {:0.3f}'.format(gini),fontsize = 20)
    ax[1].set(title='Lift-криві для трейну', xlabel="Кумулятивна частка об'єктів", 
              ylabel='Кумулятивна частка істинних класів', xlim=(-0.01, 1), ylim=(0, 1.01))
    ax[2].plot(x_values2, y_values2, lw = 2, color = 'blue', 
               label='навчена модель (Gini_m = %0.3f)' % S_pred2)
    ax[2].plot(x_values2, y_values_perfect2, lw = 2, color = 'green', 
               label='ідеальна модель (Gini_p = %0.3f)' % S_actual2)
    ax[2].fill_between(x_values2, x_values2, y_values_perfect2, color = 'green', alpha=0.1)
    ax[2].fill_between(x_values2, x_values2, y_values2, color = 'blue', alpha=0.1)
    ax[2].text(0.4,0.2,'Gini = {:0.3f}'.format(gini2),fontsize = 20)
    ax[2].set(title='Lift-криві для валідації', xlabel="Кумулятивна частка об'єктів", 
              ylabel='Кумулятивна частка істинних класів', xlim=(-0.01, 1), ylim=(0, 1.01))
    for i in range(0,3):
        ax[i].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
        ax[i].legend(loc="lower right")
    fig.savefig(path)
