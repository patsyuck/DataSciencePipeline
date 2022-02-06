# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:19:21 2021
Головний скрипт
@author: o.patsiuk
"""
from pipeline import Pipeline

if __name__ == '__main__':
    pipe = Pipeline(df='../../Scoring/Income_data/Features/2021_dev_train.csv', # датасет
        valid_data_path='../../Scoring/Income_data/Features/2021_dev_test.csv', # датасет для валідації
        problem='classification', # (classification, regression, clasterization)
        res = './results/LGB_multi', # шлях до папки, де зберігатимуться результати
        model='lgb', # модель, для якої підбиратимемо параметри (lgb, xgb, ...)
        model_type='multi', # тип моделі (binary, multi, ...)
        eval_metric='multi_error', # метрика для ранньої зупинки (auc, multi_error(lgb), merror(xgb), ...)
        metric='accuracy', # метрика для оцінки якості моделі (auc, accuracy, ...)
        minFeatures=48, # мінімальна кількість важливих фіч
        threshold=None, # порогове значення важливості фіч
        partFeatures=None, # відсоток відбору важливих фіч
        ids=['Phone_number', 'dt', 'salary_avg', 'abs_d_income'], # несуттєві колонки (ідентифікатори тощо)
        target='class', # цільова колонка (якщо є)
        method=None, # метод попереднього відбору важливих фіч
        is_weights=True, # чи використовуємо ваги
        n=100, # кількість прогонів параметрів
        cv=5, # кількість фолдів для крос-валідації
        rounds=50) # кількість раундів для ранньої зупинки
    if pipe.problem == 'classification':
        print('Розмірність вхідного датасету: ', pipe.data.shape)
        pipe.calculateIV()
        print('Важливість фіч:')
        print(pipe.allFeatures)
        pipe.getImportantFeatures()
        print('Важливі фічі: ', pipe.importantFeatures)
        pipe.getWeights()
        print('Ваги для класів: ', pipe.weights)
        pipe.getTrainVal()
        print('Розмірність датасету для тренування: ', pipe.X_train.shape)
        print('Розмірність датасету для валідації: ', pipe.X_val.shape)
        pipe.getParameters()
        print('--------------------')
        print('Оптимальні параметри: ', pipe.params)
        pipe.saveBestModel()
        print('Метрики моделі: ', pipe.metrics)
        pipe.saveConfusionMatrix()
        if pipe.type == 'binary':
            pipe.saveRocAucGini()
            pipe.savePrecisionRecall()
        pipe.saveFeatureImportance()
