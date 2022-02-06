# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:44:53 2021
Визначає методи пайплайну
@author: o.patsiuk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, pickle, shap, warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score 
#, precision_score, recall_score, f1_score, classification_report, auc
from hyperopt import hp, tpe, Trials, space_eval
from hyperopt.fmin import fmin
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
# modules import
from get_features.woe_iv import iv
from reports.confusion_matrix import ConfusionMatrix
from reports.roc_auc_gini import RocAucGini
from reports.precision_recall import PrecisionRecall

warnings.filterwarnings("ignore")

space_lgb = {
    'boosting_type': 'gbdt', # by default
    'importance_type': 'gain', # not by default!
    #'objective': by default 'regression' for Regressor, 'binary' or 'multiclass' for Classifier
    'max_depth': hp.choice('max_depth', np.arange(3, 31, dtype = int)), # must be small enough (between 1 and 10)
    'num_leaves': hp.choice('num_leaves', np.arange(5, 101, dtype = int)), # less then 2^max_depth
    'min_child_samples': hp.choice('min_child_samples', np.arange(20, 401, 5, dtype = int)), # 100-1000 for big data
    'subsample_for_bin': hp.choice('subsample_for_bin', np.arange(20000, 300001, 20000, dtype = int)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1), # L1-regularization
    'reg_lambda': hp.uniform('reg_lambda', 0, 1), # L2-regularization
    'learning_rate': hp.loguniform('learning_rate', np.log(0.05), np.log(0.25)), # greater value for greater learning speed
    #'min_child_weight': hp.loguniform('min_child_weight', np.log(0.0005), np.log(1)), # sum of the second derivatives of the loss function
    'n_estimators': 5000, # always early_stopping_rounds must be used!
    'random_state': 13, # do not have to tune
    'n_jobs': -1 # do not have to tune
}

space_xgb = {
    'booster': 'gbtree',
    #'objective': by default 'reg:squarederror' for Regressor, 'binary:logistic' or 'multi:softprob' for Classifier
    'max_depth': hp.choice('max_depth', np.arange(3, 31, dtype = int)), # must be small enough (between 1 and 10)
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel': 1, # hp.uniform('colsample_bylevel', 0.5, 1), # long time to learn
    'colsample_bynode': 1, # hp.uniform('colsample_bynode', 0.5, 1), # long time to learn
    'reg_alpha': hp.uniform('reg_alpha', 0, 1), # L1-regularization
    'reg_lambda': hp.uniform('reg_lambda', 0, 1), # L2-regularization
    'gamma': hp.loguniform('gamma', np.log(0.001), np.log(5)), # value 20 is very high
    'learning_rate': hp.loguniform('learning_rate', np.log(0.05), np.log(0.25)), # greater value for greater learning speed
    #'min_child_weight': hp.loguniform('min_child_weight', np.log(0.0005), np.log(1)), # sum of the second derivatives of the loss function
    #'max_delta_step': hp.choice('max_delta_step', np.arange(0, 10, dtype = int)),
    #'scale_pos_weight': 30, # = sum(negative)/sum(positive), or count("0")/count("1")
    'n_estimators': 5000, # always early_stopping_rounds must be used!
    'random_state': 13, # do not have to tune
    'n_jobs': -1 # do not have to tune
}

class Pipeline():
    
    # ініціалізує клас Pipeline
    def __init__(self, df, res, problem, model, model_type, eval_metric, metric, minFeatures, 
            valid_data_path=None, threshold=None, partFeatures=None, ids=None, target=None,
            method=None, is_weights=True, n=None, cv=None, rounds=None):
        self.data = pd.read_csv(df) # тут зберігаємо весь датасет
        self.valid_data_path = valid_data_path # тут зберігаємо шлях до датасету для валідації
        self.ids = ids # несуттєві колонки (ідентифікатори тощо)
        self.target = target # цільова колонка (якщо є)
        self.is_weights = is_weights # чи використовуємо ваги
        self.weights = None # ваги для незбалансованих класів (словник)
        self.X_train = None # датасет для навчання моделі
        self.X_val = None # датасет для валідації моделі
        self.y_train = None # цільова колонка для навчання моделі
        self.y_val = None # цільова колонка для валідації моделі
        self.n = n # кількість прогонів параметрів
        self.cv = cv # кількість фолдів для крос-валідації
        self.rounds = rounds # кількість раундів для ранньої зупинки
        self.cv_results = {} # мапінг скорів крос-валідації в к-ть естіматорів
        self.method = method # метод попереднього відбору важливих фіч
        self.partFeatures = partFeatures # відсоток відбору важливих фіч
        self.minFeatures = minFeatures # мінімальна кількість важливих фіч
        self.threshold = threshold # порогове значення важливості фіч
        self.allFeatures = None # таблиця фіч, відсортованих за спаданням значущості
        self.importantFeatures = None # список найважливіших фіч
        self.problem = problem # задача (класифікація, регресія, кластеризація тощо)
        self.model = model # модель, для якої підбиратимемо параметри
        self.type = model_type # тип моделі (напр., бінарна чи мультикласова)
        self.eval_metric = eval_metric # метрика для ранньої зупинки
        self.metric = metric # метрика для оцінки якості моделі
        self.params = {} # оптимальні параметри моделі (словник)
        self.best_model = None # модель, натренована при оптимальних параметрах
        self.mainFeatures = None # список фіч, найважливіших для побудованої моделі
        self.metrics = {} # метрики побудованої моделі (словник)
        self.explainer = None # SHAP-explainer моделі
        self.results_path = res # шлях до папки, де зберігатимуться результати
        
    # 1. Методи попередньої обробки даних (очистка, перетворення фіч, рядки тощо)
    
    # 2. Методи попереднього відбору важливих фіч
    # будь-який з цих методів формує датафрейм з колонками feature та importance
    # фічі мають бути відсортовані в порядку спадання importance
    
    # розраховує Importance Values
    def calculateIV(self):
        res = iv(data=self.data.drop(self.ids, axis=1), target_col=self.target, 
                 max_bin=20, force_bin=3)
        self.allFeatures = res[1]
    
    # 3. Методи підбору гіперпараметрів моделі
    def getParameters(self):
        # спочатку визначаємо простір параметрів
        if self.model == 'lgb':
            space = space_lgb.copy()
            if self.type == 'binary':
                space['objective'] = 'binary'
                space['min_child_weight'] = hp.loguniform('min_child_weight', np.log(0.0005), np.log(0.1))
            elif self.type == 'multiclass':
                space['objective'] = 'multiclass'
                space['min_child_weight'] = hp.loguniform('min_child_weight', np.log(0.0005), np.log(0.1))
            elif self.type == 'regression':
                space['objective'] = 'regression'
                space['min_child_weight'] = hp.choice('min_child_weight', np.arange(1, 21, dtype = int))
            # для LGBM додаємо ваги відразу в параметри
            if self.is_weights:
                space['class_weight'] = self.weights
        elif self.model == 'xgb':
            space = space_xgb.copy()
            if self.type == 'binary':
                space['objective'] = 'binary:logistic'
                space['min_child_weight'] = hp.loguniform('min_child_weight', np.log(0.0005), np.log(0.1))
            elif self.type == 'multiclass':
                space['objective'] = 'multi:softmax'
                space['min_child_weight'] = hp.loguniform('min_child_weight', np.log(0.0005), np.log(0.1))
            elif self.type == 'regression':
                space['objective'] = 'reg:squarederror'
                space['min_child_weight'] = hp.choice('min_child_weight', np.arange(1, 21, dtype = int))
        else: # додати інші моделі
            pass
                
        # далі до функції, яка буде мінімізуватися, передаємо тільки параметри
        def scoring(space):
            if self.model == 'lgb':
                if self.problem == 'classification':
                    model = LGBMClassifier(**space)
                elif self.problem == 'regression':
                    model = LGBMRegressor(**space) 
            elif self.model == 'xgb':
                if self.problem == 'classification':
                    model = XGBClassifier(use_label_encoder=False, **space)
                elif self.problem == 'regression':
                    model = XGBRegressor(use_label_encoder=False, **space)
            else: # додати інші моделі
                pass
            print('--------------------')
            print('Parameters testing: ', space)
            folds = StratifiedKFold(n_splits = self.cv, shuffle = True, random_state = 13)
            scores = []
            estimators = []
            #for i, (train_index, test_index) in enumerate(folds.split(self.X_train, self.y_train)):
            for train_index, test_index in folds.split(self.X_train, self.y_train):
                X_train, X_test = self.X_train.iloc[train_index, :], self.X_train.iloc[test_index, :]
                y_train, y_test = self.y_train[train_index], self.y_train[test_index]
                if self.model == 'lgb' or self.model == 'xgb' and not self.is_weights:
                    model.fit(X_train, y_train,
                        early_stopping_rounds = self.rounds, verbose = self.rounds,
                        eval_set = [(X_train, y_train), (X_test, y_test)], 
                        eval_metric = self.eval_metric)
                elif self.model == 'xgb' and self.is_weights: # для XGBoost додаємо ваги при навчанні
                    model.fit(X_train, y_train, sample_weight=y_train.map(self.weights),
                        #eval_sample_weight=y_test.map(self.weights),
                        early_stopping_rounds = self.rounds, verbose = self.rounds,
                        eval_set = [(X_train, y_train), (X_test, y_test)], 
                        eval_metric = self.eval_metric)
                else: # додати інші моделі
                    pass
                if self.model == 'lgb':
                    estimators.append(model.best_iteration_)
                elif self.model == 'xgb':
                    estimators.append(model.best_iteration)
                else: # додати інші моделі
                    pass
                if self.metric == 'auc':
                    pred = model.predict_proba(X_test)[:, 1]
                    score = roc_auc_score(y_test, pred)
                elif self.metric == 'accuracy':
                    pred = model.predict(X_test)
                    score = accuracy_score(y_test, pred)
                else: # додати інші метрики
                    pass
                scores.append(score)
            #mean_score = round(np.mean(scores), 4)
            mean_score = np.mean(scores)
            #print('Estimators: {0}, mean {1}: {2}'.format(
            #    estimators, self.metric, mean_score))
            self.cv_results[mean_score] = estimators
            return -mean_score
        
        trials = Trials()
        best = fmin(fn = scoring, space = space, algo = tpe.suggest, 
                    max_evals = self.n, trials = trials)
        self.params = space_eval(space, best)
        self.metrics['score_mean_cv'] = max(self.cv_results.keys())
        if self.model in {'lgb', 'xgb'}:
            self.params['n_estimators'] = max(self.cv_results[max(self.cv_results.keys())])
    
    # 4. Побудова моделі на оптимальних параметрах
    def saveBestModel(self):
        if self.model == 'lgb':
            if self.problem == 'classification':
                model = LGBMClassifier(**self.params)
            elif self.problem == 'regression':
                model = LGBMRegressor(**self.params) 
        elif self.model == 'xgb':
            if self.problem == 'classification':
                model = XGBClassifier(use_label_encoder=False, **self.params)
            elif self.problem == 'regression':
                model = XGBRegressor(use_label_encoder=False, **self.params)
        else: # додати інші моделі
            pass
        self.best_model = model.fit(self.X_train, self.y_train)
        if self.metric == 'auc':
            pred = self.best_model.predict_proba(self.X_train)[:, 1]
            self.metrics['score_train'] = roc_auc_score(self.y_train, pred)
            pred = self.best_model.predict_proba(self.X_val)[:, 1]
            self.metrics['score_val'] = roc_auc_score(self.y_val, pred)
        elif self.metric == 'accuracy':
            pred = self.best_model.predict(self.X_train)
            self.metrics['score_train'] = accuracy_score(self.y_train, pred)
            pred = self.best_model.predict(self.X_val)
            self.metrics['score_val'] = accuracy_score(self.y_val, pred)
        else: # додати інші метрики
            pass
        model_features = (self.best_model, list(self.X_train.columns))
        with open(os.path.join(self.results_path, 
                '{0}_{1}-{2}_{3}.pickle'.format(self.model.upper(), 
                self.metric, round(self.metrics['score_val'], 4), 
                datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))), 'wb') as f:
            pickle.dump(model_features, f)
    
    # 5. Допоміжні методи (побудова графіків, запис у файл тощо)
    
    # формує список найважливіших фіч за важливістю і заданими параметрами
    def getImportantFeatures(self):
        if (self.threshold == None) and (self.partFeatures == None):
            n = self.minFeatures
        elif self.threshold != None:
            n = max(self.minFeatures, 
                len(self.allFeatures[self.allFeatures['importance'] > self.threshold]))
        else:
            n = max(self.minFeatures, round(self.partFeatures * len(self.allFeatures)))
        self.importantFeatures = list(self.allFeatures['feature'][:n])
        
    # формує ваги для незбалансованих класів
    def getWeights(self):
        cnt = 100 * self.data[self.target].value_counts(normalize=True)
        lst = list(cnt)
        lst2 = [lst[0] / x for x in lst]
        lst3 = [round(x / sum(lst2), 3) for x in lst2]
        self.weights = dict(zip(list(cnt.index), lst3))
    
    # розбиває датасет на 2 частини -- для навчання та валідації моделі
    def getTrainVal(self):
        if self.valid_data_path:
            data_valid = pd.read_csv(self.valid_data_path)
            self.X_train, self.y_train = self.data[self.importantFeatures], self.data[self.target]
            self.X_val, self.y_val = data_valid[self.importantFeatures], data_valid[self.target]
            del data_valid
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.data[self.importantFeatures], self.data[self.target],
                test_size = 1/(self.cv + 1), random_state = 13, shuffle = True,
                stratify = self.data[self.target])
            self.X_train = self.X_train.reset_index(drop=True)
            self.X_val = self.X_val.reset_index(drop=True)
            self.y_train = self.y_train.reset_index(drop=True)
            self.y_val = self.y_val.reset_index(drop=True)
            #self.X_train = self.X_train.drop(self.ids, axis=1)
            #self.X_val = self.X_val.drop(self.ids, axis=1)
        
    # формує матрицю помилок
    def saveConfusionMatrix(self):
        ConfusionMatrix(actual_train = self.y_train, 
            predict_train = self.best_model.predict(self.X_train), 
            actual_test = self.y_val, 
            predict_test = self.best_model.predict(self.X_val), 
            path = os.path.join(self.results_path, 'Confusion_Matrices.jpg'))
    
    # формує графіки ROC AUC та Gini
    def saveRocAucGini(self):
        RocAucGini(actual_train = self.y_train, 
            predict_train = self.best_model.predict_proba(self.X_train)[:, 1], 
            actual_test = self.y_val, 
            predict_test = self.best_model.predict_proba(self.X_val)[:, 1], 
            path = os.path.join(self.results_path, 'ROC-AUC_Gini.jpg'))
        
    # формує графіки precision та recall
    def savePrecisionRecall(self):
        PrecisionRecall(actual_train = self.y_train, 
            predict_train = self.best_model.predict_proba(self.X_train)[:, 1], 
            actual_test = self.y_val, 
            predict_test = self.best_model.predict_proba(self.X_val)[:, 1], 
            path = os.path.join(self.results_path, 'Precision_Recall.jpg'))
    
    # формує графік важливості фіч
    def saveFeatureImportance(self):
        explainer = shap.TreeExplainer(self.best_model)
        if self.model == 'lgb' and self.type == 'binary':
            shap_values = explainer.shap_values(self.X_train)[1]
        elif self.model == 'xgb' or self.model == 'lgb' and self.type == 'multi':
            shap_values = explainer.shap_values(self.X_train)
        else:
            pass
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if self.type == 'binary':
            shap.summary_plot(shap_values, self.X_train, plot_type="violin", 
                          max_display=19, show=False, plot_size=(15, 5))
        elif self.type == 'multi':
            shap.summary_plot(shap_values, self.X_train, 
                          max_display=19, show=False, plot_size=(15, 5))
        fig.savefig(os.path.join(self.results_path, 'SHAP_importance.jpg'))
