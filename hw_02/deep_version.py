#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:07:57 2018

@author: petrkozyrev
"""

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

%matplotlib inline

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12,5)

df_train = pd.read_csv('data/otp_train.csv', sep='\t', encoding='utf8')

df_train.shape

df_test = pd.read_csv('data/otp_test.csv', sep='\t', encoding='utf8')

df_test.shape

df_train.loc[:, 'sample'] = 'train'
df_test.loc[:, 'sample'] = 'test'

df = df_test.append(df_train).reset_index(drop=True)

def preproc_data(df_input):
    df_output = df_input.copy()
    
    # Перечень колонок, которые нужно удалить.
    # Колонки с адресами вероятнее всего дублируются, 
    # поэтому оставим только FACT_ADDRESS_PROVINCE, поскольку фактический адрес
    # более важен, чем адрес регистрации
    # Удаляем столбец с регионами, так как субъекты входят в регионы
    remove_cols = ['AGREEMENT_RK']
    df_output = df_output.drop(remove_cols, axis=1)
    
    
    # Перечень столбцов, значения которых нужно конвертировать из str во float
    numbers_cols = ['PERSONAL_INCOME',
                    'CREDIT',
                    'FST_PAYMENT',
                    'LOAN_AVG_DLQ_AMT',
                    'LOAN_MAX_DLQ_AMT']
    for nc in numbers_cols:
        df_output[nc] = df_output[nc].map(lambda x: x.replace(',', '.')).astype('float')
        
    # Дополнительный one-hot-vector для учета людей, не имеющих работы
    df_output['НЕ РАБОТАЕТ'] = df_output['GEN_TITLE'].isna().replace(False, 0).replace(True, 1)
    
    # Переводим категории FAMILY_INCOME в числа от 0 до 4 с помощью метода replace
#    di = {'до 5000 руб.': 0, 
#          'от 5000 до 10000 руб.': 1,
#          'от 10000 до 20000 руб.': 2,
#          'от 20000 до 50000 руб.': 3,
#          'свыше 50000 руб.': 4
#           }
#    df_output['FAMILY_INCOME'] = df_output['FAMILY_INCOME'].replace(di)
    
    # Заменяем пропуски на нули в колонке PREVIOUS_CARD_NUM_UTILIZED
    df_output['PREVIOUS_CARD_NUM_UTILIZED'] = df_output['PREVIOUS_CARD_NUM_UTILIZED'].fillna(0)
    
    # Заменяем пропуски в поле WORK_TIME на Медиану
    MEDIAN = df_output['WORK_TIME'].dropna().median()
    df_output['WORK_TIME'] = df_output['WORK_TIME'].map(lambda x: MEDIAN if np.isnan(x) else x)
    
    # Перечень категориальных признаков, 
    # и разделяем one-hot-vectors с помощью метода get_dummies. По умолчанию get_dummies
    # проставит значения 0 по тем строкам, в которых пропущены значения
    categorize_cols = ['EDUCATION',
                       'MARITAL_STATUS',
                       'GEN_INDUSTRY',
                       'GEN_TITLE',
                       'ORG_TP_STATE',
                       'ORG_TP_FCAPITAL',
                       'JOB_DIR',
                       'FACT_ADDRESS_PROVINCE','FAMILY_INCOME',
                   'REG_ADDRESS_PROVINCE',
                   'POSTAL_ADDRESS_PROVINCE',
                   'TP_PROVINCE',
                   'REGION_NM']
    # конвертируем временно 'sample' в числа, иначе метод выдает ошибку
    df_output['sample'] = df_output['sample'].map(lambda x: 1 if x == 'train' else 0)
    df_output = pd.get_dummies(df_output)
    df_output['sample'] = df_output['sample'].map(lambda x: 'train' if x == 1 else 'test')
    
    return df_output

df_preproc = df.pipe(preproc_data)

df_train_preproc = df_preproc.query('sample == "train"').drop(['sample'], axis=1)
df_test_preproc = df_preproc.query('sample == "test"').drop(['sample'], axis=1)


X      = df_train_preproc.drop(['TARGET'], axis=1)
X_test = df_test_preproc.drop(['TARGET'], axis=1)

y      = df_train_preproc['TARGET']
y_test = df_test_preproc['TARGET']

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)

#%%

#hyperparameters
hidden_size = 50
learning_rate = 0.001
num_epoch = 500
weight_decay = 1e-05


class BankNN(nn.Module):
    def __init__(self):
        super(BankNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = BankNN()

#choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.98)
#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.999, 0.9999), eps=1e-9, weight_decay=1e-7, amsgrad=False)

loss_value = 0

accuracy = 0

X = np.array(X)
y = np.array(y)

#X_train = np.array(X_train)
#y_train = np.array(y_train)
X_train = np.array(X)
y_train = np.array(y)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)
X_test = np.array(X_test)
y_test = np.array(y_test)

#%%

X_torch_train = torch.FloatTensor(X_train)
    
y_torch_train = torch.LongTensor(y_train)

X_torch = torch.FloatTensor(X)
    
y_torch = torch.LongTensor(y)

X_torch_valid = torch.FloatTensor(X_valid)

y_torch_valid = torch.LongTensor(y_valid)

X_torch_test = torch.FloatTensor(X_test)

y_torch_test = torch.LongTensor(y_test)

#%%
for epoch in range(num_epoch):

    optimizer.zero_grad()

    out = net(X_torch)

    loss = criterion(out, y_torch)

    loss.backward()

    optimizer.step()

    xtest = torch.FloatTensor(X_torch_test)

    ytest = torch.LongTensor(y_torch_test)

    test_out = net(xtest)
    _, predicted = torch.max(test_out.data, 1)

    correct = (predicted == ytest).sum()

    accuracy = 100 * float(correct) / len(ytest)

    loss_value = loss.item()
    
    if epoch % 20 == 0:
        
        params = ('Epochs: {}, LR: {}, Hidden_size: {}, weight_decay: {} Loss: {}, test {}'.format(epoch, learning_rate, hidden_size, weight_decay, loss_value, accuracy))
        print(params)


#%%
        
y_test_n = (torch.LongTensor(y_torch_test)).numpy()
x_test_n = torch.FloatTensor(X_torch_test)
predict_n = net(xtest).detach().numpy()

#%%

%matplotlib inline
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

dtc_fpr, dtc_tpr, dtc_thresholds = roc_curve(y_test_n, predict_n[:,1])
plt.figure(figsize=(10, 10))

plt.plot(dtc_fpr, dtc_tpr, label='dtc')
#plt.legend('dtc')


plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0, 1], [0, 1])
plt.ylabel('tpr')
plt.xlabel('fpr')
plt.grid(True)
plt.title('ROC curve')
plt.xlim((-0.01, 1.01))
plt.ylim((-0.01, 1.01))

dtc_roc_auc_score = roc_auc_score(y_test_n, predict_n[:,1])
print('DecisionTreeClassifier roc_auc: {}'.format(dtc_roc_auc_score))


