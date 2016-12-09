# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 12:53:51 2016

@author: Shruti
"""
import numpy as np
import matplotlib.pyplot as plt

e = plt.figure(5)
plt.plot([0.821207, 0.822106, 0.822756, 0.823910])
plt.plot([0.836591, 0.836913, 0.835968, 0.836747])
plt.xticks([0,1,2,3])
plt.ylabel('Kaggle AUC score')
plt.xlabel('Iteration')
plt.title('XGBoost Results')
plt.legend(['private score', 'public score'], loc=0)
e.show()

f = plt.figure(1)
plt.plot([0.686672,0.754843,0.759599,0.791145, 0.793066, 0.798727, 0.803132, 0.804647, 0.832491, 0.832227])
plt.plot([0.671909,0.751811,0.756382,0.783696, 0.786355, 0.778860, 0.783038,0.791922,0.818314, 0.818976])
plt.ylabel('Kaggle AUC score')
plt.xlabel('Iteration')
plt.title('Random Forest Results')
plt.legend(['private score', 'public score'], loc=0)
f.show()

g = plt.figure(2)
plt.plot([0.611602,0.611613,0.611054, 0.611039])
plt.plot([0.615585,0.615595,0.615240, 0.615236])
plt.xticks([0,1,2,3])
plt.ylabel('Kaggle AUC score')
plt.xlabel('Iteration')
plt.title('Logistic Regression Results')
plt.legend(['private score', 'public score'], loc=0)
g.show()

h = plt.figure(3)
ind = np.arange(15)
y = [0.217691,0.131334,0.047315,0.040387,0.038204,
     0.029987,0.029860,0.029679,0.028397,0.025948,
     0.022221,0.021480,0.018447,0.017677,0.015900]
x_ticks = ['var15','saldo_var30','var38ismode','PCA_0','ID','PCA_3','num_var30',
'logvar38','PCA_1','PCA_2','saldo_medio_var5_hace2','saldo_var42','num_var35',
'saldo_medio_var5_ult3','saldo_medio_var5_ult1']
width = 1/1.5
plt.yticks(ind, x_ticks)
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.title('Feature Importance from Random Forests with Feature Engineering')
plt.barh(ind, y, width, color="blue")
h.show()
'''
feature 2 is var15 (0.217691)
feature 55 is saldo_var30 (0.131334)
feature 175 is var38ismode (0.047315)
feature 177 is PCA_0 (0.040387)
feature 0 is ID (0.038204)
feature 180 is PCA_3 (0.029987)
feature 32 is num_var30 (0.029860)
feature 176 is logvar38 (0.029679)
feature 178 is PCA_1 (0.028397)
feature 179 is PCA_2 (0.025948)
feature 135 is saldo_medio_var5_hace2 (0.022221)
feature 62 is saldo_var42 (0.021480)
feature 33 is num_var35 (0.018447)
feature 138 is saldo_medio_var5_ult3 (0.017677)
feature 137 is saldo_medio_var5_ult1 (0.015900)
'''

i = plt.figure(4)
ind = np.arange(15)
y = [0.264336,0.144828,0.079947,0.040971,0.024072,
     0.023815,0.023295,0.022278,0.019954,0.018729,
     0.017985,0.016738,0.015061,0.014350,0.014105]
x_ticks = ['var15','saldo_var30','var38','ID','ind_var30','saldo_medio_var5_hace2','num_var30','saldo_var42','saldo_medio_var5_ult3',
'saldo_medio_var5_hace3','num_var35','saldo_medio_var5_ult1','num_var22_ult3','saldo_var5',
'num_var4']
width = 1/1.5
plt.yticks(ind, x_ticks)
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.title('Feature Importance from Random Forests without Feature Engineering')
plt.barh(ind, y, width, color="blue")
i.show()
'''
1. feature 2 is var15 (0.264336)
2. feature 183 is saldo_var30 (0.144828)
3. feature 369 is var38 (0.079947)
4. feature 0 is ID (0.040971)
5. feature 64 is ind_var30 (0.024072)

6. feature 329 is saldo_medio_var5_hace2 (0.023815)
7. feature 139 is num_var30 (0.023295)
8. feature 191 is saldo_var42 (0.022278)
9. feature 332 is saldo_medio_var5_ult3 (0.019954)
10. feature 330 is saldo_medio_var5_hace3 (0.018729)

11. feature 148 is num_var35 (0.017985)
12. feature 331 is saldo_medio_var5_ult1 (0.016738)
13. feature 278 is num_var22_ult3 (0.015061)
14. feature 165 is saldo_var5 (0.014350)
15. feature 89 is num_var4 (0.014105)
'''

raw_input()