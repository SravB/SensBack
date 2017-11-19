#back

#imports
from time import *
from sklearn import tree
import pandas as pd
import numpy as np
import time
start_time = time.time()
    
cols = []

for i in range(12):
    cols.append([])
    
df = pd.read_csv('Dataset_spine.csv', parse_dates = True)
    
for k in range(1,13):
    for i in df[['Col' + str(k)]]:
        for j in df[i]:
            cols[k-1].append(j)

symp = []

for i in df[['Class_att']]:
    for j in df[i]:
        if j == "Abnormal":
            symp.append(0)
        else:
            symp.append(1)

x = []

for i in range(len(cols[0])):
    x.append([])
    for j in range(12):
        x[i].append(cols[j][i])

r = 6

x_train = x[:200+r] + x[211:300+r]

y_train = symp[:200+r] + symp[211:300+r]

x_test = x[200+r:211] + x[300+r:]
    
y_test = symp[200+r:211] + symp[300+r:]

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

print(str(round(clf.score(x_test,y_test)*100,2)) + "% accuracy")





