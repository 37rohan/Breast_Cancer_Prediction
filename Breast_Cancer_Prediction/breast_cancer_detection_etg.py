import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
data

data.keys()

print(data['DESCR'])

print(data['data'])

data['data'].shape

data['feature_names']

j = 0
for i in data['feature_names']:
  print(f"{i} : {data['data'][0][j]}")
  j+=1

feature = data['data']
feature.shape

label = data['target']
label.shape

scaler = StandardScaler()
feature = scaler.fit_transform(feature)

j = 0
for i in data['feature_names']:
  print(f"{i} : {feature[0][j]}")
  j+=1

print(feature[0])
print(data['target_names'][label[20]])

data['target_names'][0]

df_ftr = pd.DataFrame(feature,columns=data['feature_names'])
df_lbl = pd.DataFrame(label,columns=['label'])
df = pd.concat([df_ftr,df_lbl],axis=1)
df = df.sample(frac=1)

df

feature = df.values[:,:30]
label = df.values[:, 30:]

#500 Training
X_train = feature[:500]
y_train = label[:500]
#35 Training
X_val = feature[500:535]
y_val = label[500:535]
#34 Training
X_test = feature[535:]
y_test = label[535:]

model = Sequential()

model.add(Dense(64,activation = 'relu', input_dim = 30)) #add layer
model.add(Dense(32,activation = 'relu'))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size=1, epochs=5,validation_data=(X_val,y_val))

model.evaluate(X_test,y_test)

model.evaluate(X_test,y_test)

for i in range(10):
  sample = X_test[i]
  sample = np.reshape(sample,(1,30))

  print("Predicted")
  if model.predict(sample)[0][0] > 0.5:
    print("Benign")
  else:
    print("Malignant")


  print("Actual")

  if y_test[i] == 1:
    print("Benign\n")
  else:
    print("Malignant\n")
