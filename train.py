import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

data = pd.read_csv(r'E:\Codes\DOS_PR\New folder\Credit_Card_Fraud\Credit.csv')
print(data.head())
print(data.tail())
print(data.describe())
print(data.info())

Education = {'Unknown': 3, 'High School': 2, 'University': 1, 'Graduate school': 2}
Marietal = {'Married': 1, 'Single': 2, 'Other': 3 , '0':0}
Sex = {'F': 1, 'M': 0}
Fraud = {'Y': 1, 'N': 0}

data['FRAUD'] = data['FRAUD'].str[0].map(Fraud)
data['EDUCATION'] = data['EDUCATION'].map(Education)
data['MARRIAGE'] = data['MARRIAGE'].map(Marietal)
data['SEX'] = data['SEX'].map(Sex)

# Data Cleaning
# data.to_csv('Credit1.csv', index=False)

data = pd.read_csv(r'E:\Codes\DOS_PR\New folder\Credit_Card_Fraud\Credit1.csv')
print(data.head())
data.dropna()
data.round()
data.isnull().sum().sum()
data = data.fillna(value=0)
data.isnull().sum()
print(data.describe())
print(data.info())

# Train Test Split
y = data['FRAUD']
X = data.drop('FRAUD', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Feature Engineering

#Feature Selection using Anova
sel = SelectKBest(f_classif, k=10).fit(X_train, y_train)
print(X_train.columns[sel.get_support()])
columns = X_train.columns[sel.get_support()]
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Feature Selection using PCA
pca = PCA(n_components=5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Model Building and Evaluation
# Upsampling using Smote
sm = SMOTE(random_state = 2)
X_train, y_train = sm.fit_resample(X_train, y_train.ravel())

# Model = Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
lr_predict = model.predict(X_train)
train_accuracy = accuracy_score(lr_predict,y_train)
print(train_accuracy)

# Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating a DNN model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1:])),
    tf.keras.layers.Dense(8, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=5, batch_size=16, validation_split=0.1)
test_loss = model.evaluate(X_test_scaled, y_test)
predictions = model.predict(X_test_scaled)