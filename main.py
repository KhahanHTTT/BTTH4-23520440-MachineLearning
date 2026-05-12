import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier

# LOAD DATA

red   = pd.read_csv("data/winequality-red.csv",   sep=';')
white = pd.read_csv("data/winequality-white.csv",  sep=';')

red['type']   = 0
white['type'] = 1

df = pd.concat([red, white], axis=0)

# PREPROCESS

df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

X = df.drop('quality', axis=1).values
y = df['quality'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
std[std == 0] = 1

X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

print(f"Tập huấn luyện : {len(X_train)} mẫu")
print(f"Tập kiểm tra   : {len(X_test)}  mẫu")
print(f"Phân phối nhãn : {np.bincount(y)}")


# ASSIGNMENT 1: DECISION TREE USING NUMPY

print("\n=== ASSIGNMENT 1: DECISION TREE (NumPy) ===")
dt = DecisionTreeClassifier(max_depth=6, min_samples_split=5)
dt.fit(X_train, y_train)
print("F1 Train:", f1_score(y_train, dt.predict(X_train)))
print("F1 Test :", f1_score(y_test,  dt.predict(X_test)))


# ASSIGNMENT 2: RANDOM FOREST USING NUMPY
print("\n=== ASSIGNMENT 2: RANDOM FOREST (NumPy) ===")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
rf.fit(X_train, y_train)
print("F1 Train:", f1_score(y_train, rf.predict(X_train)))
print("F1 Test :", f1_score(y_test,  rf.predict(X_test)))



# ASSIGNMENT 3: SKLEARN

print("\n=== ASSIGNMENT 3: DECISION TREE (sklearn) ===")
dt_sk = SklearnDT(max_depth=6, random_state=42)
dt_sk.fit(X_train, y_train)
print("F1 Train:", f1_score(y_train, dt_sk.predict(X_train)))
print("F1 Test :", f1_score(y_test,  dt_sk.predict(X_test)))


print("\n=== ASSIGNMENT 3: RANDOM FOREST (sklearn) ===")
rf_sk = SklearnRF(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf_sk.fit(X_train, y_train)
print("F1 Train:", f1_score(y_train, rf_sk.predict(X_train)))
print("F1 Test :", f1_score(y_test,  rf_sk.predict(X_test)))
