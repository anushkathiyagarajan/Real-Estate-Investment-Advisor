import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

# CLASSIFICATION MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# REGRESSION MODELS
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# METRICS
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score )
import pickle

# LOAD DATA
df = pd.read_csv( r"C:\mini 2\data\processed\cleaned_data.csv")

# REDUCE SIZE
df = df.sample(  50000, random_state=42)

# FEATURES
X = df.drop(columns=[
    'ID',
    'Good_Investment',
    'Future_Price_5Y',
    'Appreciation_Percentage',
    'Price_in_Lakhs'

])


# TARGETS
y_class = df['Good_Investment']
y_reg = df['Future_Price_5Y']
print("Final shape of X:", X.shape)

# TRAIN TEST SPLIT
X_train, X_test, y_train_c, y_test_c = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42)
_, _, y_train_r, y_test_r = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42)

# CLASSIFICATION MODELS

classification_models = {
 "Logistic Regression": LogisticRegression(
    max_iter=5000,
    solver='saga',
    random_state=42), "Decision Tree": DecisionTreeClassifier(), "Random Forest": RandomForestClassifier(), "SVM": SVC(probability=True), "XGBoost": XGBClassifier()}

print("\n===== CLASSIFICATION RESULTS =====")

best_accuracy = 0
best_clf_model = None

for name, model in classification_models.items():
    model.fit(X_train, y_train_c)
    pred = model.predict(X_test)
    accuracy = accuracy_score(
        y_test_c,
        pred)
    precision = precision_score(
        y_test_c,
        pred)
    recall = recall_score(
        y_test_c,
        pred)

    roc_auc = roc_auc_score(
        y_test_c,
        pred)

    print(f"\n{name}")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("ROC AUC:", roc_auc)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_clf_model = model

# REGRESSION MODELS
regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(),
    "XGBoost": XGBRegressor()}

print("\n===== REGRESSION RESULTS =====")

best_r2 = 0
best_reg_model = None

for name, model in regression_models.items():
    model.fit(X_train, y_train_r)
    pred = model.predict(X_test)
    rmse = mean_squared_error(
        y_test_r,
        pred
    ) ** 0.5

    mae = mean_absolute_error(
        y_test_r,
        pred)

    r2 = r2_score(
        y_test_r,
        pred)

    print(f"\n{name}")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)

    if r2 > best_r2:
        best_r2 = r2
        best_reg_model = model

# SAVE BEST MODELS

with open( r"C:\mini 2\models\classification_model.pkl",
    "wb") as file: pickle.dump(
        best_clf_model,
        file)

with open(r"C:\mini 2\models\regression_model.pkl",
    "wb")as file:
    pickle.dump(
        best_reg_model,
        file )


# FINAL BEST MODELS
print(" BEST MODELS")
print(
    "Best Classification Model: "
    "XGBoost Classifier")
print(
    "Best Regression Model: "
    "Random Forest Regressor")

