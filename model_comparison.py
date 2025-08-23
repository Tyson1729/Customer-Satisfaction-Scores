import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

data = pd.read_csv('customer_feedback_satisfaction.csv')

# Encoding categorical features
feedback_map = {'Low': 0, 'Medium': 1, 'High': 2}
loyalty_map = {'Bronze': 0, 'Silver': 1, 'Gold': 2}

data['FeedbackScore'] = data['FeedbackScore'].map(feedback_map)
data['LoyaltyLevel'] = data['LoyaltyLevel'].map(loyalty_map)
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

# One-hot encode Country
data = pd.get_dummies(data, columns=['Country'], drop_first=True)

# Handle skewness
num_features = ['Age', 'Income', 'ProductQuality', 'ServiceQuality', 'PurchaseFrequency']
data['Income'] = np.log1p(data['Income'])

# Feature Scaling
scaler = StandardScaler()
data[num_features] = scaler.fit_transform(data[num_features])

# Feature Importance (Random Forest)
X = data.drop(columns=['CustomerID', 'SatisfactionScore'])
y = data['SatisfactionScore']

rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(10,6))
plt.title("Feature Importance")
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Model 1 - Linear Regression": LinearRegression(),
    "Model 2 - Ridge Regression": Ridge(alpha=1.0),
    "Model 3 - Lasso Regression": Lasso(alpha=0.01),
    "Model 4 - ElasticNet Regression": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "Model 5 - Decision Tree": DecisionTreeRegressor(random_state=42),
    "Model 6 - Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Model 7 - Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "Model 8 - K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "Model 9 - Support Vector Regressor": SVR(kernel='rbf'),
    "Model 10 - Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: Accuracy (R² Score) = {r2:.2f}\n")

# Stacking Regressors
# Option 1: GB + SVR + MLP
stack1_base = [
    ("gb", models["Model 7 - Gradient Boosting"]),
    ("svr", models["Model 9 - Support Vector Regressor"]),
    ("mlp", models["Model 10 - Neural Network (MLP)"])
]
stack1_meta = LinearRegression()
stacked_model1 = StackingRegressor(estimators=stack1_base, final_estimator=stack1_meta, passthrough=False)
stacked_model1.fit(X_train, y_train)
print(f"Stacking Option 1 (GB + SVR + MLP): R² = {r2_score(y_test, stacked_model1.predict(X_test)):.2f}\n")

# Option 2: RF + GB + SVR
stack2_base = [
    ("rf", models["Model 6 - Random Forest"]),
    ("gb", models["Model 7 - Gradient Boosting"]),
    ("svr", models["Model 9 - Support Vector Regressor"])
]
stack2_meta = Ridge(alpha=1.0)
stacked_model2 = StackingRegressor(estimators=stack2_base, final_estimator=stack2_meta, passthrough=False)
stacked_model2.fit(X_train, y_train)
print(f"Stacking Option 2 (RF + GB + SVR): R² = {r2_score(y_test, stacked_model2.predict(X_test)):.2f}\n")
