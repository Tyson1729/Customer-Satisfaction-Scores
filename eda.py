import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('customer_feedback_satisfaction.csv')

print("Dataset Shape:", data.shape)
print(data.info())
print(data.describe())
print("Missing Values:\n", data.isnull().sum())

# Target distribution
sns.histplot(data['SatisfactionScore'], bins=20, kde=True)
plt.title("Distribution of SatisfactionScore")
plt.show()

# Numerical vs Target
num_features = ['Age', 'Income', 'ProductQuality', 'ServiceQuality', 'PurchaseFrequency']
for col in num_features:
    sns.scatterplot(x=data[col], y=data['SatisfactionScore'])
    plt.title(f"{col} vs SatisfactionScore")
    plt.show()

# Categorical vs Target
cat_features = ['Gender', 'Country', 'FeedbackScore', 'LoyaltyLevel']
for col in cat_features:
    sns.boxplot(x=data[col], y=data['SatisfactionScore'])
    plt.title(f"SatisfactionScore by {col}")
    plt.show()

# Correlation Heatmap
corr = data[num_features + ['SatisfactionScore']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
