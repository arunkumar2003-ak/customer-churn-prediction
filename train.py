import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# load data
df = pd.read_csv("Telco-Customer-Churn.csv ")

df = df.dropna()

# target convert
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# encode text columns
for col in df.select_dtypes(include='object').columns:
    if col != 'customerID':
        df[col] = LabelEncoder().fit_transform(df[col])

# features
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained successfully 🚀")