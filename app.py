import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

print("Customer Churn Prediction App 🚀")

# load data again (no save model)
df = pd.read_csv("Telco-Customer-Churn.csv ")
df = df.dropna()

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

for col in df.select_dtypes(include='object').columns:
    if col != 'customerID':
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

model = RandomForestClassifier()
model.fit(X, y)

# take ONE sample row for prediction
sample = X.iloc[0].values.reshape(1, -1)

result = model.predict(sample)

if result[0] == 1:
    print("Customer WILL leave 😢")
else:
    print("Customer will STAY 😊")