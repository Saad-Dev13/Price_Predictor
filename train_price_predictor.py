import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import joblib

# 1. Load your data
df = pd.read_csv('projects.csv')

# 2. Preprocess modules column (split comma-separated string into list)
df['modules'] = df['modules'].apply(lambda x: [m.strip() for m in x.split(',')])

# 3. One-hot encode module types
mlb = MultiLabelBinarizer()
modules_encoded = mlb.fit_transform(df['modules'])
modules_df = pd.DataFrame(modules_encoded, columns=mlb.classes_)

# 4. Combine features (modules + days)
X = pd.concat([modules_df, df[['days']]], axis=1)
y = df['price']

# 5. Split into train/test (optional, for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Save the model and the encoder
joblib.dump(model, 'price_predictor.pkl')
joblib.dump(mlb, 'mlb.pkl')

# 8. (Optional) Print test score
score = model.score(X_test, y_test)
print(f"Model R^2 score on test set: {score:.2f}")