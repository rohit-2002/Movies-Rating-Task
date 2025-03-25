import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')

# Clean & preprocess
df = df[df['Rating'].notna()]
df['Year'] = pd.to_numeric(df['Year'].str.extract(r'(\d{4})')[0], errors='coerce')
df['Duration'] = pd.to_numeric(df['Duration'].str.extract(r'(\d+)')[0], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'].astype(str).str.replace(',', ''), errors='coerce')
df = df.dropna(subset=['Genre', 'Director', 'Year', 'Duration', 'Votes'])

# Feature engineering
df['Director_Avg_Rating'] = df.groupby('Director')['Rating'].transform('mean')
df['Genre_Avg_Rating'] = df.groupby('Genre')['Rating'].transform('mean')

for col in ['Director', 'Genre', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col + '_Freq'] = df[col].map(df[col].value_counts())

features = [
    'Year', 'Duration', 'Votes',
    'Director_Avg_Rating', 'Genre_Avg_Rating',
    'Director_Freq', 'Genre_Freq',
    'Actor 1_Freq', 'Actor 2_Freq', 'Actor 3_Freq'
]

X = df[features].fillna(0)
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using sklearn only
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and features
joblib.dump(model, "model.pkl")
joblib.dump(features, "features.pkl")
