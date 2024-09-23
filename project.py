import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('project/hydrogen_projects.csv')

df.columns

df['Date online'] = pd.to_datetime(df['Date online'], errors='coerce', format="%Y")
df['IEA zero-carbon estimated normalized capacity\n[Nm³ H₂/hour]'] = pd.to_numeric(df['IEA zero-carbon estimated normalized capacity\n[Nm³ H₂/hour]'], errors='coerce')

df = df.dropna(subset=['Date online', 'Technology', 'Country', 'IEA zero-carbon estimated normalized capacity\n[Nm³ H₂/hour]'])

total_projects = len(df)

time_range = f"{df['Date online'].min().year} to {df['Date online'].max().year}"

top_countries = df['Country'].value_counts().head(10)

top_technologies = df['Technology'].value_counts()

end_products = df[['EndUse_Refining', 'EndUse_Ammonia', 'EndUse_Methanol', 'EndUse_Iron&Steel', 'EndUse_Other Ind', 'EndUse_Mobility', 'EndUse_Power', 'EndUse_Grid inj.', 'EndUse_CHP', 'EndUse_Domestic heat', 'EndUse_Biofuels', 'EndUse_Synfuels', 'EndUse_CH4 grid inj.', 'EndUse_CH4 mobility']].sum()

print(f"Total projects: {total_projects}")
print(f"Time range: {time_range}")
print("\nTop 5 countries:")
print(top_countries)
print("\nMain technologies:")
print(top_technologies)
print("\nEnd products:")
print(end_products)


features = ['IEA zero-carbon estimated normalized capacity\n[Nm³ H₂/hour]', 'LOWE_CF']

scaler = StandardScaler()
X = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='IEA zero-carbon estimated normalized capacity\n[Nm³ H₂/hour]', y='LOWE_CF', hue='Cluster', palette='deep')
plt.title('Project Clusters')
plt.xlabel('Normalized Capacity')
plt.ylabel('Capacity Factor')
plt.show()

end_uses = ['EndUse_Refining', 'EndUse_Ammonia', 'EndUse_Methanol', 'EndUse_Iron&Steel', 'EndUse_Other Ind', 'EndUse_Mobility', 'EndUse_Power', 'EndUse_Grid inj.', 'EndUse_CHP', 'EndUse_Domestic heat', 'EndUse_Biofuels', 'EndUse_Synfuels', 'EndUse_CH4 grid inj.', 'EndUse_CH4 mobility']

for end_use in end_uses:
    df[end_use] = df[end_use].notna()

binary_df = df[['Technology'] + end_uses]
binary_df = pd.get_dummies(binary_df)

frequent_itemsets = apriori(binary_df, min_support=0.01, use_colnames=True)


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("Top association rules:")
print(rules.sort_values('lift', ascending=False).head(10))

rules_data = rules.sort_values('lift', ascending=False).head(10)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# آماده‌سازی داده برای پیش‌بینی
X = df[['Date online', 'LOWE_CF']]
X['Year'] = X['Date online'].dt.year
X = X.drop('Date online', axis=1)
y = df['IEA zero-carbon estimated normalized capacity\n[Nm³ H₂/hour]']

# تقسیم داده به مجموعه‌های آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ایجاد و آموزش مدل
model = LinearRegression()
model.fit(X_train, y_train)

# پیش‌بینی
y_pred = model.predict(X_test)

# ارزیابی مدل
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# پیش‌بینی ظرفیت تولید تا سال 2030
future_years = pd.DataFrame({ 'LOWE_CF': [0.5]*8, 'Year': range(2023, 2031)})  # فرض می‌کنیم LOWE_CF ثابت است
future_predictions = model.predict(future_years)

plt.figure(figsize=(10, 6))
plt.plot(future_years['Year'], future_predictions, marker='o')
plt.title('Predicted Global Production Capacity')
plt.xlabel('Year')
plt.ylabel('Normalized Capacity [Nm³ H₂/hour]')
plt.show()