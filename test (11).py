import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('hydrogen_projects.csv')

# Convert date columns to datetime
data['Date online'] = pd.to_datetime(data['Date online'], format='%Y', errors='coerce')

# Fill missing values if necessary
data['Date online'].dropna(inplace=True)

# Feature engineering: Create year and other useful features
data['year'] = data['Date online'].dt.year

# Summarize end-use counts by year to target for prediction
end_use_cols = ['EndUse_Refining', 'EndUse_Ammonia', 'EndUse_Methanol', 'EndUse_Iron&Steel', 'EndUse_Other Ind', 'EndUse_Mobility', 'EndUse_Power', 'EndUse_Grid inj.', 'EndUse_CHP', 'EndUse_Domestic heat', 'EndUse_Biofuels', 'EndUse_Synfuels', 'EndUse_CH4 grid inj.', 'EndUse_CH4 mobility']
end_use_df = data.groupby('year')[end_use_cols].sum().reset_index()

colors = sns.color_palette("tab10", len(end_use_cols))
plt.figure(figsize=(14, 8))
for (col, color) in zip(end_use_cols, colors):
    sns.lineplot(x='year', y=col, data=end_use_df, label=col, color=color)
plt.title('Trends in Hydrogen End-use Sectors Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.show()


# Creating lag features for each end-use column
for col in end_use_cols:
    end_use_df[f'{col}_lag1'] = end_use_df[col].shift(1)
    end_use_df[f'{col}_lag2'] = end_use_df[col].shift(2)
    end_use_df[f'{col}_lag3'] = end_use_df[col].shift(3)

# We will have to drop the initial rows with NaN values due to lag features
end_use_df.dropna(inplace=True)

# Split the data
X = end_use_df.drop(columns=['year'] + end_use_cols)
y = end_use_df[end_use_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a RandomForestRegressor for each end-use sector
end_use_models = {}
for col in end_use_cols:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[col])
    end_use_models[col] = model


metrics = []
for col in end_use_cols:
    y_pred = end_use_models[col].predict(X_test)
    mse = mean_squared_error(y_test[col], y_pred)
    r2 = r2_score(y_test[col], y_pred)
    metrics.append((col, mse, r2))

# Print out the metrics
metrics_df = pd.DataFrame(metrics, columns=['End-Use Sector', 'MSE', 'R2'])
print(metrics_df)

# Visualize the performance
plt.figure(figsize=(14, 8))
metrics_df.sort_values(by='R2', ascending=False, inplace=True)
sns.barplot(x='R2', y='End-Use Sector', data=metrics_df)
plt.title('Model Performance (R2 Score) by End-Use Sector')
plt.show()
