import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
import pickle
################################################
st.set_page_config(layout='wide')
data=pd.read_csv("./global_income_inequality.csv")
st.subheader("Loading Data")

st.write(data.head())
st.subheader("Data Inforamation")


st.write(data.describe().T)
#######################################
income_share_evolution = data[['Year', 'Top 10% Income Share (%)', 'Bottom 10% Income Share (%)']].dropna()

# Plotting the evolution of income shares over time
st.subheader("Evolution of Top and Bottom 10% Income Shares Over Time")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=income_share_evolution, x='Year', y='Top 10% Income Share (%)', label='Top 10% Income', ax=ax)
sns.lineplot(data=income_share_evolution, x='Year', y='Bottom 10% Income Share (%)', label='Bottom 10% Income', ax=ax)
ax.set_title("Evolution of Top and Bottom 10% Income Shares Over Time")
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)
################################## 
# Get the top and bottom 5 populated countries
most_populated = data.nlargest(5, 'Population')[['Country', 'Year', 'Population']]
least_populated = data.nsmallest(5, 'Population')[['Country', 'Year', 'Population']]

# Display the results
st.subheader("Most Populated Countries")
st.write(most_populated)

st.subheader("Least Populated Countries")
st.write(least_populated)
#########################
st.subheader("Outliers in Gini Index Across Income Groups")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=data, x='Income Group', y='Gini Index', ax=ax)
ax.set_title("Outliers in Gini Index Across Income Groups")

# Display the plot in Streamlit
st.pyplot(fig)
################################
st.subheader("Outliers in Average Income Across Income Groups")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=data, x='Income Group', y='Average Income (USD)', ax=ax)
ax.set_title("Outliers in Average Income Across Income Groups")

# Display the plot in Streamlit
st.pyplot(fig)
######################################
top_5_gini = data.nlargest(5, 'Gini Index')[['Country', 'Year', 'Gini Index']]
bottom_5_gini = data.nsmallest(5, 'Gini Index')[['Country', 'Year', 'Gini Index']]
# 2. Trend of Average Income Over Time (for top 3 countries by population)
top_3_pop_countries = data.groupby('Country')['Population'].max().nlargest(3).index
avg_income_trend = data[data['Country'].isin(top_3_pop_countries)]
st.title("Trend of Average Income Over Time (Top 3 Populous Countries)")

# Create a line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_income_trend, x='Year', y='Average Income (USD)', hue='Country', marker='o')
plt.title("Trend of Average Income Over Time")
plt.xlabel("Year")
plt.ylabel("Average Income (USD)")

# Display the plot in Streamlit
st.pyplot(plt)

st.title("Average Gini Index Trends Across Income Groups")

# Create a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Year', y='Gini Index', hue='Income Group', marker='o')
plt.title("Average Gini Index Trends Across Income Groups")
plt.xlabel("Year")
plt.ylabel("Gini Index")
plt.legend(title="Income Group", bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot in Streamlit
st.pyplot(plt)

st.title("Gini Index Increase Analysis")

# Calculate Gini Index difference
gini_diff = data.groupby('Country')['Gini Index'].agg(lambda x: x.iloc[-1] - x.iloc[0]).reset_index()
top_increase = gini_diff.nlargest(5, 'Gini Index')

# Display the results
st.write("Top 5 Countries with Highest Increase in Gini Index:")
st.dataframe(top_increase)

#########################################
st.title("Correlation Analysis between Gini Index and Average Income")

# Calculate correlation
correlation = data['Gini Index'].corr(data['Average Income (USD)'])
st.write(f"Correlation between Gini Index and Average Income: **{correlation:.2f}**")

# Create a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Gini Index', y='Average Income (USD)', hue='Income Group')
plt.title("Gini Index vs Average Income")
plt.xlabel("Gini Index")
plt.ylabel("Average Income (USD)")

# Display the plot in Streamlit
st.pyplot(plt)

st.title("Countries with High Inequality and Income")

# Filter for high Gini Index and high Average Income
high_inequality_income = data[(data['Gini Index'] > 0.5) & (data['Average Income (USD)'] > 50000)]

# Display the results
st.write("Countries with High Gini Index and High Average Income:")
st.dataframe(high_inequality_income[['Country', 'Year', 'Gini Index', 'Average Income (USD)']])

st.title("Gini Index Time Series for Selected Countries")

# Define the countries of interest
selected_countries = ['United States', 'India', 'Brazil']
subset = data[data['Country'].isin(selected_countries)]

# Create a line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=subset, x='Year', y='Gini Index', hue='Country', marker='o')
plt.title("Gini Index Time Series for Selected Countries")
plt.xlabel("Year")
plt.ylabel("Gini Index")

# Display the plot in Streamlit
st.pyplot(plt)

st.title("Economic and Social Data Analysis")

# Heatmap of Gini Index by Country and Year
st.header("Gini Index Heatmap by Country and Year")
gini_pivot = data.pivot_table(values='Gini Index', index='Year', columns='Country', aggfunc='mean')
plt.figure(figsize=(14, 8))
sns.heatmap(gini_pivot, cmap='YlGnBu', cbar_kws={'label': 'Gini Index'}, linewidths=0.5)
plt.title("Gini Index Heatmap by Country and Year")
st.pyplot(plt)
plt.clf()

# Distribution of Gini Index
st.header("Distribution of Gini Index")
plt.figure(figsize=(8, 5))
sns.histplot(data['Gini Index'], bins=20, kde=True)
plt.title("Distribution of Gini Index")
plt.xlabel("Gini Index")
plt.ylabel("Frequency")
st.pyplot(plt)
plt.clf()

# Population vs. Average Income Scatter Plot
st.header("Population vs. Average Income")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Population', y='Average Income (USD)', hue='Income Group', data=data)
plt.xscale('log')  # Use log scale for better visualization
plt.title("Population vs. Average Income")
plt.xlabel("Population (Log Scale)")
plt.ylabel("Average Income (USD)")
plt.legend(title="Income Group", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(plt)
plt.clf()

# Box Plot of Gini Index by Country
st.header("Income Inequality by Region")
plt.figure(figsize=(12, 8))
sns.boxplot(x='Country', y='Gini Index', data=data)
plt.title('Income Inequality by Region')
plt.xlabel('Region')
plt.ylabel('Gini Index')
plt.xticks(rotation=45)
st.pyplot(plt)
plt.clf()
#####################################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
st.title("Gini Index Prediction Model")

# Prepare the feature set and target variable
X = data.drop(columns=['Gini Index'])
y = data['Gini Index']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.22, random_state=48)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f'Mean Squared Error: {mse:.4f}')
st.write(f'R-squared: {r2:.4f}')

# User input for new hypothetical data
st.header("Enter Hypothetical Data for Prediction")

year = st.number_input("Year", min_value=2025, max_value=2100, value=2025)
population = st.number_input("Population", min_value=1, value=30000000)
average_income = st.number_input("Average Income (USD)", min_value=0, value=80000)
top_10_percent_share = st.number_input("Top 10% Income Share (%)", min_value=0, max_value=100, value=10)
bottom_10_percent_share = st.number_input("Bottom 10% Income Share (%)", min_value=0, max_value=100, value=5)

# Dropdown for selecting country
country = st.selectbox("Country", options=['United States', 'India', 'Brazil', 'Germany'])

# Creating DataFrame with user input
new_data = pd.DataFrame({
    'Year': [year],
    'Population': [population],
    'Average Income (USD)': [average_income],
    'Top 10% Income Share (%)': [top_10_percent_share],
    'Bottom 10% Income Share (%)': [bottom_10_percent_share],
    'Country': [country]  # Ensure this matches one of the training countries
})

# One-hot encode the new data to match training data's feature set
new_data_encoded = pd.get_dummies(new_data, drop_first=True)

# Align the columns of new_data_encoded with the original training data
new_data_encoded = new_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Scale the new data
new_data_scaled = scaler.transform(new_data_encoded)

# Predict the Gini index for the new data
future_prediction = model.predict(new_data_scaled)
st.write(f'Predicted Gini Index for {year}: {future_prediction[0]:.4f}')