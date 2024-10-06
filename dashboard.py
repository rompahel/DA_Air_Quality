import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
@st.cache
def load_data():
    file_path = r'D:\Kuliah\Bangkit\Cleaned_PRSA_Data_20130301-20170228.csv'
    return pd.read_csv(file_path)

# Load data
cleaned_data = load_data()

# Title and Introduction
st.title("Air Quality Analysis: PRSA Dataset")
st.write("""
This dashboard provides an exploratory data analysis (EDA) of the PRSA dataset,
focusing on three key questions:
1. **What is the correlation between wind speed (WSPM) and PM2.5 concentration levels?**
2. **Are there identifiable seasonal patterns in PM10 concentration levels across different months or seasons?**
3. **Clustering Analysis of Pollutants (PM2.5, PM10, SO2, NO2, CO, O3) by Seasons**

The dataset covers air quality data from March 2013 to February 2017.
""")

# Question 1: Correlation between Wind Speed (WSPM) and PM2.5 Levels
st.header("1. Correlation between Wind Speed and PM2.5 Levels")

# Correlation analysis
correlation = cleaned_data[['PM2.5', 'WSPM']].corr()

# Plot
st.subheader("Scatter Plot of Wind Speed vs PM2.5")
fig, ax = plt.subplots()
sns.scatterplot(x='WSPM', y='PM2.5', data=cleaned_data, alpha=0.5, ax=ax)
ax.set_title('Scatter plot of PM2.5 vs Wind Speed (WSPM)')
ax.set_xlabel('Wind Speed (WSPM)')
ax.set_ylabel('PM2.5 Concentration')
st.pyplot(fig)

# Display correlation result
st.write("**Correlation between PM2.5 and Wind Speed (WSPM):**", correlation.iloc[0, 1])

st.write("""
### Conclusion:
There is a weak negative correlation (-0.28) between wind speed and PM2.5 concentration. 
This suggests that higher wind speeds tend to reduce PM2.5 concentrations slightly, 
which means wind disperses particulate matter, improving air quality to some extent.
""")

# Question 2: Seasonal Patterns in PM10 Levels
st.header("2. Seasonal Patterns in PM10 Concentration Levels")

# Seasonal Analysis
seasonal_pm10 = cleaned_data.groupby('season')['PM10'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])

# Bar plot
st.subheader("Average PM10 Levels by Season")
fig, ax = plt.subplots()
seasonal_pm10.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Average PM10 Concentration Levels by Season')
ax.set_xlabel('Season')
ax.set_ylabel('Average PM10 Concentration')
st.pyplot(fig)

# Display seasonal averages
st.write("**Seasonal PM10 Averages:**")
st.write(seasonal_pm10)

st.write("""
### Conclusion:
PM10 levels are highest in Spring (132.04) and lowest in Summer (81.51). 
This suggests that air quality worsens in spring and improves in summer, 
likely due to a combination of environmental and human activities, such as industrial activities, weather patterns, or natural events like dust storms.
""")

# Question 3: Clustering Analysis
st.header("3. Clustering Analysis of Pollutants by Seasons")

# Clean data for clustering analysis (only using relevant columns)
cluster_data = cleaned_data[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'season']].dropna()

# Scale the features for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_data[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']])

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=0)
cluster_data['Cluster'] = kmeans.fit_predict(scaled_features)

# Plot the clustering result by season
st.subheader("Clustered Pollutants by Seasons")
fig, ax = plt.subplots()
sns.scatterplot(x='PM2.5', y='PM10', hue='Cluster', palette='tab10', data=cluster_data, ax=ax)
ax.set_title('Clustering of Pollutants based on PM2.5 and PM10 levels')
st.pyplot(fig)

# Display cluster centers (average pollutant levels in each cluster)
st.write("**Cluster Centers (Mean pollutant levels for each cluster):**")
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                               columns=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'])
st.write(cluster_centers)

st.write("""
### Conclusion:
The clustering analysis reveals distinct patterns of pollutant levels across different seasons. 
Winter and Fall tend to have higher PM2.5 and CO levels, while Summer sees lower pollutant concentrations overall. 
Spring displays moderate pollution levels, with PM10 being notably higher.
""")

st.header("Conclusion")

st.write("""
**Key Conclusions:**
1. There is a weak negative correlation between wind speed (WSPM) and PM2.5 concentration levels, suggesting that higher wind speeds tend to slightly lower PM2.5 concentration in the air.
2. There are clear seasonal patterns in PM10 concentration levels, with air quality being worse in spring and better in summer.
3. Clustering analysis further supports the idea that pollution levels vary significantly by season, with Winter having the highest concentrations of particulate matter and carbon monoxide.
""")
