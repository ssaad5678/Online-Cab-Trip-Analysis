import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from PIL import Image


image=Image.open('Poster For Mini Project.jpg')
st.image(image,use_column_width=True)
# Define some CSS styles
css = """
h1 {
    color: #FFD700;
    text-align: center;
}
p {
    font-size: 20px;
}
"""

# Display the styles
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
# Display some content
st.write('# Online Cab Trip Analysis')

# Reading data
df = pd.read_csv("Uber Trip.csv", encoding='latin1')
df.columns = df.columns.str.replace("*","")
st.write('Simple DataSet')
st.write(df.head())
st.write('The Complete Data Set')
st.write(df)


# Missing Data
st.write('# Missing Data')
st.write('Heatmap of missing data')
fig=plt.figure(figsize=(15,10))
sns.heatmap(df.isnull(), cmap='magma', yticklabels=False)
st.pyplot(fig)

st.write('# Bar chart of missing data')
fig,ax=plt.subplots()
msno.bar(df,ax=ax)
st.pyplot(fig)

# Handling missing data
df.drop(index=1155, axis=0, inplace=True)
df['PURPOSE'].fillna(method='ffill', inplace=True)

# Data cleaning
df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')
df['END_DATE'] = pd.to_datetime(df['END_DATE'], errors='coerce')

# Categorical data
st.write('# Categorical Data')
category = pd.crosstab(index=df['CATEGORY'], columns='count of travel as per category')
st.write(category)
fig, ax=plt.subplots()
category.plot(kind='bar', color='r',ax=ax)
plt.legend()
st.pyplot(fig)

st.write('# Start points with more than 10 trips')
start_point = df.START.value_counts()
st.write(start_point[start_point>10])
fig, ax=plt.subplots()
start_point[start_point>10].plot(kind='pie', shadow=True,ax=ax)
st.pyplot(fig)

st.write('# Start points with 10 or fewer trips')
st.write(start_point[start_point<=10])

st.write('# Stop points with more than 10 trips')
stop_point = df.STOP.value_counts()
st.write(stop_point[stop_point>10])

st.write('# Stop points with 10 or fewer trips')
st.write(stop_point[stop_point<=10])

st.write('# Miles with more than 10 trips')
miles = df.MILES.value_counts()
st.write(miles[miles>10])
fig,ax=plt.subplots()
miles[miles>10].plot(kind='bar',ax=ax)
st.pyplot(fig)

st.write('# Miles with 10 or fewer trips')
st.write(miles[miles<=10])
data = {'MILES': [840, 31],
        'Round_TRIP': [False, True]}

dfe = pd.DataFrame(data)

st.dataframe(dfe)

miles = pd.crosstab(index=df['MILES']>10, columns='count of miles')
fig, ax=plt.subplots()
miles.plot(kind='bar', color='r',ax=ax)
st.pyplot(fig)

st.write('# Trips per Purpose')
st.write(df.PURPOSE.value_counts())

# Number of Trips per Purpose
purpose_counts = df.groupby('PURPOSE')['PURPOSE'].count().sort_values(ascending=False)
fig, ax=plt.subplots(figsize=(15,6))
purpose_counts.plot(kind='bar', color='blue',ax=ax)
plt.xlabel('Purpose')
plt.ylabel('Count')
plt.title('Number of Trips per Purpose')
st.pyplot(fig)

# Trip duration
df['minutes'] = df.END_DATE - df.START_DATE
df['minutes'] = df['minutes'].dt.total_seconds()/60

st.write('# Trip Duration')
st.write(pd.DataFrame({'Mean': df.groupby(['PURPOSE'])['MILES'].mean().round(1),
                       'Min': df.groupby(['PURPOSE'])['MILES'].min(),
                       'Max': df.groupby(['PURPOSE'])['MILES'].max()}
                      ).reset_index())

ax=plt.figure(figsize=(16,7))
plt.subplot(1,2,1)
sns.boxplot(data=df,x=df.PURPOSE, y=df.MILES)
plt.xticks(rotation=45)
plt.subplot(1,2,2)
sns.boxplot(data=df,x=df.PURPOSE,y=df.minutes)
plt.xticks(rotation=45)
st.pyplot(ax)
ab=plt.figure(figsize=(16,7))
plt.subplot(1,2,1)
sns.boxplot(data=df,x=df.PURPOSE,y=df.MILES,showfliers=False)
plt.xticks(rotation=45)
plt.subplot(1,2,2)
sns.boxplot(data=df,x=df.PURPOSE,y=df.minutes,showfliers=False)
plt.xticks(rotation=45)
st.pyplot(ab)

fig,ax=plt.subplots(figsize=(8,5))
# Create a new column indicating whether a trip is round or not
df['Round_TRIP'] = df.apply(lambda x: 'yes' if x['START']==x['STOP'] else 'no', axis=1)
# Count the number of trips for each category of Round_TRIP and create a bar plot
round_trip_counts = df['Round_TRIP'].value_counts().sort_values(ascending=False)
round_trip_counts.plot(kind='bar', color='blue',ax=ax)
# Set the axis labels and title
plt.xlabel('Round Trip')
plt.ylabel('Count')
plt.title('Number of Round Trips')
st.pyplot(fig)
df['month']= pd.DatetimeIndex(df['START_DATE']).month
dic = {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sept',10:'oct',11:'nov',12:'dec'}
df['month'] =df['month'].map(dic)
st.write(df)
fig,ax=plt.subplots(figsize=(12,7))
sns.countplot(x='month', data=df, order=df['month'].value_counts().index, palette='deep',ax=ax)
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Number of Trips per Month')
st.pyplot(fig)
fig,ax=plt.subplots(figsize=(12,7))
round_trip_counts = df.groupby(['Round_TRIP', 'month'])['Round_TRIP'].count().unstack()
round_trip_counts.plot(kind='bar', stacked=True,ax=ax)
plt.xlabel('Round Trip')
plt.ylabel('Count')
plt.title('Number of Trips per Round Trip and Month')
st.pyplot(fig)
fig,ax=plt.subplots(1,2,figsize=(16,7))
sns.lineplot(data=df,x=df.minutes,y=df.MILES,ax=ax[0])
ax[0].set_title('Line Plot')
sns.scatterplot(data=df,x=df.minutes,y=df.MILES,ax=ax[1])
ax[1].set_title('Scatter Plot')
fig,ax=plt.subplots(figsize=(9,5))
sns.countplot(data=df,x='PURPOSE',hue='CATEGORY',dodge=False,ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
#Display the bar plots of start and stop locations
st.write('# Bar Plots of Start and Stop Locations')
fig,ax=plt.subplots(figsize=(15,4))
pd.Series(df['START']).value_counts()[:25].plot(kind='bar')
plt.title('Car rides start location frequency')
plt.xticks(rotation=45)
st.pyplot(fig)

fig,ax=plt.subplots(figsize=(15,4))
pd.Series(df['STOP']).value_counts()[:25].plot(kind='bar')
plt.title('cab rides stop location frequency')
plt.xticks(rotation=45)
st.pyplot(fig)
st.subheader('Conclusions')
st.write("1.Business cabs were not only used more in volumne but also have travelled more distance.")
st.write("2.Round trips were more in decemnber")
st.write("3.December can prove to be the best month for earning profit by raising fare as demand is more")
st.write("4.Seasonal pattern is there")
st.write("5.Cab traffic was high in just 5 cities comparitevely")
("6.Most of the cab rides are within a distance of 35 miles taking about 30 minutes")
("7.For Airport cabs are taking more time than usual.")
