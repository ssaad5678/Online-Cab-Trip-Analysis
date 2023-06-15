import csv
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score,silhouette_samples,silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


image=Image.open('Poster For Mini Project.jpg')
st.image(image,use_column_width=True)
# Define some CSS styles
def load_css():
    with open("designing.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the load_css function
load_css()
# Display some content
st.markdown('<div id="sentence1">Online Cab Trip Analysis</div>', unsafe_allow_html=True)
# Add a file uploader button
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file, encoding='latin1')
    df.columns = df.columns.str.replace("*", "")
    st.write('Analysis of Uploaded DataSet')

#--------> seperate into pages ----->

# Define the number of pages
num_pages = 7

# Check if the 'current_page' session state variable exists
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

# Create the sidebar navigation buttons
button_names = ["Introduction", "Dataset", "Missing Values", "Data Analysis", "Kmeans Analysis", "Conclusions","Thank You"]

st.sidebar.header("Page Navigation")
for page in range(1, num_pages + 1):
    if st.sidebar.button(button_names[page-1]):
        st.session_state.current_page = page

# Check the current page and display the corresponding content
if st.session_state.current_page == 1:
    st.write(" ")
elif st.session_state.current_page == 2:
    # Page 2 content
    st.write("Datasets")
    # Reading data
    st.write('Simple DataSet')
    st.write(df.head())
    st.write('The Complete Data Set')
    
    styled_df = df.style \
    .set_properties(**{'border-color': 'red', 'background-color': 'lightyellow'}) \
    .set_table_styles([{'selector': 'th', 'props': [('border-color', 'blue')]}])

    # Display the styled DataFrame
    st.dataframe(styled_df)

elif st.session_state.current_page == 3:
    # Page 3 content
    st.write("Page 3")
    #-------------->Page 3
    # Missing Data
    st.markdown('<div id="sentence2">Missing Data</div>', unsafe_allow_html=True)
    st.write('Heatmap of missing data')
    fig=plt.figure(figsize=(15,10))
    sns.heatmap(df.isnull(), cmap='magma', yticklabels=False)
    st.pyplot(fig)

    st.markdown('<div id="sentence3">Barchart of Missing Data</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    msno.bar(df, ax=ax, color="#0083B8")  # Set the color of the bars

    ax.set_title("Barchart of Missing Data")  # Set the title of the plot
    ax.set_xlabel("Variables")  # Set the x-axis label
    ax.set_ylabel("Missing Data")  # Set the y-axis label

    st.pyplot(fig)


    # Handling missing data
    last_index = df.index[-1]
    df.drop(index=last_index, axis=0, inplace=True)
    df['PURPOSE'].fillna(method='ffill', inplace=True)

    # Data cleaning
    df['START_DATE*'] = pd.to_datetime(df['START_DATE'], errors='coerce')
    df['END_DATE*'] = pd.to_datetime(df['END_DATE'], errors='coerce')

    # Categorical data
    st.markdown('<div id="sentence4">Categorical Data</div>', unsafe_allow_html=True)
    category = pd.crosstab(index=df['CATEGORY'], columns='count of travel as per category')
    st.write(category)
    fig, ax=plt.subplots()
    category.plot(kind='bar', color='r',ax=ax)
    plt.legend()
    st.pyplot(fig)

elif st.session_state.current_page == 4:
    # Page 4 content
    st.write("Page 4")
    #-------------->Page 4
    # Reading data
    st.markdown('<div id="sentence5">Start Points with More Than 10 Trips</div>', unsafe_allow_html=True)
    start_point = df.START.value_counts()
    st.write(start_point[start_point>10])
    fig, ax=plt.subplots()
    start_point[start_point>10].plot(kind='pie', shadow=True,ax=ax)
    st.pyplot(fig)

    st.markdown('<div id="sentence6">Start Points with 10 or Fewer Trips</div>', unsafe_allow_html=True)
    st.write(start_point[start_point<=10])

    st.markdown('<div id="sentence7">Stop Points with more than 10 trips</div>', unsafe_allow_html=True)
    stop_point = df.STOP.value_counts()
    st.write(stop_point[stop_point>10])

    st.markdown('<div id="sentence8">Stop Points with 10 or Fewer Trips</div>', unsafe_allow_html=True)
    st.write(stop_point[stop_point<=10])

    st.markdown('<div id="sentence9">Miles With More Than 10 Trips</div>', unsafe_allow_html=True)
    miles = df.MILES.value_counts()
    st.write(miles[miles>10])
    fig,ax=plt.subplots()
    miles[miles>10].plot(kind='bar',ax=ax)
    st.pyplot(fig)

    st.markdown('<div id="sentence10">Miles with 10 or fewer trips</div>', unsafe_allow_html=True)
    st.write(miles[miles<=10])
    data = {'MILES': [840, 31],
        'Round_TRIP': [False, True]}

    dfe = pd.DataFrame(data)

    st.dataframe(dfe)

    miles = pd.crosstab(index=df['MILES']>10, columns='count of miles')
    fig, ax=plt.subplots()
    miles.plot(kind='bar', color='r',ax=ax)
    st.pyplot(fig)

    st.markdown('<div id="sentence11">Trips Per Purpose</div>', unsafe_allow_html=True)
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
    df['END_DATE'] = pd.to_datetime(df['END_DATE'])
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')
    df['minutes'] = df['END_DATE'] - df['START_DATE']

    df['minutes'] = df['minutes'].dt.total_seconds()/60

    st.markdown('<div id="sentence12">Trip Duration</div>', unsafe_allow_html=True)
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
    st.markdown('<div id="sentence13">Bar Plots of Start and Stop Locations</div>', unsafe_allow_html=True)
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


elif st.session_state.current_page == 5:
    # Page 6 content
    st.write("Page 5")
    #-------------->Page 5
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key="csv_uploader")

    if uploaded_file is not None:
        data = []
        file_wrapper = io.TextIOWrapper(uploaded_file, encoding='utf-8')
        reader = csv.reader(file_wrapper)
        header = next(reader)  # Read header row
        for row in reader:
            data.append(row)

    # Extract the features for clustering (START, STOP, MILES)
    features = np.array([[row[3], row[4], row[5]] for row in data])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for i in range(3):
        features[:, i] = label_encoder.fit_transform(features[:, i])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    # Elbow Method to determine the optimal number of clusters
    wcss = []
    max_clusters = 10
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
    fig,ax=plt.subplots()
    # Plotting the WCSS values against the number of clusters
    ax.plot(range(1, max_clusters + 1), wcss)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('WCSS')
    ax.set_title('Elbow Method')
    st.pyplot(fig)

    # Silhouette Analysis
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        cluster_labels = kmeans.fit_predict(features)
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    fig,ax=plt.subplots()

    # Plotting the silhouette scores
    ax.plot(range(2, max_clusters + 1), silhouette_scores)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Analysis')
    st.pyplot(fig)

    # Perform K-means clustering with the optimal number of clusters
    optimal_k = 20  # Set the optimal number of clusters based on the analysis
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(features)

    # Get the predicted cluster labels
    predicted_labels = kmeans.labels_

    # Extract the ground truth labels
    ground_truth_labels = np.array([int(row[7]) for row in data])

    # Calculate the accuracy
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)

    # Calculate the confusion matrix
    cm = confusion_matrix(ground_truth_labels, predicted_labels)

    # Display the accuracy and confusion matrix
    st.write("Confusion Matrix:")
    st.write(cm)
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy*1000,2))    
elif st.session_state.current_page == 6:
    # Page 6 content
    st.write("Page 6")
    st.markdown('<div id="conclusion">Conclusions</div>', unsafe_allow_html=True)
    # Create a Pandas DataFrame with the text
    data = {'Data Analysis': ['Business cabs were not only used more in volume but also have traveled more distance.',
                'Round trips were more in December.',
                'December can prove to be the best month for earning profit by raising fare as demand is more.',
                'Seasonal pattern is there.',
                'Cab traffic was high in just 5 places comparatively.',
                'Most of the cab rides are within a distance of 35 miles taking about 30 minutes.',
                'For Airport cabs are taking more time than usual.']}
    df = pd.DataFrame(data)
    # Display the DataFrame as a table
    st.table(df)
elif st.session_state.current_page == 7:
    # Page 7 content
    st.write("Page 6")
    #-------------->Page 7
    image=Image.open('Front.png')
    st.image(image,use_column_width=True)
