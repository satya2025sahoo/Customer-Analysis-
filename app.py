import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # For revenue forecasting


# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

data = load_data()

# Convert 'Time' to datetime
data['Time'] = pd.to_datetime(data['Time'])

# Sentiment analysis
def get_sentiment(feedback):
    if pd.isna(feedback):
        return "Neutral"
    analysis = TextBlob(feedback)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

data['Sentiment'] = data['Feedback'].apply(get_sentiment)


# Login system
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    # Add a login button
    if st.sidebar.button("Login"):
        if username == "amit" and password == "123":
            st.sidebar.success("Login successful!")
            return True
        else:
            st.sidebar.error("Incorrect username or password")
            return False
    return False  # Default return value if the button is not clicked

# Check login status
if not login():
    st.stop()  # Stop the app if login is not successful

# Dashboard
st.title("🚀 Bangalore Kitchen - Advanced Revenue and Feedback Analysis Dashboard")

# Filters
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", data['Time'].min())
end_date = st.sidebar.date_input("End Date", data['Time'].max())
selected_sentiment = st.sidebar.multiselect("Select Sentiment", data['Sentiment'].unique(), default=data['Sentiment'].unique())
selected_indian_dish = st.sidebar.multiselect("Select Indian Dish", data['Indian Dishes'].dropna().unique(), default=data['Indian Dishes'].dropna().unique())
selected_english_dish = st.sidebar.multiselect("Select English Dish", data['English Dishes'].dropna().unique(), default=data['English Dishes'].dropna().unique())
selected_beverage = st.sidebar.multiselect("Select Beverage", data['Beverages'].dropna().unique(), default=data['Beverages'].dropna().unique())

# Apply filters
filtered_data = data[
    (data['Time'].dt.date >= start_date) &
    (data['Time'].dt.date <= end_date) &
    (data['Sentiment'].isin(selected_sentiment)) &
    (data['Indian Dishes'].isin(selected_indian_dish)) &
    (data['English Dishes'].isin(selected_english_dish)) &
    (data['Beverages'].isin(selected_beverage))
]

# Sentiment Analysis
st.header("📊 Customer Sentiment Analysis")
sentiment_counts = filtered_data['Sentiment'].value_counts(normalize=True) * 100  # Convert to percentage
st.write(sentiment_counts)

fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution (%)")
st.plotly_chart(fig)

# Most Ordered Items
st.header("🍽️ Most Ordered Items")

# Indian Dishes
st.subheader("Most Ordered Indian Dishes (%)")
indian_dishes = (filtered_data['Indian Dishes'].value_counts(normalize=True) * 100).head(5)  # Convert to percentage
fig = px.bar(indian_dishes, x=indian_dishes.index, y=indian_dishes.values, labels={'x': 'Dish', 'y': 'Percentage'}, title="Top 5 Indian Dishes (%)")
st.plotly_chart(fig)

# English Dishes
st.subheader("Most Ordered English Dishes (%)")
english_dishes = (filtered_data['English Dishes'].value_counts(normalize=True) * 100).head(5)  # Convert to percentage
fig = px.bar(english_dishes, x=english_dishes.index, y=english_dishes.values, labels={'x': 'Dish', 'y': 'Percentage'}, title="Top 5 English Dishes (%)")
st.plotly_chart(fig)

# Beverages
st.subheader("Most Ordered Beverages (%)")
beverages = (filtered_data['Beverages'].value_counts(normalize=True) * 100).head(5)  # Convert to percentage
fig = px.bar(beverages, x=beverages.index, y=beverages.values, labels={'x': 'Beverage', 'y': 'Percentage'}, title="Top 5 Beverages (%)")
st.plotly_chart(fig)

# Revenue Analysis
st.header("💰 Advanced Revenue Analysis")

# Revenue by Day
st.subheader("Daily Revenue Trend")
revenue_by_day = filtered_data.groupby(filtered_data['Time'].dt.date)['Amount'].sum().reset_index()
fig = px.line(revenue_by_day, x='Time', y='Amount', title="Daily Revenue Trend")
st.plotly_chart(fig)

# Revenue by Dish and Beverage
st.subheader("Revenue Contribution by Dish and Beverage")
revenue_by_dish_beverage = filtered_data.groupby(['Indian Dishes', 'English Dishes', 'Beverages'])['Amount'].sum().reset_index()
revenue_by_dish_beverage = revenue_by_dish_beverage.sort_values(by='Amount', ascending=False).head(10)
fig = px.bar(revenue_by_dish_beverage, x='Amount', y='Indian Dishes', color='Beverages', title="Top Revenue-Generating Combinations")
st.plotly_chart(fig)

# Revenue by Sentiment
st.subheader("Revenue by Customer Sentiment")
revenue_by_sentiment = filtered_data.groupby('Sentiment')['Amount'].sum().reset_index()
fig = px.pie(revenue_by_sentiment, values='Amount', names='Sentiment', title="Revenue Contribution by Sentiment")
st.plotly_chart(fig)

# Revenue by Time of Day
st.subheader("Revenue by Time of Day")
filtered_data['Hour'] = filtered_data['Time'].dt.hour
revenue_by_hour = filtered_data.groupby('Hour')['Amount'].sum().reset_index()
fig = px.bar(revenue_by_hour, x='Hour', y='Amount', title="Revenue by Hour of the Day")
st.plotly_chart(fig)

# Revenue by Customer Segment
st.subheader("Revenue by Customer Segment")
customer_spending = filtered_data.groupby('Phone Number')['Amount'].sum().reset_index()
customer_spending['Segment'] = pd.cut(customer_spending['Amount'], bins=[0, 100, 200, np.inf], labels=['Low', 'Medium', 'High'])
revenue_by_segment = customer_spending.groupby('Segment')['Amount'].sum().reset_index()
fig = px.pie(revenue_by_segment, values='Amount', names='Segment', title="Revenue Contribution by Customer Segment")
st.plotly_chart(fig)

# Revenue Forecasting
st.subheader("Revenue Forecasting")
if len(revenue_by_day) < 14:  # Less than 2 full weeks
    st.warning("Not enough data for seasonal forecasting. Showing trend-only forecast.")
    model = ExponentialSmoothing(revenue_by_day['Amount'], trend='add', seasonal=None)
else:
    model = ExponentialSmoothing(revenue_by_day['Amount'], trend='add', seasonal='add', seasonal_periods=7)

try:
    # Prepare data for forecasting
    revenue_by_day = filtered_data.groupby(filtered_data['Time'].dt.date)['Amount'].sum().reset_index()
    revenue_by_day.set_index('Time', inplace=True)
    revenue_by_day.index = pd.to_datetime(revenue_by_day.index)

    # Fit Holt-Winters model
    #model = ExponentialSmoothing(revenue_by_day['Amount'], trend='add', seasonal='add', seasonal_periods=7)
    fit = model.fit()
    forecast = fit.forecast(steps=2)  # Forecast next 2 days

    # Plot forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=revenue_by_day.index, y=revenue_by_day['Amount'], name="Historical Revenue"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Forecasted Revenue"))
    fig.update_layout(title="7-Day Revenue Forecast", xaxis_title="Date", yaxis_title="Revenue")
    st.plotly_chart(fig)
except Exception as e:
    st.warning(f"Revenue forecasting failed: {e}")

# Customer Segmentation
st.header("👥 Customer Segmentation")
segment_percentage = customer_spending['Segment'].value_counts(normalize=True) * 100  # Convert to percentage
fig = px.pie(segment_percentage, names=segment_percentage.index, values=segment_percentage.values, title="Customer Spending Segments (%)")
st.plotly_chart(fig)

# Predictive Insights
st.header("🔮 Predictive Insights")
# Prepare data for prediction
X = filtered_data[['Amount']]
y = filtered_data['Sentiment'].apply(lambda x: 1 if x == 'Happy' else 0)

# Train a simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy:.2f}")
st.write("This model predicts customer satisfaction (Happy or Not Happy) based on the order amount.")

# Feedback Trends Over Time
st.header("📈 Feedback Trends Over Time")
feedback_trends = filtered_data.groupby([filtered_data['Time'].dt.date, 'Sentiment']).size().unstack().fillna(0)
feedback_trends_percentage = feedback_trends.div(feedback_trends.sum(axis=1), axis=0) * 100  # Convert to percentage
fig = px.line(feedback_trends_percentage, x=feedback_trends_percentage.index, y=feedback_trends_percentage.columns, title="Feedback Sentiment Over Time (%)")
st.plotly_chart(fig)

# Best-Selling Combinations
st.header("🍴 Best-Selling Combinations of Indian/English Dishes with Beverages (%)")

# Group by Indian Dishes, English Dishes, and Beverages
combinations = filtered_data.groupby(['Indian Dishes', 'English Dishes', 'Beverages']).size().reset_index(name='Count')
combinations['Percentage'] = (combinations['Count'] / combinations['Count'].sum()) * 100  # Convert to percentage

# Find the best-selling combination
best_combination = combinations.loc[combinations['Count'].idxmax()]
st.write(f"Best-Selling Combination: {best_combination['Indian Dishes']} (Indian), {best_combination['English Dishes']} (English), {best_combination['Beverages']} (Beverage) with {best_combination['Percentage']:.2f}% of orders")

# Display top 10 combinations
st.subheader("Top 10 Combinations (%)")
top_combinations = combinations.sort_values(by='Count', ascending=False).head(10)
st.write(top_combinations[['Indian Dishes', 'English Dishes', 'Beverages', 'Percentage']])

# Visualize top combinations
fig = px.bar(top_combinations, x='Percentage', y='Indian Dishes', color='Beverages', title="Top 10 Combinations of Indian/English Dishes with Beverages (%)")
st.plotly_chart(fig)

import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

nltk.download('punkt_tab')

def summarize_feedback(feedback_list, num_sentences=3):
    combined_text = " ".join(feedback_list)
    sentences = sent_tokenize(combined_text)
    word_freq = Counter(combined_text.split())

    ranked_sentences = sorted(sentences, key=lambda s: sum(word_freq[word] for word in s.split()), reverse=True)
    return " ".join(ranked_sentences[:num_sentences])

feedbacks = data['Feedback']

feedbacks = data['Feedback'].dropna()  # Drop missing values

if not feedbacks.empty:  # Proper check for pandas Series
    summary = summarize_feedback(feedbacks.tolist())  # Convert Series to list
    st.subheader("📝 Summarized Feedback")
    st.write(summary)


# Extract improvement sentences
def extract_improvement_sentences(feedback):
    improvement_keywords = ['should', 'could', 'improve', 'suggest', 'recommend', 'better', 'need', 'would', 'advise', 'propose', "should", "could", "need", "improve", "better", "recommend", "suggest", "wish", "hope", "would be great", "it would be better"]
    sentences = sent_tokenize(feedback)
    improvement_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in improvement_keywords)]
    return improvement_sentences

# Apply the function to the Feedback column
data['Improvement_Sentences'] = data['Feedback'].apply(extract_improvement_sentences)

# Filter out rows where no improvement sentences were found
improvement_data = data[data['Improvement_Sentences'].apply(len) > 0]

# Display the improvement sentences
im = data['Improvement_Sentences'].dropna()

if not im.empty:  # Proper check for pandas Series
    # Flatten the list of improvement sentences
    flattened_improvement_sentences = [sentence for sublist in im for sentence in sublist]
    
    # Summarize the flattened list
    summary = summarize_feedback(flattened_improvement_sentences)  # Pass the flattened list
    
    # Split the summary into sentences
    summary_sentences = sent_tokenize(summary)
    
    # Join the sentences with a comma separator
    comma_separated_summary = ", ".join(summary_sentences)
    
    # Display the summarized improvement sentences
    st.subheader("📝 Summarized Improvement")
    st.write(comma_separated_summary)


# Raw Data
st.header("📂 Raw Data")
st.write(filtered_data)