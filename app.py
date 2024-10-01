import httpx
from flask import Flask, request, render_template, url_for
import pandas as pd 
from bs4 import BeautifulSoup
from textblob import TextBlob
import streamlit as st
import cleantext
import re
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
import io
import base64
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def clean_text(text):
    if not isinstance(text, str):
        return text
    # Convert to lowercase
    text = text.lower()
    # Remove emojis using a regular expression
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # Additional cleaning (e.g., removing URLs)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

#Calculating the sentiment values
def polarity_scores(text):
    encoded_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'neg' : scores[0],
        'neu' : scores[1],
        'pos' : scores[2]
    }
    return scores_dict

def get_emojis(score):
    if score > 0.5:
        return '😊'
    elif score < -0.5:
        return '😞'
    else:
        return '😐'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
        # Get user input
        celebrity_name = request.form['celebrity-name']
        clean_text(celebrity_name)
        print(celebrity_name)

        # Define URL with user input
        base_url = 'https://www.reddit.com/r/'
        category = '/hot'
        url = base_url + celebrity_name + category+'.json'
        print(url)

        # Define parameters for the API request
        params = {
            'limit': 100,
            't': 'year'  # time unit (hour, day, week, month, year, all)
        }

        # Make a single GET request to the Reddit API
        response = httpx.get(url, params=params)

        if response.status_code != 200:
            return "Failed load url", 500

        # Parse the JSON response
        json_data = response.json()

        # Extract the relevant data
        dataset = [rec['data'] for rec in json_data['data']['children']]
        df = pd.DataFrame(dataset)

        # Clean text and calculate sentiment
        df = df[['selftext']]
        df['selftext'] = df['selftext'].apply(clean_text)
        df_cleaned = df.dropna(how='all').replace('', pd.NA).dropna()

        df_cleaned[['neg', 'neu', 'pos']] = df_cleaned['selftext'].apply(lambda x: pd.Series(polarity_scores(x)))

        # Calculate overall sentiment score
        weights = {
            'negative': -1,
            'neutral': 0,
            'positive': 1
        }

        df_cleaned['overall_sentiment'] = (
            df_cleaned['neg'] * weights['negative'] +
            df_cleaned['neu'] * weights['neutral'] +
            df_cleaned['pos'] * weights['positive']
        )

        # Prepare data for the result template
        overall_score = df_cleaned['overall_sentiment'].mean()
        df_cleaned['emoji'] = df_cleaned['overall_sentiment'].apply(get_emojis)

        # Filter to only keep the comment and overall sentiment score
        df_filtered = df_cleaned[['selftext', 'overall_sentiment', 'emoji']]

        # Calculate sentiment distribution for pie chart
        positive_count = (df_cleaned['overall_sentiment'] > 0.5).sum()
        negative_count = (df_cleaned['overall_sentiment'] < -0.5).sum()
        neutral_count = ((df_cleaned['overall_sentiment'] >= -0.5) & (df_cleaned['overall_sentiment'] <= 0.5)).sum()

        # Create pie chart
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive_count, negative_count, neutral_count]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save pie chart to a BytesIO object and encode it to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        pie_chart = base64.b64encode(img.getvalue()).decode()


        return render_template('result.html', overall_score=overall_score, table=df_filtered.to_html(classes='table table-striped'), pie_chart=pie_chart)

if __name__ == '__main__':
    app.run(debug=True)