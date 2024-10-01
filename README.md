# Sentiment Analysis Website

This project is a web application for performing sentiment analysis on Reddit posts related to a specific celebrity. It uses machine learning models and APIs to extract, clean, and analyze text data, then visualize the sentiment distribution through a pie chart.

## Features

- Fetches Reddit posts related to a user-input celebrity name.
- Cleans text data by removing unnecessary elements (like emojis and URLs).
- Performs sentiment analysis using the `twitter-roberta-base-sentiment-latest` model from Hugging Face.
- Displays the overall sentiment score and emoji representation of the sentiment.
- Visualizes the sentiment distribution with a pie chart.

## Tools and Technologies

- **Flask**: Web framework for building the server-side logic.
- **httpx**: For making API requests to Reddit.
- **TextBlob & CleanText**: For text cleaning and preprocessing.
- **Hugging Face Transformers**: Pretrained model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) for sentiment analysis.
- **Streamlit**: Provides additional user-friendly components.
- **Matplotlib**: For generating pie charts to visualize sentiment.
- **BeautifulSoup**: Web scraping tool (not used in the current version).
- **Pandas**: Data manipulation and analysis.

## Setup

### Prerequisites

- Python 3.x
- Pip (Python package installer)
- A Reddit account (for testing purposes)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/preethikrdy/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**:

   ```bash
   python app.py
   ```

5. **Access the app**:

   Navigate to `http://127.0.0.1:5000/` in your web browser.

## Usage

1. Enter a celebrity's name on the home page.
2. The app will fetch Reddit posts related to the celebrity.
3. Sentiment analysis will be performed on the fetched text data.
4. Results will be displayed with:
   - **Overall Sentiment Score**: A numerical score indicating the general sentiment (positive, neutral, or negative).
   - **Detailed Sentiment Analysis**: A table showing individual post sentiments and corresponding emojis.
   - **Sentiment Distribution**: A pie chart showing the proportion of positive, neutral, and negative sentiments.

## How It Works

1. **Text Cleaning**:
   - The input text is cleaned by converting to lowercase and removing emojis and URLs.
   
2. **Sentiment Analysis**:
   - Text is tokenized using `AutoTokenizer`, and sentiment scores are predicted using the `AutoModelForSequenceClassification` model.
   - The sentiment is categorized as negative, neutral, or positive using softmax activation.
   
3. **Visualization**:
   - The sentiment distribution (positive, neutral, negative) is plotted as a pie chart using `matplotlib`.
