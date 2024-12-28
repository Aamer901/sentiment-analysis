import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download necessary NLTK data files (run only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset
data = pd.read_csv("burj_al_arab_reviews.csv")

# Preprocessing function
def preprocess_text(text):
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text.lower())  # Convert to lowercase and tokenize
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize each word (reduce it to its root form)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Create a 'sentiment' column based on ratings (1 for positive, 0 for negative)
data['sentiment'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Split the data into train and test sets
X = data['text']
y = data['sentiment']

# Preprocess the text data
X = X.apply(preprocess_text)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)  # Fit and transform on the training data
X_test_tfidf = tfidf.transform(X_test)  # Transform the test data

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict sentiment on the test data
y_pred = model.predict(X_test_tfidf)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Set page layout and background image
st.set_page_config(page_title="Burj Al Arab Reviews", page_icon="🏨", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-image: url('http://localhost:8888/files/sentiment%20hotel/image/hotelback.jpg?_xsrf=2%7Cdf246835%7C22584ada6877df6919d259f66c4e0d24%7C1735312272');
        background-size: cover;
        background-position: center;
    }

    /* Make all the text white */
    .stTextInput input,
    .stTextArea textarea,
    .stButton button,
    .stTitle,
    .stHeader,
    .stSubheader,
    .stMarkdown,
    .stText,
    .stRadio label,
    .stSelectbox label {
        color: white !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: white !important;
    }

    /* Styling for title and box */
    .title {
        font-size: 30px;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 20px;
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
    }

    .box {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 10px;
        border-radius: 10px;
        color: black;
        margin: 10px;
        font-size: 14px;  /* Smaller accuracy box font size */
        max-width: 300px;  /* Reduce width of the accuracy box */
    }

    .prediction {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        color: black;  /* Output text in black */
    }

    /* Make the button text black */
    .stButton button {
        color: black !important;
    }

    /* Make the input/output text black */
    .stTextInput input,
    .stTextArea textarea {
        color: black !important;
    }

    /* Styling for the instructions text */
    .instructions {
        font-size: 22px;  /* Larger font size for the review text */
        font-weight: normal;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Styling for text
st.markdown('<div class="title">Dubai\'s 7-Star Icon Burj Al Arab Reviews</div>', unsafe_allow_html=True)

# Display model accuracy in a smaller box
st.markdown(f"<div class='box'>Model Accuracy: {accuracy*100:.2f}%</div>", unsafe_allow_html=True)

# Function to predict sentiment
def predict_sentiment(review, model, tfidf):
    # Preprocess the review text
    review_cleaned = preprocess_text(review)
    # Convert the review to tf-idf features (transform using the already fitted vectorizer)
    review_tfidf = tfidf.transform([review_cleaned])
    # Predict sentiment (0 = Negative, 1 = Positive)
    sentiment = model.predict(review_tfidf)
    return sentiment

# Streamlit app for user input and prediction
st.markdown('<div class="instructions">Enter a hotel review below and the model will predict whether it\'s positive or negative.</div>', unsafe_allow_html=True)

# User input
user_review = st.text_area("Enter your hotel review here:")

# Predict and display sentiment
if st.button("Predict Sentiment"):
    if user_review:
        predicted_sentiment = predict_sentiment(user_review, model, tfidf)
        if predicted_sentiment == 1:
            sentiment_text = "<div class='prediction' style='color: green;'>Predicted Sentiment for the review: Positive</div>"
        else:
            sentiment_text = "<div class='prediction' style='color: red;'>Predicted Sentiment for the review: Negative</div>"
        st.markdown(sentiment_text, unsafe_allow_html=True)
    else:
        st.write("Please enter a review to predict sentiment.")
