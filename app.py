# Step 1: Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st  # For building the web app
import plotly.graph_objects as go  # For creating interactive plots
import json
import requests  # For making HTTP requests (to load animations)
from streamlit_lottie import st_lottie  # For adding Lottie animations

# Step 2: Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained RNN model
model = load_model('simple_rnn_imdb.h5')

# Step 3: Define helper functions
def decode_review(encoded_review):
    """Decode encoded review into human-readable text."""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    """Preprocess user input text into a format suitable for the model."""
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def load_lottie_url(url: str):
    """Load a Lottie animation from a URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Step 4: Set up the Streamlit app
# Add custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background: url("https://images.unsplash.com/photo-1496065187959-7f07b022a32d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
    }
    .stButton>button {
        background-color: #1e90ff;
        color: white;
        border-radius: 10px;
        font-size: 16px;
    }
    .stTextArea>label {
        color: #1e90ff;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# Set the title and instructions for the app
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below and click 'Classify' to determine its sentiment!")

# Step 5: Add user input and Lottie animation
# Load Lottie animation (replace with a valid URL or use a local file)
lottie_animation = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_q5pk6p1k.json")

# Display Lottie animation if loaded correctly
if lottie_animation:
    st_lottie(lottie_animation, height=200, key="sentiment-animation")

# User input for the movie review
user_input = st.text_area(
    "Enter your movie review here:",
    placeholder="Type your review...",
    height=150
)

# Add a slider for adjusting the sentiment threshold
threshold = st.slider("Adjust Sentiment Threshold", 0.0, 1.0, 0.5, 0.01)

# Step 6: Predict sentiment and display results
if st.button("Classify"):
    with st.spinner("Analyzing sentiment..."):
        # Preprocess the input text
        preprocessed_input = preprocess_text(user_input)
        
        # Predict sentiment using the pre-trained model
        prediction = model.predict(preprocessed_input)
        
        # Determine the sentiment based on the threshold
        sentiment = "Positive" if prediction[0][0] > threshold else "Negative"
        
        # Create two columns for displaying the results
        col1, col2 = st.columns(2)
        
        with col1:
            ## Display results based on the sentiment prediction

        # Show snow for negative sentiment and success message for positive sentiment
          if sentiment == "Positive":
            st.success("😊 Positive")
          else:
            st.snow()
          st.error("😞 Negative")

        with col2:
            # Display the prediction score as a metric
            st.metric(label="Prediction Confidence", value=f"{prediction[0][0] * 100:.2f}%")

        # Create a gauge chart to visually show the prediction score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction[0][0] * 100,  # Convert the score to percentage
            title={'text': "Prediction Score (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "lightblue"}}
        ))

        # Display the gauge chart
        st.plotly_chart(fig)

        # Add an expander for detailed analysis
        with st.expander("See Detailed Analysis"):
            st.write("Decoded Review:")
            st.write(decode_review(preprocessed_input[0]))
            st.write("This review has been converted into a numerical format and processed through the RNN model.")

# Display a message if the button hasn't been clicked yet
else:
    st.write("Please enter a movie review and click 'Classify' to analyze its sentiment.")
