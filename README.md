## IMDB Movie Review Sentiment Analysis

#### Overview
This project is a web application that analyzes the sentiment of user-inputted movie reviews using a pre-trained Recurrent Neural Network (RNN) model. It determines whether a review is Positive or Negative based on the model's prediction and a user-adjustable threshold.

The app is built using:

1. TensorFlow/Keras for loading the pre-trained RNN model.
2. Streamlit for building an interactive web application.
3. Plotly for visualizing prediction confidence.
4. Lottie for adding animations.

### Features
- User Input: Type in a movie review to classify its sentiment.
- Sentiment Prediction: Uses a pre-trained RNN model to classify reviews as Positive or Negative.
- Customizable Threshold: Adjust the sentiment threshold to control the sensitivity of the model.
- Visualization: Displays the prediction confidence with a gauge chart.
- Interactive UI: Includes Lottie animations and a visually appealing layout.

## Installation and Setup
### Prerequisites
- Python 3.11
- Virtual environment setup (optional but recommended)

## Step 1: Clone the Repository

``` bash git clone https://github.com/sm0311/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

```

## Step 2: Set Up a Virtual Environment
Set up a virtual environment to keep dependencies isolated:

``` bash python -m venv venv ```

## Step 3: Install Required Packages

## Install the necessary libraries from the requirements.txt file:

``` bash
pip install -r requirements.txt

```

## Step 5: Run the Application
Start the Streamlit application by running the following command:

``` bash
streamlit run app.py

```

## Step 4: Download the Pre-trained Model

Download the simple_rnn_imdb.h5 model file and place it in the project directory. This file is necessary for making predictions.

## Step 5: Run the Application
Start the Streamlit application by running the following command:

``` bash
streamlit run app.py 
```

## Usage
- Enter a Review: Type a movie review into the text area.
- Adjust Threshold: Use the slider to adjust the threshold for sentiment classification.
- Click "Classify": Click the "Classify" button to see the sentiment prediction.
- View Results: The app will display the sentiment, confidence score, and a detailed analysis of the review.

## Project Structure

imdb-sentiment-analysis/
│
├── app.py                 # Main Streamlit app code
├── simple_rnn_imdb.h5     # Pre-trained RNN model (place in the root directory)
├── requirements.txt       # List of required Python packages
├── README.md              # Project documentation
└── assets/                # Folder for additional assets (e.g., images, animations)
