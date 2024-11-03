This project evaluates student responses to questions using a video-to-text process, NLP, and machine learning. It compares a student's spoken answers to expected responses, providing feedback on similarity.
Project Overview

    Framework: Flask (Python-based web framework)
    Purpose: Convert student answers from video files to text, preprocess them, and evaluate similarity with expected answers.
    Primary Libraries:
        speech_recognition for audio-to-text conversion
        sklearn for TF-IDF vectorization and cosine similarity
        nltk for text preprocessing

Features

    Video Upload: Users upload a video of their response.
    Audio Extraction: The video’s audio is extracted using FFmpeg.
    Speech-to-Text: The audio is converted to text via Google’s Speech Recognition API.
    Text Preprocessing: Converts the text to lowercase, removes punctuation, tokenizes, removes stopwords, and applies stemming.
    Similarity Calculation: Compares the response with expected answers using TF-IDF vectorization and cosine similarity.
    Question Navigation: Provides functionality to get the next question from the dataset.

Project Structure

    app.py: Main Flask application file that includes routes for handling video uploads, processing responses, and fetching questions.
    preprocess_data.pkl: Preprocessed dataset file containing questions and expected answers.

Files and Directories

    uploads/: Directory to store uploaded video files temporarily.
    templates/: Contains the HTML templates for displaying the application in a web browser.

Setup Instructions
Prerequisites

    Python: Version 3.x
    FFmpeg: Install FFmpeg and update the path in the extract_audio function in app.py.

Libraries

Install required libraries using the following command:

bash

pip install Flask SpeechRecognition nltk sklearn

Run the Application

    Clone this repository.
    Update the ffmpeg_path variable in app.py to the location of your FFmpeg executable.
    Start the Flask app:

bash

python app.py

    Open your browser and go to http://127.0.0.1:5000 to access the application.

Code Explanation
Key Functions in app.py

    extract_audio(video_path, audio_path): Extracts audio from a video file using FFmpeg.
    convert_audio_to_text(audio_path): Converts extracted audio to text.
    preprocess_text(text): Preprocesses text by converting to lowercase, removing non-alphanumeric characters, tokenizing, removing stopwords, and stemming.
    calculate_similarity(user_ans, question_ans): Compares student response with the expected answer using cosine similarity on TF-IDF vectors.

Routes

    / (GET & POST): Main page to upload videos, process responses, and display similarity scores.
    /get_next_question: Fetches the next question from the dataset.
    /get_question: Fetches a specific question from the dataset by index.

Example Use Case

    A student uploads a video response to a question.
    The application extracts and converts the audio to text.
    Text is preprocessed and compared with the model answer.
    A similarity score is displayed, giving feedback on response accuracy.

Future Improvements

    Add support for multi-language speech recognition.
    Incorporate more advanced NLP techniques for a deeper understanding of student responses.
    Optimize similarity calculations for larger datasets.
