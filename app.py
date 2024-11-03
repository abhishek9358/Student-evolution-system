from flask import Flask, render_template, request, jsonify
import os
import subprocess
import speech_recognition as sr
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def extract_audio(video_path, audio_path):
    try:
        # Provide the correct path to the FFmpeg executable
        ffmpeg_path = "C:/desktop_item/AI_ML_Internship/Student Evalution using AI/Student Evalution using AI/FFMPEG/ffmpeg-2024-04-25-git-cae0f2bc55-full_build/bin/ffmpeg.exe"  # Update with the correct path to FFmpeg
        command = [ffmpeg_path, '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("Error extracting audio:", e)
        return False

def convert_audio_to_text(audio_path):
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

def calculate_similarity(user_ans, question_ans):
    cleaned_user_ans = preprocess_text(user_ans)
    cleaned_question_ans = preprocess_text(question_ans)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_user_ans, cleaned_question_ans])
    
    cosine_similarity_score = cosine_similarity(tfidf_matrix)[0][1]
    
    return cosine_similarity_score

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        question_index = request.args.get('question_index', 0, type=int)

        new_ds = pickle.load(open('C:/desktop_item/AI_ML_Internship/Student Evalution using AI/Student Evalution using AI/preprocess_data.pkl', 'rb'))

        if question_index >= len(new_ds):
            question_index = 0

        question_text = new_ds.loc[question_index, 'question']
        question_ans = new_ds.loc[question_index, 'ans1']

        return render_template('index.html', question=question_text, question_index=question_index, expected_answer=question_ans, similarity_score=None)
    
    elif request.method == 'POST':
        try:
            if 'video' not in request.files:
                return jsonify({'error': 'No video file uploaded'})

            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({'error': 'No selected file'})

            video_filename = video_file.filename
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video_file.save(video_path)
            print("Received video file:", video_filename)
            print("Video saved to:", video_path)

            audio_filename = os.path.splitext(video_filename)[0] + '.wav'
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

            if extract_audio(video_path, audio_path):
                print("Audio extraction successful")
                text = convert_audio_to_text(audio_path)
                print("Video Text:", text)
                os.remove(video_path)
                os.remove(audio_path)

                new_ds = pickle.load(open('C:/desktop_item/AI_ML_Internship/Student Evalution using AI/Student Evalution using AI/preprocess_data.pkl', 'rb'))

                question_index = int(request.form.get('question_index', 0))
                expected_answer = new_ds.loc[question_index, 'ans1']
                original_answer = new_ds.loc[question_index, 'ans']

                similarity_score = calculate_similarity(text, expected_answer)

                print("Expected Answer:", original_answer)
                print("Similarity Score:", round(similarity_score * 100, 2))

                return jsonify({
                    'text': text,
                    'top_match': original_answer,
                    'similarity_score': round(similarity_score * 100, 2),
                    'question_index': question_index
                })
            else:
                return jsonify({'error': 'Audio extraction failed'})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/get_next_question', methods=['GET'])
def get_next_question():
    try:
        new_ds = pickle.load(open('C:/desktop_item/AI_ML_Internship/Student Evalution using AI/Student Evalution using AI/preprocess_data.pkl', 'rb'))
        question_index = request.args.get('question_index', type=int)

        total_questions = len(new_ds)
        question_index = (question_index + 1) % total_questions

        question_text = new_ds.loc[question_index, 'question']
        question_ans = new_ds.loc[question_index, 'ans1']
        original_answer = new_ds.loc[question_index, 'ans']

        return jsonify({
            'question': question_text,
            'question_index': question_index,
            'question_ans': question_ans,
            'original_answer': original_answer
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_question', methods=['GET'])
def get_question():
    # Load the dataset
    new_ds = pickle.load(open('C:/desktop_item/AI_ML_Internship/Student Evalution using AI/Student Evalution using AI/preprocess_data.pkl', 'rb'))

    # Get the question index from the request or set it to 0 by default
    question_index = request.args.get('question_index', 0, type=int)

    # Get the question text and answer for the specified question index
    question_text = new_ds.loc[question_index, 'question']
    question_ans = new_ds.loc[question_index, 'ans']

    # Return the question, answer, and the question index as a JSON response
    return jsonify({'question': question_text, 'question_index': question_index, 'question_ans': question_ans})

if __name__ == '__main__':
    app.run(debug=True)
