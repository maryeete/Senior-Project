from flask import Blueprint, Flask, json, render_template, request, jsonify, Response
from flask_login import current_user, login_required
import mysql.connector
from secret import db_host, db_user, db_password, db_database  # Importing MySQL DB credentials from secret.py
from werkzeug.utils import secure_filename
import base64

dashboard = Blueprint(
    'dashboard',  # Blueprint name
    __name__,
    static_folder='static',
    template_folder='templates'
)

def create_db_connection():
    """
    Create a database connection using MySQL credentials from secret.py.

    Returns:
        connection: MySQL connection object.
    """
    return mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_database,
    )
    

import json

@dashboard.route('/analytics')
def analytics():
    # Connect to the database
    conn = create_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Query all emotion data
    cursor.execute("SELECT emotion_data, file_type FROM user_emotions")
    emotions = cursor.fetchall()

    sentiment_data = {}
    emotion_data = {
        'dominant_emotion': {},
        'other_emotions': {}
    }

    # Process emotion data
    for record in emotions:
        emotion_data_json = record['emotion_data']
        file_type = record['file_type']

        # Ensure that emotion_data is deserialized if it's a string
        if isinstance(emotion_data_json, str):
            try:
                emotion_data_json = json.loads(emotion_data_json)
            except json.JSONDecodeError:
                continue  # If the data is not valid JSON, skip this record

        if file_type == 'text':
            sentiment = emotion_data_json.get('sentiment')
            sentiment_data[sentiment] = sentiment_data.get(sentiment, 0) + 1
        else:
            # Initialize default values for dominant emotion and other emotions
            dominant_emotion = {}
            other_emotions = []

            if isinstance(emotion_data_json, dict):
                dominant_emotion = emotion_data_json.get('dominant_emotion', {})
                other_emotions = emotion_data_json.get('other_emotions', [])
            elif isinstance(emotion_data_json, list) and len(emotion_data_json) > 0:
                dominant_emotion = emotion_data_json[0].get('dominant_emotion', {})
                other_emotions = emotion_data_json[0].get('other_emotions', [])

            # Check if the dominant_emotion dictionary is not empty and update accordingly
            if dominant_emotion:
                dominant_emotion_name = dominant_emotion.get('name')
                if dominant_emotion_name:
                    emotion_data['dominant_emotion'][dominant_emotion_name] = emotion_data['dominant_emotion'].get(dominant_emotion_name, 0) + 1

            # Update other emotions if available
            for emotion in other_emotions:
                emotion_name = emotion.get('name')
                if emotion_name:
                    emotion_data['other_emotions'][emotion_name] = emotion_data['other_emotions'].get(emotion_name, 0) + 1

    # Close the connection
    cursor.close()
    conn.close()

    return render_template('analytics.html', sentiment_data=sentiment_data, emotion_data=emotion_data)




@dashboard.route('/dashboard', methods=['GET', 'POST'])
def dashboard_route():
    user_id = current_user.id  # Assuming you have a logged-in user

    # Filter by file type if a specific type is selected
    file_type = request.form.get('file_type', 'all')

    # Fetch data from the database
    conn = create_db_connection()
    cursor = conn.cursor(dictionary=True)

    if file_type == 'all':
        cursor.execute("SELECT * FROM user_emotions WHERE user_id = %s", (user_id,))
    else:
        cursor.execute("SELECT * FROM user_emotions WHERE user_id = %s AND file_type = %s", 
                       (user_id, file_type))

    files = cursor.fetchall()
    conn.close()

    # Process the blob data and encode it to base64
    for file in files:
        file_data = file['file']
        file_extension = file['file_type']

        # Encode binary data to base64
        encoded_file_data = base64.b64encode(file_data).decode('utf-8')

        if file_extension == 'text':
            file['file_data'] = f'{file_data.decode()}'
        else:
            file['file_data'] = f"data:{file_extension};base64,{encoded_file_data}" # Set file type for image, video, or audio

        # Access the emotion_data field (assuming it's stored as JSON)
        emotion_data = file.get('emotion_data', '[]')  # Default to an empty list if not available
        
        # If it's a JSON string, load it into a list
        if isinstance(emotion_data, str):
            file['emotions'] = json.loads(emotion_data)
        else:
            file['emotions'] = emotion_data

        # Normalize the emotion data to ensure consistency
        # If 'emotions' is a list containing one dictionary, we treat it as such
        if isinstance(file['emotions'], list) and len(file['emotions']) == 1:
            # If the list contains a single dictionary, treat it as such
            file['dominant_emotion'] = file['emotions'][0].get('dominant_emotion', {})
            file['other_emotions'] = file['emotions'][0].get('other_emotions', [])
            file['sentiment'] = file['emotions'][0].get('sentiment', None)
        elif isinstance(file['emotions'], list) and len(file['emotions']) > 1:
            # If the list contains multiple emotions, extract them
            file['dominant_emotion'] = file['emotions'][0].get('dominant_emotion', {})
            file['other_emotions'] = file['emotions'][0].get('other_emotions', [])
            file['sentiment'] = file['emotions'][0].get('sentiment', None)
        elif isinstance(file['emotions'], dict):
            # If emotions are a dictionary, treat it as a list with one item
            file['dominant_emotion'] = file['emotions'].get('dominant_emotion', {})
            file['other_emotions'] = file['emotions'].get('other_emotions', [])
            file['sentiment'] = file['emotions'].get('sentiment', None)
        else:
            # Handle case where no emotions are available
            file['dominant_emotion'] = {}
            file['other_emotions'] = []
            file['sentiment'] = None

    return render_template('dashboard.html', files=files, file_type=file_type)
