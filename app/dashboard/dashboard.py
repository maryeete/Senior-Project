import os
from flask import Blueprint, Flask, json, render_template, request, jsonify, Response
from flask_login import current_user, login_required
import mysql.connector
from secret import db_host, db_user, db_password, db_database  # Importing MySQL DB credentials from secret.py
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

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
    
@dashboard.route('/analytics', methods=['GET', 'POST'])
def analytics():
    file_type = request.form.get('file_type', 'all')  # Default to 'all' (overall sort)
    conn = create_db_connection()
    cursor = conn.cursor(dictionary=True)

    if file_type == 'all':
        cursor.execute("SELECT emotion_data, file_type FROM User_Emotions WHERE user_id = %s", (current_user.id,))
    else:
        cursor.execute("SELECT emotion_data, file_type FROM User_Emotions WHERE user_id = %s AND file_type = %s", 
                       (current_user.id, file_type))

    emotions_data = cursor.fetchall()

    emotions = []
    for data in emotions_data:
        try:
            emotion_json = json.loads(data['emotion_data'])
            
            # Normalize the emotion data for both formats
            if isinstance(emotion_json, list):  # When data is in list format
                for emotion in emotion_json:
                    if isinstance(emotion, dict):
                        # Add dominant emotion
                        emotions.append(emotion['dominant_emotion']['name'])
                        
                        # Add other emotions
                        for other_emotion in emotion.get('other_emotions', []):
                            emotions.append(other_emotion['name'])
            elif isinstance(emotion_json, dict):  # When data is in dictionary format
                # Add dominant emotion
                emotions.append(emotion_json['dominant_emotion']['name'])
                
                # Add other emotions
                for other_emotion in emotion_json.get('other_emotions', []):
                    emotions.append(other_emotion['name'])
        
        except Exception as e:
            print(f"Error processing emotion data: {e}")

    # Process the data for charts
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    chart_data = {
        "labels": list(emotion_counts.keys()),
        "data": list(emotion_counts.values())
    }

    return render_template('analytics.html', chart_data=chart_data, selected_file_type=file_type)

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

        # Set file type for image, video, or audio
        file['file_data'] = f"data:{file_extension};base64,{encoded_file_data}"

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
