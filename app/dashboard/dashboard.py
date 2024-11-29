from flask import Blueprint, Flask, json, render_template, request, jsonify, Response
from flask_login import current_user, login_required
import mysql.connector
from secret import db_host, db_user, db_password, db_database  # Importing MySQL DB credentials from secret.py

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
        database=db_database
    )
    
@dashboard.route('/dashboard', methods=['GET', 'POST'])
def dashboard_route():
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
            if isinstance(emotion_json, list):
                for emotion in emotion_json:
                    if isinstance(emotion, dict):
                        for other_emotion in emotion.get('other_emotions', []):
                            emotions.append(other_emotion)
        except Exception as e:
            print(f"Error processing emotion data: {e}")

    conn.close()

    emotion_count = {}
    for emotion in emotions:
        name = emotion['name']
        confidence = emotion['confidence']
        if name in emotion_count:
            emotion_count[name] += confidence
        else:
            emotion_count[name] = confidence

    labels = list(emotion_count.keys())
    values = list(emotion_count.values())

    return render_template('dashboard.html', labels=labels, values=values, file_type=file_type)