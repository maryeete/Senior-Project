from flask import Flask
from flask_login import LoginManager, UserMixin
import mysql.connector
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from secret import db_host, db_user, db_password, db_database

class User(UserMixin):
    def __init__(self, id, full_name, email, username):
        self.id = id
        self.full_name = full_name
        self.email = email
        self.username = username

    @staticmethod
    def get_user(id):
        connection = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_database
        )
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM User_data WHERE id = %s", (id,)) # Using parameterized query to prevent SQL injection
        user_data = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if user_data:
            return User(user_data['id'], user_data['full_name'], user_data['email'], user_data['username'])
        return None

def create_app():
    """
    Create and configure the Flask application.

    Returns:
        Flask: The configured Flask application.
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test'
    app.config['UPLOAD_FOLDER'] = 'uploads/'

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.get_user(id)

    from app.auth.routes import auth
    from app.video.video import video
    from app.dashboard.dashboard import dashboard
    app.register_blueprint(auth)
    app.register_blueprint(video)
    app.register_blueprint(dashboard)
    
    return app
