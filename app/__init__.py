"""
User model for authentication
"""

from flask import Flask
from flask_login import LoginManager
from app.auth.models import db, DB_NAME

def create_app():
    """
    Create and configure the Flask application.

    Returns:
        Flask: The configured Flask application.
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    # Import and register blueprints (routes)
    from app.auth.routes import auth
    app.register_blueprint(auth)

    # Import models and set up user loader
    with app.app_context():
        from app.auth.models import User  # Import here to avoid circular import
        @login_manager.user_loader
        def load_user(user_id):
            return User.query.get(int(user_id))

        try:
            db.create_all()  # Create database tables
        except Exception as e:
            print(f"Error creating database tables: {e}")
    
    return app
