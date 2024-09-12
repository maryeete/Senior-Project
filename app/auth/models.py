"""
User model for authentication
"""

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
DB_NAME = "users.db"


class User(db.Model, UserMixin):
    """
    User model for the database.

    Attributes:
        id (int): The unique identifier for the user (primary key).
        email (str): The user's email address (primary key, unique).
        password (str): The hashed password for the user.
        name (str): The name of the user.
    """
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(32), nullable=False)
    name = db.Column(db.String(100), nullable=False)