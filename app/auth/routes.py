"""
Authentication Blueprint for Flask Application

This module handles user authentication, including sign-up, sign-in, and session management.
It includes routes for logging in, signing up, logging out, and accessing protected content.
The module also contains helper functions for validating user input and ensuring secure password storage.

Blueprints:
    - auth: Handles all routes and functions related to user authentication.

Routes:
    - /: Displays a "Hello World" message on the home page.
    - /signup: Handles sign-up form submissions and renders the sign-up page.
    - /login: Handles login form submissions and renders the login page.
    - /protected: Accessible only to logged-in users, renders a protected page.
    - /logout: Clears session data and logs out the user.

Functions:
    - login_is_required: A decorator to restrict access to routes unless the user is logged in.
    - handle_sign_up: Processes the sign-up form, validates input, and stores the user in the database.
    - handle_sign_in: Processes the login form, validates credentials, and initiates the user session.

Usage:
    This blueprint can be registered in the main Flask application to enable user authentication
    functionality. The routes rely on session management to keep track of the logged-in state.
"""

from flask import Blueprint, redirect, render_template, request, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, current_user
import re
import mysql.connector
from secret import db_host, db_user, db_password, db_database  # Importing MySQL DB credentials from secret.py
from app import User

auth = Blueprint('auth', __name__)

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


def is_valid_email(email):
    """
    Validates the email format using regex.

    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def handle_sign_up(form):
    """
    Processes the sign-up form submission.

    Args:
        form (dict): The form data containing 'name', 'email', and 'password'.

    Returns:
        None: Flash messages are used to indicate success or error.
    """
    name = form.get('name')
    email = form.get('email')
    password = form.get('password')

    # Check if email already exists in the User_data table
    connection = create_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM User_data WHERE email = %s", (email,))
    existing_user = cursor.fetchone()

    if existing_user:
        flash('Email already exists. Please log in.', category='error')
    elif not is_valid_email(email):
        flash('Invalid email format.', category='error')
    elif len(password) < 6:
        flash('Password must be at least 6 characters.', category='error')
    elif len(password) > 32:
        flash('Password cannot exceed 32 characters.', category='error')
    else:
        hashed_password = generate_password_hash(password, method='scrypt')
        try:
            # Insert new user into the User_data table
            cursor.execute("INSERT INTO User_data (full_name, email, username, password) VALUES (%s, %s, %s, %s)",
                           (name, email, email.split('@')[0], hashed_password))  # Using email prefix as username
            connection.commit()  # Save the new user to the database
            flash('Account created!', category='success')
        except Exception as e:
            flash(f'Error creating account: {str(e)}', category='error')
        finally:
            cursor.close()
            connection.close()


def handle_sign_in(form):
    """
    Processes the login form submission.

    Args:
        form (dict): The form data containing 'email' and 'password'.

    Returns:
        Response: Redirects to the protected page if successful, otherwise returns None.
    """
    email = form.get('email')
    password = form.get('password')

    # Fetch user by email from User_data table
    connection = create_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM User_data WHERE email = %s", (email,))
    user_data = cursor.fetchone()
    
    try:
        if user_data and check_password_hash(user_data['password'], password):
            # Create a User object and log in the user
            user = User(user_data['user_id'], user_data['full_name'], user_data['email'], user_data['username'])
            login_user(user)  # Use Flask-Login's login_user
            return redirect(url_for("auth.protected"))  # Redirect to protected page upon successful login
        elif user_data:
            flash('Incorrect password.', category='error')
        else:
            flash('Email does not exist.', category='error')
    except Exception:
        flash('An error occurred during login.', category='error')
    finally:
        cursor.close()
        connection.close()
    
    return redirect(url_for('auth.login'))


@auth.route("/")
def index():
    """
    Renders a test index page that displays 'Hello World'.
    
    Returns:
        Response: Renders the 'index.html' template.
    """
    return render_template('index.html')


@auth.route("/signup", methods=['GET', 'POST'])
def sign_up():
    """
    Handles the sign-up form submission.

    POST: Processes the sign-up form and creates a new user if valid.
    GET: Renders the sign-up page.
    """
    if request.method == 'POST':
        handle_sign_up(request.form)
        return redirect(url_for('auth.login'))
    
    return render_template('signup.html')


@auth.route("/login", methods=['GET', 'POST'])
def login():
    """
    Handles the login form submission.

    POST: Processes the login form and starts a session if credentials are correct.
    GET: Renders the login page.
    """
    if request.method == 'POST':
        return handle_sign_in(request.form)
    
    return render_template('login.html')


@auth.route("/protected")
@login_required
def protected():
    """
    Renders the protected page, accessible only to logged-in users.
    
    Returns:
        Response: Renders the 'protected.html' template with the user's name and reviews.
    """
    user_id = current_user.id  # Assuming you have an `id` attribute in your User model
    
    # Connect to the MySQL database
    connection = create_db_connection()
    cursor = connection.cursor(dictionary=True)  # Use dictionary=True for results as dictionaries

    # Query to get reviews for the logged-in user
    cursor.execute("SELECT review_id, rating, review_text, sentiment_id FROM Reviews WHERE user_id = %s", (user_id,))
    user_reviews = cursor.fetchall()  # Fetch all reviews for the user
    
    cursor.close()
    connection.close()

    return render_template('protected.html', name=current_user.full_name, reviews=user_reviews)


@auth.route("/logout")
def logout():
    """
    Logs out the user by clearing the session data and redirects to the login page.
    """
    logout_user()  # Use Flask-Login's logout_user
    return redirect(url_for("auth.login"))