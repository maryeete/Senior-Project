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
from app.auth.models import db, User
import re


auth = Blueprint('auth', __name__)


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

    user = User.query.filter_by(email=email).first()  # Check if the email already exists
    if user:
        flash('Email already exists. Please log in.', category='error')
    elif not is_valid_email(email):
        flash('Invalid email format.', category='error')
    elif len(password) < 6:
        flash('Password must be at least 6 characters.', category='error')
    elif len(password) > 32:
        flash('Password cannot exceed 32 characters.', category='error')
    else:
        new_user = User(name=name, email=email, password=generate_password_hash(password, method='scrypt'))
        try:
            db.session.add(new_user)
            db.session.commit()  # Save the new user to the database
            flash('Account created!', category='success')
        except Exception as e:
            flash(f'Error creating account: {str(e)}', category='error')


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

    user = User.query.filter_by(email=email).first()
    try:
        if user and check_password_hash(user.password, password):
            login_user(user)  # Use Flask-Login's login_user
            return redirect(url_for("auth.protected"))  # Redirect to protected page upon successful login
        elif user:
            flash('Incorrect password.', category='error')
        else:
            flash('Email does not exist.', category='error')
    except:
        flash('An error occurred during login.', category='error')
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
        success = handle_sign_up(request.form)
        if success:
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
        Response: Renders the 'protected.html' template with the user's name.
    """
    return render_template('protected.html', name=current_user.name)


@auth.route("/logout")
def logout():
    """
    Logs out the user by clearing the session data and redirects to the login page.
    """
    logout_user()  # Use Flask-Login's logout_user
    return redirect(url_for("auth.login"))