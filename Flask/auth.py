from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   ##means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
from email_validator import validate_email, EmailNotValidError

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('views.home'))  # Redirects to the home page
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("login.html", user=current_user)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!')
    return redirect(url_for('auth.login'))

@auth.route('/signup', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('views.home'))  # Redirect authenticated users to home page

    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('first_name')
        password1 = request.form.get('password1')
        # password2 = request.form.get('password2')

        if not email or not first_name or not password1:
            flash('Please fill out all fields.', category='error')
        else:
            try:
                # Validate email format
                valid = validate_email(email)
                email = valid.email
            except EmailNotValidError as e:
                flash(str(e), category='error')
                return render_template("register.html", user=current_user)


            user = User.query.filter_by(email=email).first()
            if user:
                flash('Email already exists', category='error')
            elif User.query.filter_by(first_name=first_name).first():
                flash('The username has been taken. Please input a different username.', category='error')

            elif len(first_name) < 2:
                flash('First name must be greater than 1 character.', category='error')
            elif not first_name[0].isalpha():
                flash('Username should start with an alphabet.', category='error')
            # elif password1 != password2:
            #     flash('Passwords do not match.', category='error')
            elif len(password1) < 8:
                flash('Password must be at least 8 characters.', category='error')
            elif not any(char.islower() for char in password1):
                flash('Password must contain at least one lowercase letter.', category='error')
            elif not any(char.isupper() for char in password1):
                flash('Password must contain at least one uppercase letter.', category='error')
            elif not any(char.isdigit() for char in password1):
                flash('Password must contain at least one digit.', category='error')
            elif not any(char in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/" for char in password1):
                flash('Password must contain at least one special character.', category='error')
            else:
                new_user = User(email=email, first_name=first_name,
                                password=generate_password_hash(password1, method='pbkdf2:sha256'))
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user, remember=True)
                flash('Account created!', category='success')
                return redirect(url_for('views.home'))

    return render_template("register.html", user=current_user)