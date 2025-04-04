from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path 
from flask_login import LoginManager

db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'Forex'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    # Import blueprints
    from .views import views
    from .auth import auth
    from .prediction import prediction  # Import the prediction blueprint

    # Register blueprints
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(prediction, url_prefix='/')  # Register the prediction blueprint

    from .models import User

    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.register'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
    return app

def create_database(app):
    if not path.exists('Flask/' + DB_NAME):
        with app.app_context():
            db.create_all()
            print('Created Database')
