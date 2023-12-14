from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = "slideit"
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"*": {"origins": "*"}})

# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.sqlite3'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

from app import routes