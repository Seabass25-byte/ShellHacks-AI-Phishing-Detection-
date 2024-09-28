from flask import Blueprint

# Create a blueprint
main = Blueprint('main', __name__)

@main.route('/')
def index():
    return "Hello World!"
