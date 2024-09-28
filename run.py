from flask import Flask, render_template
from app.routes import api_bp  # Adjust based on your project structure

def create_app():
    app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

    # Register the blueprint for the API (Make sure only '/api' is added once)
    app.register_blueprint(api_bp, url_prefix='/api')

    @app.route('/')
    def index():
        return render_template('index.html')  # Serve the HTML template

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
