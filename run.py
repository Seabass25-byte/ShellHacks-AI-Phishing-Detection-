from app import create_app  # Assuming the app package contains __init__.py

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
