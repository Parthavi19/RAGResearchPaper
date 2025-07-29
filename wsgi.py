from app import app  # app.py must contain app = Flask(__name__)
application = app    # This is what gunicorn expects
