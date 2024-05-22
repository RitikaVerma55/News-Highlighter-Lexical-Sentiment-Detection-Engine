from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mysqldb import MySQL
from flask import jsonify
from reddit_news1 import scrape_reddit
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import XLMRobertaModel
import numpy as np
from summarizer import summarize_text


app = Flask(__name__)

class XLMRobertaClass(torch.nn.Module):
    def __init__(self):
        super(XLMRobertaClass, self).__init__()
        self.l1 = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    

app.secret_key = 'your_secret_key'  # Change this to a random value

# Your database connection and other setup code
# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Rose@2015'
app.config['MYSQL_DB'] = 'User_Database'

mysql = MySQL(app)

# Routes to render HTML templates
@app.route('/')
def index():
    return render_template('index_news.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    # Sign-up logic goes here
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = generate_password_hash(password)  # Hash the password
        
        # Check if username already exists
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            return render_template('signup.html', message='Username already exists. Please choose another one.')
        else:
            # Insert new user into the database
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, password_hash))
            mysql.connection.commit()
            cursor.close()
            
            # Set a flag indicating successful sign-up
            session['signup_success'] = True
            
            # Return JSON response indicating success
            return jsonify({'success': True})

    # Check if sign-up success flag is set
    signup_success = session.pop('signup_success', False)
    return render_template('signup.html', signup_success=signup_success)

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Login logic goes here
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Query the database for the provided username
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        
        # If user exists and password matches
        if user and check_password_hash(user[2], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', message='Invalid username or password')
    
    return render_template('login.html')

@app.route('/home')
def home():
    # Home page logic goes here
    if 'username' in session:
        # If logged in, render the home page template
        return render_template('home.html', username=session['username'], search = True)
    else:
        # If not logged in, redirect to the login page
        return redirect(url_for('login'))
    

@app.route('/search', methods=['POST'])  # Updated to handle only POST requests
def search():
    if request.method == 'POST':
        # Redirect the search request to the root URL of the second Flask application
        return redirect('http://localhost:5001/', code=307)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/index_data_scraper')
def index_data_scraper():
    return render_template('index_data_scraper.html')

@app.route('/scrape_reddit', methods=['POST'])
def scrape_reddit_route():
    # Get the URL from the form submission
    url = request.form.get('url')

    result= scrape_reddit(url)

    # Return the result to the client
    return result


from model_integration import summarize
from model_integration import emotion_counts
from model_integration import get_title, get_userId, get_upvote,get_createdAt, get_full_article

@app.route('/dashboard')
def show_dashboard():

    summary = summarize()
    count_array = emotion_counts()
    positive_count = count_array['Positive']
    negative_count = count_array['Negative']
    neutral_count = count_array['Neutral']
    

    # Assume you have sentiment counts stored in variables
    article_topic = get_title()
    writer_name = get_userId()
    upvotes_count = get_upvote()
    created_At = get_createdAt()
    full_article = get_full_article()

    
    # Pass article details and sentiment counts to the template
    return render_template('dashboard.html', article_topic=article_topic, writer_name=writer_name,
                           upvotes_count=upvotes_count, created_At=created_At,
                           positive_count=positive_count, negative_count=negative_count,
                           neutral_count=neutral_count, summary=summary, full_article =full_article )
    

if __name__ == '__main__':
    app.run(debug=True)
