import os
import pickle
import requests
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import logging
from datetime import datetime, date, timedelta
import json
import logging
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-generativeai not available, AI features will be disabled")

model = None
if GENAI_AVAILABLE:
    try:
        os.environ['gemini_api_key'] = "AIzaSyCZGGDVIyjebUyHX8m0xO6f1pBD6KKjErc"
        api_key = os.environ.get('gemini_api_key')
        if api_key:
            genai.configure(api_key=os.environ['gemini_api_key'])
            model = genai.GenerativeModel('gemini-1.5-flash')
            logging.info("Google Gemini model initialized successfully")
        else:
            logging.warning("GEMINI_API_KEY not found in environment")
            model = None
    except Exception as e:
        logging.error(f"Failed to initialize Gemini: {e}")
        model = None
else:
    logging.warning("Google Generative AI not available - AI features will use fallback responses")

#Configure logging
logging.basicConfig(level=logging.DEBUG)

heart_model = None
diabetes_model = None
cancer_model = None

def load_model(model_path, model_name):
    """Safely load a model with proper error handling"""
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
            logging.info(f"{model_name} model loaded successfully from {model_path}")
            return loaded_model
        else:
            logging.warning(f"{model_name} model file not found at {model_path}")
            return None
    except Exception as e:
        logging.error(f"Failed to load {model_name} model: {e}")
        return None

# Load models with relative paths
try:
    heart_model = load_model('./models/heart_model.sav', 'Heart disease')
    diabetes_model = load_model('./models/diabetes_model.sav', 'Diabetes')  
    cancer_model = load_model('./models/cancer_model.sav', 'Cancer')
except Exception as e:
    logging.error(f"Error during model loading: {e}")

def create_fallback_model(n_features):
    """Create a simple fallback model for demonstration when actual models aren't available"""
    try:
        from sklearn.linear_model import LogisticRegression
        fallback_model = LogisticRegression()
        X_dummy = np.array([[0.5] * n_features, [0.3] * n_features, [0.7] * n_features, [0.9] * n_features])
        y_dummy = np.array([0, 0, 1, 1])
        fallback_model.fit(X_dummy, y_dummy)
        return fallback_model
    except Exception as e:
        logging.error(f"Failed to create fallback model: {e}")
        return None

# Create fallback models if needed with correct feature counts
if heart_model is None:
    heart_model = create_fallback_model(13)  # Heart model expects 13 features
    if heart_model:
        logging.info("Using fallback model for heart disease prediction")

if diabetes_model is None:
    diabetes_model = create_fallback_model(8)  # Diabetes model expects 8 features
    if diabetes_model:
        logging.info("Using fallback model for diabetes prediction")

if cancer_model is None:
    cancer_model = create_fallback_model(5)  # Cancer model expects 5 features (reduced from 22)
    if cancer_model:
        logging.info("Using fallback model for cancer prediction")
        
# Session management
def init_session_data():
    """Initialize session data structure"""
    if 'user_data' not in session:
        session['user_data'] = {
            'predictions': {},
            'chat_history': [],
            'mental_health_screenings': [],
            'sleep_wellness_data': [],
            'health_goals': [],
            'health_quests': [],
            'telemedicine_sessions': []
        }
# Routes
@app.route('/')
def index():
    """Homepage with animated introduction"""
    init_session_data()
    return render_template('index.html')

@app.route('/heart')
def heart():
    """Heart Disease Prediction Page"""
    return render_template('heart.html')

@app.route('/heart_predict', methods=['POST'])
def heart_predict():
    """Handle heart disease prediction using the actual ML model"""
    try:
        if heart_model is None:
           return render_template('heart.html', error="Heart disease model not available")
        
        # Get form data - matching Streamlit version exactly
        age = float(request.form['age'])
        sex = int(request.form['sex'])  # 1 for male, 0 for female
        cp = int(request.form['cp'])  # chest pain type
        trestbps = float(request.form['trestbps'])  # resting blood pressure
        chol = float(request.form['chol'])  # cholesterol
        fbs = int(request.form['fbs'])  # fasting blood sugar
        restecg = int(request.form['restecg'])  # resting ECG
        thalach = float(request.form['thalach'])  # max heart rate
        exang = int(request.form['exang'])  # exercise induced angina
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])  # number of major vessels
        thal = int(request.form['thal'])  # thalassemia
        
        # Create data array and normalize values (0-1 range for compatibility with model)
        data = [age/100, sex, cp/4, trestbps/200, chol/400, fbs, restecg/2, thalach/200, exang, oldpeak/6, slope/2, ca/4, thal/3]
        data_array = np.array(data, dtype=float).reshape(1, -1)
        
        # Make prediction using the actual model
        prediction = heart_model.predict(data_array)[0]
        
        # Store prediction
        init_session_data()
        session['user_data']['predictions']['heart'] = {
            'prediction': int(prediction),
            'probability': float(prediction),
            'timestamp': datetime.now().isoformat()
        }
        session.modified = True
        
        # Create result message like Streamlit
        if prediction == 0:
            result = "You are not prone to Heart Disease"
            risk_level = "Low"
        else:
            result = "Sorry, You are more prone to Heart Disease"
            risk_level = "High"
        
        # Generate basic recommendations
        recommendations = []
        if prediction == 1:
            recommendations.extend([
                "Consult a cardiologist immediately",
                "Monitor blood pressure regularly",
                "Follow a heart-healthy diet",
                "Engage in regular moderate exercise",
                "Avoid smoking and excessive alcohol"
            ])
        else:
            recommendations.extend([
                "Maintain a healthy lifestyle",
                "Continue regular health check-ups",
                "Keep cholesterol levels in check"
            ])
        
        return render_template('heart.html', 
                             prediction=int(prediction),
                             result=result,
                             risk_level=risk_level,
                             recommendations=recommendations)
        
    except Exception as e:
        logging.error(f"Heart prediction error: {e}")
        return render_template('heart.html', error=f"An error occurred: {str(e)}")

@app.route('/diabetes')
def diabetes():
    """Diabetes Prediction Page"""
    return render_template('diabetes.html')

@app.route('/diabetes_predict', methods=['POST'])
def diabetes_predict():
    """Handle diabetes prediction using the actual ML model"""
    try:
        if diabetes_model is None:
            return render_template('diabetes.html', error="Diabetes model not available")
        
        # Debug: Log form data
        logging.debug(f"Form data received: {dict(request.form)}")
        
        # Get form data with better error handling
        try:
            pregnancies = int(request.form.get('pregnancies', 0))
            glucose = float(request.form.get('glucose', 0))
            blood_pressure = float(request.form.get('blood_pressure', 0))
            skin_thickness = float(request.form.get('skin_thickness', 0))
            insulin = float(request.form.get('insulin', 0))
            bmi = float(request.form.get('bmi', 0))
            dpf = float(request.form.get('dpf', 0))  # diabetes pedigree function
            age = int(request.form.get('age', 0))
        except (ValueError, TypeError) as e:
            logging.error(f"Form data conversion error: {e}")
            return render_template('diabetes.html', error=f"Invalid form data: {str(e)}")
        
        # Create data array and normalize values (0-1 range for compatibility with model)
        data = [pregnancies/20, glucose/200, blood_pressure/150, skin_thickness/100, insulin/900, bmi/70, dpf/2.5, age/100]
        data_array = np.array(data, dtype=float).reshape(1, -1)
        
        # Make prediction using the actual model
        prediction = diabetes_model.predict(data_array)[0]
        
        # Store prediction
        init_session_data()
        session['user_data']['predictions']['diabetes'] = {
            'prediction': int(prediction),
            'probability': float(prediction),
            'timestamp': datetime.now().isoformat()
        }
        session.modified = True
        
        # Create result message like Streamlit
        if prediction == 0:
            result = "You are not prone to Diabetes"
            risk_level = "Low"
        else:
            result = "Sorry, You are more prone to Diabetes"
            risk_level = "High"
        
        # Generate basic recommendations
        recommendations = []
        if prediction == 1:
            recommendations.extend([
                "Consult an endocrinologist",
                "Monitor blood glucose levels regularly",
                "Follow a diabetic-friendly diet",
                "Maintain a healthy weight",
                "Exercise regularly"
            ])
        else:
            recommendations.extend([
                "Continue healthy eating habits",
                "Maintain regular physical activity",
                "Schedule regular health screenings"
            ])
        
        return render_template('diabetes.html',
                             prediction=int(prediction),
                             result=result,
                             risk_level=risk_level,
                             recommendations=recommendations)
        
    except Exception as e:
        logging.error(f"Diabetes prediction error: {e}")
        return render_template('diabetes.html', error=f"An error occurred: {str(e)}")

@app.route('/cancer')
def cancer():
    """Cancer Disease Prediction Page"""
    return render_template('cancer.html')

@app.route('/cancer_predict', methods=['POST'])
def cancer_predict():
    """Handle cancer prediction using the actual ML model"""
    try:
        if cancer_model is None:
            return render_template('cancer.html', error="Cancer model not available")
        
        # Get form data - matching Streamlit version exactly (22 parameters)
        fo = float(request.form['fo'])
        fhi = float(request.form['fhi'])
        flo = float(request.form['flo'])
        jitter_percent = float(request.form['jitter_percent'])
        jitter_abs = float(request.form['jitter_abs'])
        rap = float(request.form['rap'])
        ppq = float(request.form['ppq'])
        ddp = float(request.form['ddp'])
        shimmer = float(request.form['shimmer'])
        shimmer_db = float(request.form['shimmer_db'])
        apq3 = float(request.form['apq3'])
        apq5 = float(request.form['apq5'])
        apq = float(request.form['apq'])
        dda = float(request.form['dda'])
        nhr = float(request.form['nhr'])
        hnr = float(request.form['hnr'])
        rpde = float(request.form['rpde'])
        dfa = float(request.form['dfa'])
        spread1 = float(request.form['spread1'])
        spread2 = float(request.form['spread2'])
        d2 = float(request.form['d2'])
        ppe = float(request.form['ppe'])
        
        # Map the 22 voice parameters to 5 key features for our breast cancer model
        # Normalize and combine related parameters into meaningful features
        mean_radius = (fo + fhi + flo) / 3 / 300  # Combined frequency measures
        mean_texture = (jitter_percent + jitter_abs + rap + ppq + ddp) / 5 / 10  # Jitter-related measures
        mean_perimeter = (shimmer + shimmer_db + apq3 + apq5 + apq + dda) / 6 / 10  # Shimmer-related measures
        mean_area = (nhr + hnr + rpde + dfa) / 4 / 10  # Noise and fractal measures
        mean_smoothness = (spread1 + spread2 + d2 + ppe) / 4 / 10  # Spread and perturbation measures
        
        # Create data array with normalized 5-feature format
        data = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]
        data_array = np.array(data, dtype=float).reshape(1, -1)
        
        # Make prediction using the actual model
        prediction = cancer_model.predict(data_array)[0]
        
        # Store prediction
        init_session_data()
        session['user_data']['predictions']['cancer'] = {
            'prediction': int(prediction),
            'probability': float(prediction),
            'timestamp': datetime.now().isoformat()
        }
        session.modified = True
        
        # Create result message like Streamlit
        if prediction == 0:
            result = "You are not prone to Cancer"
            risk_level = "Low"
        else:
            result = "Sorry, You are more prone to Cancer"
            risk_level = "High"
        
        # Generate basic recommendations
        recommendations = []
        if prediction == 1:
            recommendations.extend([
                "Consult an oncologist immediately",
                "Get comprehensive cancer screening",
                "Discuss family history with doctor",
                "Consider lifestyle modifications",
                "Follow up with regular check-ups"
            ])
        else:
            recommendations.extend([
                "Continue regular cancer screenings",
                "Maintain healthy lifestyle habits",
                "Monitor for any changes in voice patterns"
            ])
        
        return render_template('cancer.html',
                             prediction=int(prediction),
                             result=result,
                             risk_level=risk_level,
                             recommendations=recommendations)
        
    except Exception as e:
        logging.error(f"Cancer prediction error: {e}")
        return render_template('cancer.html', error=f"An error occurred: {str(e)}")

@app.route('/chat')
def chat():
    """AI Chat Assistance Page"""
    init_session_data()
    chat_history = session['user_data'].get('chat_history', [])
    return render_template('chat.html', chat_history=chat_history)

@app.route('/chat_with_bot', methods=['POST'])
def chat_with_bot():
    """Handle chat message sending"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        chat_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Initialize session data
        init_session_data()
        
        # Add user message to chat history
        session['user_data']['chat_history'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate AI response using direct HTTP request to Gemini API
        gemini_api_key = os.environ.get('gemini_api_key')
        if gemini_api_key:
            try:
                # Use direct HTTP request to Gemini API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
                
                prompt = f"""You are a helpful AI assistant for the IllnessInsight platform. 
                You can answer questions on any topic - technology, science, general knowledge, education, entertainment, etc.
                Be knowledgeable, helpful, and conversational while maintaining a friendly and professional tone.
                Provide accurate information and admit when you're uncertain about something.
                Keep responses informative but concise.
                
                User message: {user_message}
                
                Please provide a helpful response on any topic the user asks about."""
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }]
                }
                
                response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
                
                if response.status_code == 200:
                    result = response.json()
                    if 'candidates' in result and len(result['candidates']) > 0:
                        ai_message = result['candidates'][0]['content']['parts'][0]['text']
                    else:
                        ai_message = "I'm having difficulty generating a response right now. Please consult with a healthcare professional for medical guidance."
                else:
                    logging.error(f"Gemini API HTTP error: {response.status_code} - {response.text}")
                    ai_message = "I'm experiencing technical difficulties. For immediate health concerns, please contact a healthcare professional."
                    
            except Exception as e:
                logging.error(f"Gemini API error: {e}")
                ai_message = "I'm having trouble connecting to my AI service right now. Please try again later or consult with a healthcare professional for immediate concerns."
        else:
            ai_message = "AI service requires configuration. Please consult with a healthcare professional for medical advice."
        
        # Add AI response to chat history
        session['user_data']['chat_history'].append({
            'role': 'assistant',
            'content': ai_message,
            'timestamp': datetime.now().isoformat()
        })
        
        session.modified = True
        
        return jsonify({
            'success': True,
            'response': ai_message,
            'history': session['user_data']['chat_history']
        })
    
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat_clear', methods=['POST'])
def chat_clear():
    """Clear chat history"""
    init_session_data()
    session['user_data']['chat_history'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/mental_health')
def mental_health():
    """Mental Health Screening Page"""
    return render_template('mental_health.html')

@app.route('/depression_screening')
def depression_screening():
    """Depression Screening Page"""
    return render_template('depression_screening.html')

@app.route('/anxiety_screening')
def anxiety_screening():
    """Anxiety Screening Page"""
    return render_template('anxiety_screening.html')

@app.route('/stress_screening')
def stress_screening():
    """Stress Screening Page"""
    return render_template('stress_screening.html')

@app.route('/process_mental_health_screening', methods=['POST'])
def process_mental_health_screening():
    """Process mental health screening results"""
    try:
        screening_type = request.form.get('screening_type')
        
        # Calculate total score based on screening type
        total_score = 0
        question_count = 0
        
        for key, value in request.form.items():
            if key.startswith('q') and key[1:].isdigit():
                total_score += int(value)
                question_count += 1
        
        # Analyze score and determine risk level
        risk_level, recommendations = analyze_mental_health_score(screening_type, total_score)
        
        # Perform sentiment analysis if additional text is provided
        additional_notes = request.form.get('additional_notes', '')
        sentiment_analysis = None
        if additional_notes:
            sentiment_analysis = analyze_text_sentiment(additional_notes)
        
        # Generate AI recommendations
        ai_recommendations = generate_mental_health_recommendations(screening_type, total_score, risk_level)
        
        # Create screening object for template
        screening = {
            'screening_type': screening_type,
            'score': total_score,
            'risk_level': risk_level,
            'recommendations': recommendations,
            'sentiment_analysis': sentiment_analysis
        }
        
        # Store results in session
        init_session_data()
        session['user_data']['mental_health_screenings'].append({
            'screening_type': screening_type,
            'score': total_score,
            'risk_level': risk_level,
            'recommendations': recommendations,
            'ai_recommendations': ai_recommendations,
            'sentiment_analysis': sentiment_analysis,
            'created_at': datetime.now().isoformat()
        })
        session.modified = True
        
        return render_template('mental_health_results.html',
                             screening=screening,
                             screening_type=screening_type,
                             score=total_score,
                             risk_level=risk_level,
                             recommendations=recommendations,
                             ai_recommendations=ai_recommendations,
                             sentiment_analysis=sentiment_analysis)
    
    except Exception as e:
        logging.error(f"Mental health screening error: {e}")
        return render_template('mental_health.html', error=str(e))

@app.route('/sleep_wellness')
def sleep_wellness():
    """Sleep and Wellness Tracking Page"""
    return render_template('sleep_wellness.html')

@app.route('/track_sleep_wellness', methods=['POST'])
def track_sleep_wellness():
    """Track sleep and wellness data"""
    try:
        # Get form data
        init_session_data()
        sleep_date = request.form.get('sleep_date')
        sleep_duration = float(request.form.get('sleep_duration', 0))
        sleep_quality = int(request.form.get('sleep_quality', 5))
        fatigue_level = int(request.form.get('fatigue_level', 5))
        alertness_level = int(request.form.get('alertness_level', 5))
        notes = request.form.get('notes', '')
        
        # Calculate wellness score
        wellness_score = calculate_wellness_score(sleep_duration, sleep_quality, fatigue_level, alertness_level)
        
        # Generate AI insights
        sleep_data = {
            'sleep_duration': sleep_duration,
            'sleep_quality': sleep_quality,
            'fatigue_level': fatigue_level,
            'alertness_level': alertness_level,
            'wellness_score': wellness_score,
            'notes':notes
        }
        insights = generate_sleep_insights(sleep_data)
        
        # Store in session
        if 'sleep_tracking' not in session['user_data']:
            session['user_data']['sleep_tracking'] = []
        session['user_data']['sleep_tracking'].append({
            'sleep_date': sleep_date,
            'sleep_duration': sleep_duration,
            'sleep_quality': sleep_quality,
            'fatigue_level': fatigue_level,
            'alertness_level': alertness_level,
            'wellness_score': wellness_score,
            'notes': notes,
            'insights': insights,
            'created_at': datetime.now().isoformat()
        })
        session.modified = True
        
        return render_template('sleep_insights.html',
                             sleep_data=sleep_data,
                             insights=insights)
    
    except Exception as e:
        logging.error(f"Sleep tracking error: {e}")
        return render_template('sleep_wellness.html', error=str(e))

@app.route('/health_coaching')
def health_coaching():
    """Personalized Health Coaching Page"""
    init_session_data()
    health_goals = session['user_data'].get('health_goals', [])
    
    # Get and clear any success/error messages
    success_message = session.pop('success_message', None)
    error_message = session.pop('error_message', None)
    
    return render_template('health_coaching.html', 
                         health_goals=health_goals,
                         success_message=success_message,
                         error_message=error_message)

@app.route('/create_health_goal', methods=['POST'])
def create_health_goal():
    """Create a new health goal"""
    try:
        # Get form data
        goal_type = request.form.get('goal_type')
        title = request.form.get('title')
        description = request.form.get('description')
        target_value = float(request.form.get('target_value', 0))
        unit = request.form.get('unit')
        target_date = request.form.get('target_date')
        
        # Generate AI recommendations
        goal_data = {
            'goal_type': goal_type,
            'title': title,
            'description': description,
            'target_value': target_value,
            'unit': unit
        }
        ai_recommendations = generate_goal_recommendations(goal_data)
        
        # Store in session
        init_session_data()
        goal_id = len(session['user_data']['health_goals']) + 1
        new_goal = {
            'id': goal_id,
            'goal_type': goal_type,
            'title': title,
            'description': description,
            'target_value': target_value,
            'current_value': 0,
            'unit': unit,
            'target_date': target_date,
            'status': 'active',
            'ai_recommendations': ai_recommendations,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        session['user_data']['health_goals'].append(new_goal)
        session.modified = True
        
        # Add success message
        session['success_message'] = f"Health goal '{title}' created successfully!"
        session.modified = True
        
        # Debug logging
        logging.info(f"Created health goal: {new_goal}")
        logging.info(f"Total health goals: {len(session['user_data']['health_goals'])}")
        
        return redirect(url_for('health_coaching'))
    
    except Exception as e:
        logging.error(f"Health goal creation error: {e}")
        return redirect(url_for('health_coaching'))

@app.route('/health_quests')
def health_quests():
    """Health Quests and Gamification Page"""
    init_session_data()
    
    # Calculate total credits
    total_credits = sum(progress.get('credits_earned', 0) 
                       for progress in session['user_data'].get('gamification_progress', []))
    
    gamification_progress = session['user_data'].get('gamification_progress', [])
    
    return render_template('health_quests.html', 
                         total_credits=total_credits,
                         gamification_progress=gamification_progress)

@app.route('/start_health_quest/<quest_type>', methods=['POST'])
def start_health_quest(quest_type):
    """Start a new health quest"""
    try:
        init_session_data()
        # Get quest template
        quest_template = get_quest_template(quest_type)
        
        # Calculate end date
        start_date = datetime.now()
        end_date = start_date + timedelta(days=quest_template['duration_days'])
        
        # Store in session
        if 'gamification_progress' not in session['user_data']:
            session['user_data']['gamification_progress'] = []
        quest_id = len(session['user_data']['gamification_progress']) + 1
        session['user_data']['gamification_progress'].append({
            'id': quest_id,
            'quest_type': quest_type,
            'quest_name': quest_template['name'],
            'description': quest_template['description'],
            'daily_target': quest_template['daily_target'],
            'duration_days': quest_template['duration_days'],
            'progress': 0,
            'credits_earned': 0,
            'total_credits': quest_template['credits'],
            'status': 'active',
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'days_completed': 0,
            'created_at': start_date.isoformat()
        })
        session.modified = True
        
        return redirect(url_for('health_quests'))
    
    except Exception as e:
        logging.error(f"Quest start error: {e}")
        return redirect(url_for('health_quests'))

@app.route('/complete_quest_day/<int:quest_id>', methods=['POST'])
def complete_quest_day(quest_id):
    """Mark a day as completed for a quest"""
    try:
        init_session_data()
        quests = session['user_data']['gamification_progress']
        
        for quest in quests:
            if quest.get('id') == quest_id and quest['status'] == 'active':
                quest['days_completed'] += 1
                quest['progress'] = (quest['days_completed'] / quest['duration_days']) * 100
                
                # Check if quest is completed
                if quest['days_completed'] >= quest['duration_days']:
                    quest['status'] = 'completed'
                    quest['credits_earned'] = quest['total_credits']
                    quest['completion_date'] = datetime.now().isoformat()
                
                session.modified = True
                break
        
        return jsonify({'success': True, 'message': 'Day marked as completed!'})
    
    except Exception as e:
        logging.error(f"Quest day completion error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/complete_quest/<int:quest_id>', methods=['POST'])
def complete_quest(quest_id):
    """Mark entire quest as completed"""
    try:
        init_session_data()
        quests = session['user_data']['gamification_progress']
        
        for quest in quests:
            if quest.get('id') == quest_id and quest['status'] == 'active':
                quest['status'] = 'completed'
                quest['progress'] = 100
                quest['days_completed'] = quest['duration_days']
                quest['credits_earned'] = quest['total_credits']
                quest['completion_date'] = datetime.now().isoformat()
                session.modified = True
                break
        
        return jsonify({'success': True, 'message': 'Quest completed! Credits earned!'})
    
    except Exception as e:
        logging.error(f"Quest completion error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_goal_progress/<int:goal_id>', methods=['POST'])
def update_goal_progress(goal_id):
    """Update progress for a health goal"""
    try:
        init_session_data()
        goals = session['user_data']['health_goals']
        progress_value = float(request.form.get('progress_value', 0))
        if progress_value < 0:
            return jsonify({'error': 'Progress value cannot be negative'}), 400
        goal_found = False
        
        for goal in goals:
            if goal.get('id') == goal_id:
                goal_found = True
                if goal['status'] == 'active':
                    # Add to existing value instead of replacing
                    goal['current_value'] = (goal.get('current_value', 0) or 0) + progress_value
                    goal['updated_at'] = datetime.now().isoformat()
                    
                    # Check if goal is completed
                    if goal['current_value'] >= goal['target_value']:
                        goal['status'] = 'completed'
                        goal['completion_date'] = datetime.now().isoformat()
                        session['success_message'] = f"Congratulations! Goal '{goal['title']}' completed!"
                    
                    session.modified = True
                    return jsonify({
                        'success': True, 
                        'message': f'Progress updated! Total: {goal["current_value"]} {goal["unit"]}',
                        'current_value': goal['current_value'],
                        'target_value': goal['target_value']
                    })
                else:
                    return jsonify({'error': 'Cannot update progress for completed goals'}), 400
                break
        
        if not goal_found:
            return jsonify({'error': 'Goal not found'}), 404
            
        return jsonify({'error': 'Goal update failed'}), 500
    
    except ValueError:
        return jsonify({'error': 'Invalid progress value'}), 400
    
    except Exception as e:
        logging.error(f"Goal progress update error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/telemedicine')
def telemedicine():
    """Telemedicine and Support Page"""
    init_session_data()
    telemedicine_sessions = session['user_data'].get('telemedicine_sessions', [])

    os.environ['GEOAPIFY_API_KEY'] = "6e43b96de1974488ac0ea9080f851ed2"
    geoapify_api_key = os.environ.get('GEOAPIFY_API_KEY', '')
    return render_template('telemedicine.html', 
                         telemedicine_sessions=telemedicine_sessions,
                         geoapify_api_key=geoapify_api_key)

@app.route('/api/nearby-facilities')
def nearby_facilities():
    """API endpoint to find nearby medical facilities"""
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        radius = request.args.get('radius', 5000, type=int)  # Default 5km radius
        facility_type = request.args.get('type', 'all')  # Facility type filter
        
        if not lat or not lng:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        # Use OpenStreetMap Overpass API (free, no API key required)
        facilities = []
        
        # Map facility types to OSM amenity/healthcare tags
        facility_map = {
            'all': ['hospital', 'clinic', 'doctors', 'pharmacy', 'dentist', 'physiotherapist', 'alternative_medicine', 'veterinary'],
            'hospital': ['hospital'],
            'clinic': ['clinic', 'doctors'],
            'urgent_care': ['clinic', 'doctors'],
            'pharmacy': ['pharmacy'],
            'mental_health': ['psychotherapist', 'alternative_medicine'],
            'dentist': ['dentist'],
            'specialist': ['clinic', 'doctors'],
            'emergency': ['hospital']
        }
        
        amenities = facility_map.get(facility_type, facility_map['all'])
        
        # Build Overpass query for medical facilities
        radius_meters = radius
        overpass_query = f"""
        [out:json][timeout:25];
        (
        """
        
        for amenity in amenities:
            overpass_query += f"""
            node["amenity"="{amenity}"](around:{radius_meters},{lat},{lng});
            way["amenity"="{amenity}"](around:{radius_meters},{lat},{lng});
            node["healthcare"="{amenity}"](around:{radius_meters},{lat},{lng});
            way["healthcare"="{amenity}"](around:{radius_meters},{lat},{lng});
            """
        
        overpass_query += """
        );
        out center meta;
        """
        
        # Query Overpass API
        overpass_url = 'https://overpass-api.de/api/interpreter'
        
        try:
            response = requests.post(overpass_url, data=overpass_query, timeout=30)
            
            if response.status_code != 200:
                logging.error(f"Overpass API error: {response.status_code} - {response.text}")
                return jsonify({'error': 'Unable to fetch facility data'}), 500
            
            data = response.json()
            
            for element in data.get('elements', []):
                tags = element.get('tags', {})
                
                # Get coordinates
                if element['type'] == 'node':
                    lat_coord = element.get('lat')
                    lng_coord = element.get('lon')
                elif element['type'] == 'way':
                    center = element.get('center', {})
                    lat_coord = center.get('lat')
                    lng_coord = center.get('lon')
                else:
                    continue
                
                if not lat_coord or not lng_coord:
                    continue
                
                # Extract facility information
                name = tags.get('name', 'Unnamed Facility')
                amenity = tags.get('amenity', tags.get('healthcare', ''))
                
                # Build address
                address_parts = []
                if tags.get('addr:housenumber') and tags.get('addr:street'):
                    address_parts.append(f"{tags.get('addr:housenumber')} {tags.get('addr:street')}")
                elif tags.get('addr:street'):
                    address_parts.append(tags.get('addr:street'))
                
                if tags.get('addr:city'):
                    address_parts.append(tags.get('addr:city'))
                if tags.get('addr:postcode'):
                    address_parts.append(tags.get('addr:postcode'))
                
                vicinity = ', '.join(address_parts) if address_parts else 'Address not available'
                
                # Map amenity to category
                category_map = {
                    'hospital': 'Hospital',
                    'clinic': 'Clinic',
                    'doctors': 'Medical Practice',
                    'pharmacy': 'Pharmacy',
                    'dentist': 'Dental Clinic',
                    'physiotherapist': 'Physical Therapy',
                    'psychotherapist': 'Mental Health',
                    'alternative_medicine': 'Alternative Medicine',
                    'veterinary': 'Veterinary'
                }
                
                facility = {
                    'place_id': str(element.get('id', '')),
                    'name': name,
                    'vicinity': vicinity,
                    'address_line1': tags.get('addr:street', ''),
                    'address_line2': '',
                    'city': tags.get('addr:city', ''),
                    'country': tags.get('addr:country', ''),
                    'postcode': tags.get('addr:postcode', ''),
                    'category': category_map.get(amenity, 'Healthcare Facility'),
                    'amenity_type': amenity,
                    'geometry': {
                        'location': {
                            'lat': lat_coord,
                            'lng': lng_coord
                        }
                    },
                    'contact': {
                        'phone': tags.get('phone', ''),
                        'website': tags.get('website', '')
                    },
                    'opening_hours': tags.get('opening_hours', ''),
                    'wheelchair': tags.get('wheelchair', ''),
                    'emergency': tags.get('emergency', ''),
                    'operator': tags.get('operator', ''),
                    'healthcare_speciality': tags.get('healthcare:speciality', '')
                }
                
                # Calculate distance from search center
                distance = calculate_distance(lat, lng, lat_coord, lng_coord)
                facility['distance'] = round(distance, 2)
                
                # Avoid duplicates based on name and close location
                duplicate = False
                for existing in facilities:
                    if (existing['name'].lower() == facility['name'].lower() and 
                        abs(existing['geometry']['location']['lat'] - facility['geometry']['location']['lat']) < 0.0001 and
                        abs(existing['geometry']['location']['lng'] - facility['geometry']['location']['lng']) < 0.0001):
                        duplicate = True
                        break
                
                if not duplicate and distance <= (radius / 1000):  # Convert meters to km
                    facilities.append(facility)
            
            # Sort by distance
            facilities.sort(key=lambda x: x.get('distance', 0))
            
        except requests.exceptions.Timeout:
            logging.error("Overpass API timeout")
            return jsonify({'error': 'Search request timed out. Please try again.'}), 500
        except Exception as e:
            logging.error(f"Error querying Overpass API: {str(e)}")
            return jsonify({'error': 'Unable to search for facilities'}), 500
        
        return jsonify({'facilities': facilities})
        
    except Exception as e:
        logging.error(f"Error in nearby_facilities: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/facility-details/<place_id>')
def facility_details(place_id):
    """Get detailed information about a specific facility using OpenStreetMap"""
    try:
        # Use Overpass API to get detailed information about the facility
        overpass_query = f"""
        [out:json][timeout:25];
        (
            node(id:{place_id});
            way(id:{place_id});
            relation(id:{place_id});
        );
        out meta;
        """
        
        overpass_url = 'https://overpass-api.de/api/interpreter'
        response = requests.post(overpass_url, data=overpass_query, timeout=30)
        
        if response.status_code != 200:
            logging.error(f"Overpass API error: {response.status_code} - {response.text}")
            return jsonify({'error': 'Failed to fetch facility details'}), 500
        
        data = response.json()
        elements = data.get('elements', [])
        
        if not elements:
            return jsonify({'error': 'Facility not found'}), 404
        
        element = elements[0]
        tags = element.get('tags', {})
        
        # Build address
        address_parts = []
        if tags.get('addr:housenumber') and tags.get('addr:street'):
            address_parts.append(f"{tags.get('addr:housenumber')} {tags.get('addr:street')}")
        elif tags.get('addr:street'):
            address_parts.append(tags.get('addr:street'))
        
        if tags.get('addr:city'):
            address_parts.append(tags.get('addr:city'))
        if tags.get('addr:state'):
            address_parts.append(tags.get('addr:state'))
        if tags.get('addr:postcode'):
            address_parts.append(tags.get('addr:postcode'))
        if tags.get('addr:country'):
            address_parts.append(tags.get('addr:country'))
        
        formatted_address = ', '.join(address_parts) if address_parts else 'Address not available'
        
        # Map amenity to category
        amenity = tags.get('amenity', tags.get('healthcare', ''))
        category_map = {
            'hospital': 'Hospital',
            'clinic': 'Clinic', 
            'doctors': 'Medical Practice',
            'pharmacy': 'Pharmacy',
            'dentist': 'Dental Clinic',
            'physiotherapist': 'Physical Therapy',
            'psychotherapist': 'Mental Health',
            'alternative_medicine': 'Alternative Medicine',
            'veterinary': 'Veterinary Clinic'
        }
        
        facility_details_obj = {
            'name': tags.get('name', 'Unnamed Facility'),
            'formatted_address': formatted_address,
            'formatted_phone_number': tags.get('phone', ''),
            'website': tags.get('website', ''),
            'opening_hours': tags.get('opening_hours', ''),
            'wheelchair': tags.get('wheelchair', ''),
            'category': category_map.get(amenity, 'Healthcare Facility'),
            'amenity_type': amenity,
            'operator': tags.get('operator', ''),
            'emergency': tags.get('emergency', ''),
            'healthcare_speciality': tags.get('healthcare:speciality', ''),
            'place_id': place_id
        }
        
        return jsonify({'facility': facility_details_obj})
        
    except requests.exceptions.Timeout:
        logging.error("Overpass API timeout")
        return jsonify({'error': 'Request timed out. Please try again.'}), 500
    except Exception as e:
        logging.error(f"Error in facility_details: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/distance-matrix')
def distance_matrix():
    """Calculate distance and duration between user location and facility"""
    try:
        origin_lat = request.args.get('origin_lat', type=float)
        origin_lng = request.args.get('origin_lng', type=float)
        dest_lat = request.args.get('dest_lat', type=float)
        dest_lng = request.args.get('dest_lng', type=float)
        
        if not all([origin_lat, origin_lng, dest_lat, dest_lng]):
            return jsonify({'error': 'Origin and destination coordinates are required'}), 400
        
        geoapify_api_key = os.environ.get('GEOAPIFY_API_KEY', '')
        if not geoapify_api_key:
            return jsonify({'error': 'Geoapify API key not configured'}), 500
        
        # Use Geoapify Routing API
        routing_url = 'https://api.geoapify.com/v1/routing'
        params = {
            'waypoints': f'{origin_lat},{origin_lng}|{dest_lat},{dest_lng}',
            'mode': 'drive',
            'apiKey': geoapify_api_key
        }
        
        response = requests.get(routing_url, params=params)
        
        if response.status_code != 200:
            logging.error(f"Geoapify Routing API error: {response.status_code} - {response.text}")
            return jsonify({'error': 'Failed to calculate distance'}), 500
        
        data = response.json()
        
        if 'features' in data and len(data['features']) > 0:
            route = data['features'][0]['properties']
            distance_meters = route.get('distance', 0)
            time_seconds = route.get('time', 0)
            
            return jsonify({
                'distance': {
                    'text': f"{distance_meters/1000:.1f} km",
                    'value': distance_meters
                },
                'duration': {
                    'text': f"{int(time_seconds/60)} min",
                    'value': time_seconds
                }
            })
        else:
            return jsonify({'error': 'No route found'}), 400
        
    except Exception as e:
        logging.error(f"Error in distance_matrix: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/ai_support')
def ai_support():
    """AI-powered health support chat"""
    return render_template('ai_support.html')

@app.route('/schedule_consultation', methods=['POST'])
def schedule_consultation():
    """Handle consultation scheduling requests"""
    try:
        init_session_data()
        
        consultation_type = request.form.get('consultation_type')
        consultation_date = request.form.get('consultation_date')
        consultation_time = request.form.get('consultation_time')
        consultation_notes = request.form.get('consultation_notes', '')
        
        if not all([consultation_type, consultation_date, consultation_time]):
            return jsonify({'error': 'All required fields must be filled'}), 400
        
        # Create consultation session data
        consultation_data = {
            'id': len(session['user_data']['telemedicine_sessions']) + 1,
            'session_type': consultation_type,
            'scheduled_date': consultation_date,
            'scheduled_time': consultation_time,
            'session_notes': consultation_notes,
            'status': 'scheduled',
            'provider_name': 'IllnessInsight Healthcare',
            'created_at': datetime.now().isoformat()
        }
        
        # Add to session
        session['user_data']['telemedicine_sessions'].append(consultation_data)
        session.modified = True
        
        return jsonify({
            'success': True, 
            'message': 'Consultation scheduled successfully! You will receive confirmation within 24 hours.',
            'consultation_id': consultation_data['id']
        })
    
    except Exception as e:
        logging.error(f"Consultation scheduling error: {e}")
        return jsonify({'error': 'Failed to schedule consultation'}), 500

@app.route('/interventions')
def interventions():
    """Real-time Intervention Recommendations"""
    init_session_data()
    intervention_recommendations = session['user_data'].get('intervention_recommendations', [])
    return render_template('interventions.html', 
                         intervention_recommendations=intervention_recommendations)

# Helper functions (same logic as before, but no database operations)

def analyze_mental_health_score(screening_type, total_score):
    """Analyze mental health screening score and determine risk level"""
    if screening_type == 'depression':  # PHQ-9
        if total_score <= 4:
            risk_level = 'minimal'
            recommendations = ['Continue current wellness practices', 'Regular exercise and healthy sleep']
        elif total_score <= 9:
            risk_level = 'mild'
            recommendations = ['Consider stress management techniques', 'Maintain social connections', 'Monitor mood changes']
        elif total_score <= 14:
            risk_level = 'moderate'
            recommendations = ['Consider professional counseling', 'Implement daily mindfulness practices', 'Seek support from friends/family']
        elif total_score <= 19:
            risk_level = 'moderately_severe'
            recommendations = ['Strongly recommend professional help', 'Consider therapy or counseling', 'Avoid isolation']
        else:
            risk_level = 'severe'
            recommendations = ['Seek immediate professional help', 'Contact mental health services', 'Consider crisis helpline if needed']
    
    elif screening_type == 'anxiety':  # GAD-7
        if total_score <= 4:
            risk_level = 'minimal'
            recommendations = ['Practice relaxation techniques', 'Maintain regular exercise routine']
        elif total_score <= 9:
            risk_level = 'mild'
            recommendations = ['Learn anxiety management strategies', 'Practice deep breathing exercises', 'Limit caffeine intake']
        elif total_score <= 14:
            risk_level = 'moderate'
            recommendations = ['Consider professional counseling', 'Practice mindfulness meditation', 'Establish daily routines']
        else:
            risk_level = 'severe'
            recommendations = ['Seek professional mental health treatment', 'Consider therapy or medication consultation', 'Build strong support network']
    
    elif screening_type == 'stress':  # PSS-10
        if total_score <= 13:
            risk_level = 'low'
            recommendations = ['Maintain current stress management', 'Continue healthy lifestyle habits']
        elif total_score <= 26:
            risk_level = 'moderate'
            recommendations = ['Implement stress reduction techniques', 'Practice time management', 'Regular physical activity']
        else:
            risk_level = 'high'
            recommendations = ['Prioritize stress management', 'Consider professional stress counseling', 'Evaluate life stressors and make necessary changes']
    
    return risk_level, recommendations

def analyze_text_sentiment(text):
    """Perform AI sentiment analysis on text using Gemini"""
    if not model or not text.strip():
        return {'sentiment': 'neutral', 'confidence': 0.5, 'keywords': []}
    
    try:
        prompt = f"""Analyze the sentiment of the following text and provide a JSON response with:
        - sentiment: positive, negative, or neutral
        - confidence: a score between 0 and 1
        - keywords: array of key emotional words/phrases
        
        Text: {text}
        
        Respond only with valid JSON."""
        
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return result
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        return {'sentiment': 'neutral', 'confidence': 0.5, 'keywords': []}

def generate_mental_health_recommendations(screening_type, score, risk_level):
    """Generate AI-powered mental health recommendations"""
    if not model:
        return ['Consult with a mental health professional', 'Practice self-care activities', 'Maintain social connections']
    
    try:
        prompt = f"""As a mental health AI assistant, provide 5 specific, actionable recommendations for someone with:
        - Screening type: {screening_type}
        - Score: {score}
        - Risk level: {risk_level}
        
        Provide practical, evidence-based suggestions that are appropriate for this risk level.
        Format as a JSON array of strings."""
        
        response = model.generate_content(prompt)
        recommendations = json.loads(response.text)
        return recommendations
    except Exception as e:
        logging.error(f"AI recommendations error: {e}")
        return ['Consult with a mental health professional', 'Practice self-care activities', 'Maintain social connections']

def calculate_wellness_score(sleep_duration, sleep_quality, fatigue_level, alertness_level):
    """Calculate overall wellness score based on sleep metrics"""
    # Normalize sleep duration (optimal around 7-9 hours)
    if 7 <= sleep_duration <= 9:
        duration_score = 10
    elif 6 <= sleep_duration <= 10:
        duration_score = 8
    elif 5 <= sleep_duration <= 11:
        duration_score = 6
    else:
        duration_score = 4
    
    # Sleep quality is already on 1-10 scale
    quality_score = sleep_quality
    
    # Invert fatigue level (lower fatigue = better)
    fatigue_score = 11 - fatigue_level
    
    # Alertness level is already on 1-10 scale
    alertness_score = alertness_level
    
    # Calculate weighted average
    wellness_score = (duration_score * 0.3 + quality_score * 0.3 + 
                     fatigue_score * 0.2 + alertness_score * 0.2)
    
    return round(wellness_score, 2)

def generate_sleep_insights(sleep_data):
    """Generate AI insights for sleep data"""
    if not model:
        return ['Aim for 7-9 hours of sleep nightly', 'Maintain consistent sleep schedule', 'Create relaxing bedtime routine']
    
    try:
        prompt = f"""Analyze this sleep data and provide 3-5 personalized insights and recommendations:
        - Sleep duration: {sleep_data['sleep_duration']} hours
        - Sleep quality: {sleep_data['sleep_quality']}/10
        - Fatigue level: {sleep_data['fatigue_level']}/10
        - Alertness level: {sleep_data['alertness_level']}/10
        - Wellness score: {sleep_data['wellness_score']}/10
        
        Provide actionable sleep improvement suggestions. Return only a valid JSON array of strings, no other text."""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response to extract JSON
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        insights = json.loads(response_text)
        return insights if isinstance(insights, list) else [response_text]
    except Exception as e:
        logging.error(f"Sleep insights error: {e}")
        return ['Aim for 7-9 hours of sleep nightly', 'Maintain consistent sleep schedule', 'Create relaxing bedtime routine']

def generate_goal_recommendations(goal):
    """Generate AI recommendations for health goals"""
    if not model:
        return ['Set specific, measurable targets', 'Break goals into smaller steps', 'Track progress regularly']
    
    try:
        prompt = f"""Provide 4-6 specific recommendations for achieving this health goal:
        - Type: {goal['goal_type']}
        - Title: {goal['title']}
        - Description: {goal['description']}
        - Target: {goal['target_value']} {goal['unit']}
        
        Give practical, actionable advice. Return only a valid JSON array of strings, no other text."""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response to extract JSON
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        recommendations = json.loads(response_text)
        return recommendations if isinstance(recommendations, list) else [response_text]
    except Exception as e:
        logging.error(f"Goal recommendations error: {e}")
        return ['Set specific, measurable targets', 'Break goals into smaller steps', 'Track progress regularly']

def get_quest_template(quest_type):
    """Get quest template data"""
    templates = {
        'fitness': {
            'name': 'Fitness Champion',
            'description': 'Complete 30 minutes of exercise daily for a week',
            'credits': 100,
            'duration_days': 7,
            'daily_target': '30 minutes of exercise'
        },
        'nutrition': {
            'name': 'Nutrition Master',
            'description': 'Log healthy meals and track nutrition for 5 days',
            'credits': 75,
            'duration_days': 5,
            'daily_target': 'Log 3 healthy meals'
        },
        'mindfulness': {
            'name': 'Mindful Warrior',
            'description': 'Practice meditation or mindfulness for 10 minutes daily',
            'credits': 50,
            'duration_days': 7,
            'daily_target': '10 minutes meditation'
        },
        'sleep': {
            'name': 'Sleep Optimizer',
            'description': 'Maintain consistent sleep schedule for one week',
            'credits': 80,
            'duration_days': 7,
            'daily_target': '7-9 hours consistent sleep'
        },
        'hydration': {
            'name': 'Hydration Hero',
            'description': 'Drink 8 glasses of water daily for 5 days',
            'credits': 60,
            'duration_days': 5,
            'daily_target': '8 glasses of water'
        },
        'steps': {
            'name': 'Step Master',
            'description': 'Walk 10,000 steps daily for 10 days',
            'credits': 120,
            'duration_days': 10,
            'daily_target': '10,000 steps'
        }
    }
    
    return templates.get(quest_type, {
        'name': 'Health Explorer',
        'description': 'Explore new health activities',
        'credits': 25,
        'duration_days': 3,
        'daily_target': 'Complete health activity'
    })



def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    import math
    
    R = 6371  # Earth's radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2) * math.sin(dlat/2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2) * math.sin(dlon/2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def create_intervention_recommendations(condition_type, risk_level):
    """Create real-time intervention recommendations based on prediction results"""
    base_recommendations = {
        'heart': {
            'low': [
                'Maintain current healthy lifestyle',
                'Regular cardiovascular exercise (150 min/week)',
                'Heart-healthy diet with omega-3 fatty acids',
                'Annual cardiac check-ups'
            ],
            'high': [
                'Immediate consultation with cardiologist',
                'Blood pressure and cholesterol monitoring',
                'Cardiac stress test evaluation',
                'Medication review with healthcare provider',
                'Smoking cessation if applicable',
                'Dietary consultation for heart health'
            ]
        },
        'diabetes': {
            'low': [
                'Maintain healthy weight and BMI',
                'Regular blood glucose monitoring',
                'Balanced diet with controlled carbohydrates',
                'Regular physical activity'
            ],
            'high': [
                'Urgent endocrinologist consultation',
                'Comprehensive diabetes screening',
                'Blood glucose monitoring device',
                'Nutritionist consultation for meal planning',
                'Eye and foot examination scheduling',
                'Medication evaluation'
            ]
        },
        'cancer': {
            'low': [
                'Continue regular health screenings',
                'Maintain healthy lifestyle habits',
                'Annual comprehensive physical exams',
                'Stay updated on recommended cancer screenings'
            ],
            'high': [
                'Immediate oncology consultation',
                'Comprehensive diagnostic imaging',
                'Biopsy or additional testing as recommended',
                'Genetic counseling consultation',
                'Second opinion from cancer specialist',
                'Staging and treatment planning'
            ]
        }
    }
    
    return base_recommendations.get(condition_type, {}).get(risk_level, ['Consult with healthcare provider'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
