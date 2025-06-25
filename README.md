# IllnessInsight - AI Health Prediction Platform

A comprehensive AI-powered health platform that provides predictive analytics for various diseases, mental health screening, sleep tracking, and personalized health coaching.

## 🧠 Team & Approach
 
 ### Team Name:  
 `Outlier Coders`
 
 ### Team Members:  
 - Mokshit Kaushik (Team Leader)  
 - Kanishka Sharma 
 - Sumukhi Tripathi
 - Parth Sharma

### Approach:  
1. Comprehensive Health Ecosystem Design
 - Holistic Solution: Instead of building separate tools, you created an integrated platform that addresses multiple health aspects - prediction, monitoring, coaching, and support
 - User-Centric Design: Focused on making complex health data accessible and actionable for everyday users
2. AI-First Architecture
 - Multi-Model Integration: Combined traditional ML models (scikit-learn) with generative AI (Google Gemini) for different use cases
 - Fallback Systems: Implemented graceful degradation when AI services aren't available, ensuring the platform always functions
 - Context-Aware Recommendations: AI generates personalized advice based on user data and risk profiles
3. Session-Based Lightweight Architecture
 - No Database Dependency: Used Flask sessions for data storage, making deployment simple and reducing infrastructure complexity

## ✨ Key Features
 
The most important features of our project:
## 🤖 AI-Powered Disease Prediction
   - Heart Disease Risk Assessment: ML model-based prediction with risk factor analysis
   - Diabetes Prediction: Comprehensive screening using health metrics and family history
   - Cancer Detection: Voice biomarker and risk factor analysis for early detection
   - Real-time Risk Analysis: Instant predictions with confidence scores and recommendations
##🧠 Mental Health Screening
   - PHQ-9 Depression Screening: Validated assessment with severity classification
   - GAD-7 Anxiety Evaluation: Professional-grade anxiety disorder screening
   - PSS-10 Stress Assessment: Perceived stress scale with personalized insights
   - Sentiment Analysis: AI-powered text analysis for emotional state detection
## 💬 AI Health Assistant
   - 24/7 Chat Support: Google Gemini-powered conversational AI for health guidance
   - Instant Medical Information: Real-time answers to health questions and concerns
   - Personalized Recommendations: Context-aware health advice based on user data
   - Multi-topic Support: General health, wellness, and medical query assistance
## 🎯 Personalized Health Coaching
   - Custom Goal Setting: Create fitness, nutrition, weight, and wellness goals
   - Progress Tracking: Real-time updates with visual progress bars and analytics
   - AI Recommendations: Personalized coaching tips and milestone suggestions
   - Goal Management: Active goal monitoring with completion tracking
## 🎮 Gamified Health Quests
   - Health Challenges: Fitness, nutrition, mindfulness, sleep, and hydration quests
   - Credit Rewards System: Earn credits for completing health activities
   - Progress Gamification: Daily targets with completion tracking
   - Multiple Quest Types: 6 different quest categories with varying difficulty levels
## 😴 Sleep & Wellness Tracking
   - Sleep Quality Monitoring: Duration, quality, fatigue, and alertness tracking
   - Wellness Score Calculation: Comprehensive health metrics analysis
   - Sleep Insights: AI-generated recommendations for sleep improvement
   - Pattern Analysis: Historical sleep data tracking and trend analysis
## 🏥 Telemedicine Support
   - Medical Facility Finder: Geolocation-based healthcare provider search
   - Consultation Scheduling: Virtual appointment booking system
   - Emergency Contacts: Quick access to crisis hotlines and emergency services
   - Facility Details: Comprehensive information about medical facilities
## 📊 Real-time Interventions
   - Risk-based Recommendations: Immediate interventions based on prediction results
   - Personalized Action Plans: Tailored recommendations for different risk levels
   - Healthcare Referrals: Professional consultation suggestions when needed
   - Emergency Protocols: Crisis intervention and emergency contact information
![123](https://github.com/user-attachments/assets/c439e7f3-3641-4db1-ba6d-a1bd71d82d15)
![3](https://github.com/user-attachments/assets/aa7ddc79-edd5-485c-8e8c-e84d849cde8f)
![4](https://github.com/user-attachments/assets/e442f9c6-5030-49f5-a7c7-5f1b8317aaf1)
![5](https://github.com/user-attachments/assets/dd996c22-d0fd-48fb-a483-928f7569fc25)
![7](https://github.com/user-attachments/assets/7f01054b-9013-4c26-b70c-c6d7fcb89066)
![8](https://github.com/user-attachments/assets/cd3b8d62-db7f-4688-9ab4-edf5714fb19d)
![9](https://github.com/user-attachments/assets/1be23d9d-d37d-41c2-b5a9-eb6a19eb29c7)
![10](https://github.com/user-attachments/assets/d5b7123e-a2e7-409e-b062-f89639274f32)
![11](https://github.com/user-attachments/assets/330c556a-cd37-4573-977c-b3619f5b909a)
![13](https://github.com/user-attachments/assets/a69b77a9-5b56-4b00-a739-f813f4c7a161)
![14](https://github.com/user-attachments/assets/67257bec-3a01-440d-9422-86a84c05f798)
![16](https://github.com/user-attachments/assets/411c3979-6029-4c5c-93fa-c1545f5055b7)

## Local Setup Instructions

### Prerequisites

- Python 3.11 or higher
- uv package manager (recommended) or pip

### Installation

1. **Clone the repo**
   ```bash
   git clone (https://github.com/GeneralReznov/illnessInsightforpeople)
   cd illnessinsight_deploy
   ```

2. **Install uv (if not already installed)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```
   
   Or with pip:
   ```bash
   pip install -e .
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_google_gemini_api_key_here
   SESSION_SECRET=your_random_session_secret_here
   ```

5. **Run the application**
   ```bash
   uv run gunicorn --bind 0.0.0.0:5000 --reload main:app
   ```
   
   Or with Python directly:
   ```bash
   python -m gunicorn --bind 0.0.0.0:5000 --reload main:app
   ```

6. **Access the application**
   Open your browser and navigate to: `http://localhost:5000`

## 📽️ Demo & Deliverables
 
 - **Demo Video Link:** https://drive.google.com/file/d/1EJN-jRF0Y7BPh4ieNoUu61W5SckYFbsj/view?usp=drive_link
 - **App Link:** https://illnessinsightforpeople.onrender.com/


## Environment Variables

- `GEMINI_API_KEY`: Required for AI chat functionality (get from Google AI Studio)
- `SESSION_SECRET`: Secret key for Flask sessions (generate a random string)
- `GEOAPIFY_API_KEY`: Required for Location Access and searching.
### Key Challenges:
 - Dependency Management & Graceful Degradation
 - Session-Based Storage Complexity
 - Ensuring data privacy and security, leading to local recording to protect user content.
 - AI Integration Challenges
 - Cross-Platform ML Model Loading
 - Gamification Logic
 - Real-Time Progress Tracking

## Architecture

- **Backend**: Flask web framework with session-based storage
- **AI Integration**: Google Gemini 2.0 Flash for chat assistance
- **Frontend**: Bootstrap 5 with vanilla JavaScript
- **ML Models**: Scikit-learn for disease prediction
