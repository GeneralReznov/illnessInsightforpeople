# IllnessInsight - AI Health Prediction Platform

A comprehensive AI-powered health platform that provides predictive analytics for various diseases, mental health screening, sleep tracking, and personalized health coaching.

## Features

- **Disease Prediction Models**: Heart disease, diabetes, and cancer risk assessment
- **AI Assistant**: General-purpose chatbot powered by Google Gemini
- **Mental Health Screening**: PHQ-9, GAD-7, PSS-10 assessments
- **Telemedicine Support**: AI health assistance and medical facility finder
- **Sleep & Wellness Tracking**: Sleep quality monitoring and insights
- **Health Coaching**: Personalized goals and gamified quests

## Local Setup Instructions

### Prerequisites

- Python 3.11 or higher
- uv package manager (recommended) or pip

### Installation

1. **Extract the project files**
   ```bash
   unzip illnessinsight_deploy.zip
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

## Environment Variables

- `GEMINI_API_KEY`: Required for AI chat functionality (get from Google AI Studio)
- `SESSION_SECRET`: Secret key for Flask sessions (generate a random string)

## Emergency Contact Numbers

The platform includes these emergency contacts:
- Emergency Services: 108
- Crisis Hotline: 112
- Suicide Crisis Line: 9152987821
- Poison Control: 1800-425-1213

## Architecture

- **Backend**: Flask web framework with session-based storage
- **AI Integration**: Google Gemini 2.0 Flash for chat assistance
- **Frontend**: Bootstrap 5 with vanilla JavaScript
- **ML Models**: Scikit-learn for disease prediction (fallback responses if not available)

## Notes

- The application uses session-based storage (no database required)
- ML prediction models are optional - the app works without them
- All emergency numbers and medical information should be verified for your region
- This is a demonstration platform - consult healthcare professionals for medical advice

## Support

For issues or questions about setup, refer to the code comments or check the main.py file for configuration details.