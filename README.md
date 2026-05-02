# Headline Emotionality Rater

A crowdsourcing app for rating the emotional intensity of news headlines, built with Streamlit and Supabase.

**Live app:** [headline-shift.streamlit.app](https://headline-shift.streamlit.app/)

## What it does

- Shows pairs of news headlines side by side
- You pick which one feels more emotional or sensational
- Ratings are stored in Supabase and used to compute emotionality scores via the **Bradley-Terry** model
- **Active learning** prioritizes the most informative pairs to label next
- Export a CSV of labeled headlines for model training

## Running locally

1. Install dependencies:
   ```bash
   pip install -r requirements-app.txt
   ```

2. Create `.streamlit/secrets.toml` with your Supabase credentials:
   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your-anon-key"
   ```

3. Run the app:
   ```bash
   streamlit run app/active_learning_app.py
   ```

## Deployment

The app is deployed on [Streamlit Community Cloud](https://share.streamlit.io) and updates automatically on every push to `main`. Annotations persist in Supabase (PostgreSQL).
