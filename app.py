from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# -------------------------------------------------------------
# Load Model, TF-IDF Vectorizer, and Label Encoder (using joblib)
# -------------------------------------------------------------
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    model, vectorizer, label_encoder = None, None, None


# -------------------------------------------------------------
# Home Route
# -------------------------------------------------------------
@app.route("/")
def home():
    """Render the home page with the input form."""
    return render_template("index.html")


# -------------------------------------------------------------
# Prediction Route
# -------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    if not model or not vectorizer or not label_encoder:
        return render_template("index.html", error="Model not loaded properly. Please check your files.")

    lyrics = request.form.get("lyrics", "")

    # Empty input check
    if not lyrics.strip():
        return render_template("index.html", error="Please enter some lyrics before predicting!")

    try:
        # Convert lyrics into TF-IDF features
        features = vectorizer.transform([lyrics])

        # Predict the encoded label
        pred = model.predict(features)[0]

        # Decode label to mood name (e.g., "Happy", "Sad", etc.)
        mood = label_encoder.inverse_transform([pred])[0]

        return render_template("index.html", lyrics=lyrics, prediction=mood)

    except Exception as e:
        return render_template("index.html", error=f"Something went wrong while predicting: {e}")


# -------------------------------------------------------------
# Run the Flask app
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
