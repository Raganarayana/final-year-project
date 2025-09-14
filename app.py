import json
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import joblib
import pandas as pd

APP_SECRET = "super-secret-change-me"
DB_URL = "sqlite:///mental_health.db"

app = Flask(__name__)
app.secret_key = APP_SECRET

# ------------- DB Setup -------------
Base = declarative_base()
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)

class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    predicted_status = Column(String(100), nullable=False)
    specialist = Column(String(100), nullable=False)
    appt_date = Column(String(20), nullable=False)
    appt_time = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

Base.metadata.create_all(engine)

# Create default user if not exists
with SessionLocal() as db:
    if not db.query(User).filter_by(username="admin").first():
        u = User(username="admin", password_hash=generate_password_hash("admin123"))
        db.add(u)
        db.commit()

# ------------- Model + Meta -------------
MODEL_PATH = Path("model.pkl")
META_PATH = Path("model_meta.json")
if not MODEL_PATH.exists() or not META_PATH.exists():
    raise RuntimeError("Model not found. Please run `python train_model.py` first.")

clf = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

TARGET = META["target"]
NUM_FEATURES = META["numeric_features"]
CAT_FEATURES = META["categorical_features"]
CAT_CHOICES = META["categorical_choices"]
CLASSES = META["classes_"]
POSITIVE_LABEL = META["positive_label"]

# ------------- Helpers -------------
def login_required(func):
    """Simple decorator to protect routes"""
    def wrapper(*args, **kwargs):
        if "username" not in session:
            flash("Please log in first!", "warning")
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

def positive_prob(prob_vector):
    if not hasattr(clf, "predict_proba"):
        return 0.5
    labels = list(CLASSES)
    if POSITIVE_LABEL in labels:
        idx = labels.index(POSITIVE_LABEL)
        return float(prob_vector[idx])
    return float(max(prob_vector))

def status_text(pred_label):
    if str(pred_label) == str(POSITIVE_LABEL):
        return "Needs Medical Guidance"
    return "Healthy"

# ------------- Routes -------------
@app.route("/")
def root():
    return redirect(url_for("login"))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # ✅ Allow any username, password must be exactly 6 digits
        if password.isdigit() and len(password) == 6:
            session['username'] = username
            flash(f"Welcome {username}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Password must be exactly 6 digits!", "danger")

    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session.get("username", "User"))

@app.route("/survey")
@login_required
def survey():
    return render_template(
        "survey.html",
        username=session.get("username", "User"),
        numeric_features=NUM_FEATURES,
        categorical_features=CAT_FEATURES,
        categorical_choices=CAT_CHOICES,
    )

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    # Build feature row
    row = {}
    for c in NUM_FEATURES:
        v = request.form.get(c, "")
        try:
            row[c] = float(v) if v != "" else None
        except:
            row[c] = None
    for c in CAT_FEATURES:
        row[c] = request.form.get(c, "") or None

    X_input = pd.DataFrame([row])
    pred_label = clf.predict(X_input)[0]
    proba_vec = clf.predict_proba(X_input)[0] if hasattr(clf, "predict_proba") else [0.5, 0.5]

    pos_p = positive_prob(proba_vec)
    stress_score = round(pos_p * 100, 2)
    overall_status = status_text(pred_label)

    # ✅ Dynamic recommendations
    recommendations = []
    if overall_status == "Needs Medical Guidance":
        if stress_score >= 70:
            recommendations = [
                "Consult a psychiatrist or psychologist as soon as possible.",
                "Practice daily relaxation techniques (meditation, deep breathing).",
                "Avoid overworking yourself, maintain a proper sleep schedule.",
            ]
        elif 40 <= stress_score < 70:
            recommendations = [
                "Try physical activities like yoga or walking regularly.",
                "Talk to friends or family about how you feel.",
                "If stress persists, consider professional counseling.",
            ]
        else:
            recommendations = [
                "Maintain a balanced lifestyle.",
                "Do regular self check-ins on your mental health.",
            ]
    else:  # Healthy
        recommendations = [
            "Keep up your healthy lifestyle (exercise, nutrition, sleep).",
            "Stay socially connected and active.",
            "Continue stress-relieving hobbies like reading, sports, or music.",
        ]

    # ✅ Pie chart data
    pie_data = {
        "labels": ["Stressed (prob.)", "Normal (prob.)"],
        "values": [pos_p, 1 - pos_p],
    }

    # ✅ Stress history (keep in session)
    history = session.get("stress_history", [])
    labels = session.get("history_labels", [])

    history.append(stress_score)
    labels.append(datetime.now().strftime("%H:%M"))

    history = history[-5:]
    labels = labels[-5:]

    session["stress_history"] = history
    session["history_labels"] = labels

    return render_template(
        "result.html",
        username=session.get("username", "User"),
        overall_status=overall_status,
        predicted_label=str(pred_label),
        positive_label=str(POSITIVE_LABEL),
        stress_score=stress_score,
        classes=CLASSES,
        prob_values=[float(x) for x in proba_vec],
        pie_labels=pie_data["labels"],
        pie_values=pie_data["values"],
        recommendations=recommendations,
        stress_history=history,
        history_labels=labels,
    )

@app.route("/book", methods=["GET", "POST"])
@login_required
def book():
    if request.method == "POST":
        specialist = request.form.get("specialist", "Psychiatrist")
        appt_date = request.form.get("appt_date", "")
        appt_time = request.form.get("appt_time", "")
        predicted_status = request.form.get("predicted_status", "Needs Medical Guidance")

        with SessionLocal() as dbs:
            appt = Appointment(
                user_id=1,  # simple static user ID (no flask-login)
                predicted_status=predicted_status,
                specialist=specialist,
                appt_date=appt_date,
                appt_time=appt_time,
                notes=request.form.get("notes", ""),
            )
            dbs.add(appt)
            dbs.commit()
            appt_id = appt.id

        return render_template(
            "booking_confirmation.html",
            username=session.get("username", "User"),
            appt_id=appt_id,
            specialist=specialist,
            appt_date=appt_date,
            appt_time=appt_time,
        )

    predicted_status = request.args.get("predicted_status", "Needs Medical Guidance")
    return render_template(
        "result.html",
        username=session.get("username", "User"),
        overall_status=predicted_status,
        predicted_label=predicted_status,
        positive_label=str(POSITIVE_LABEL),
        stress_score=None,
        classes=CLASSES,
        prob_values=None,
        pie_labels=["Stressed (prob.)", "Normal (prob.)"],
        pie_values=[0.5, 0.5],
        recommendations=[],
        show_booking_only=True,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
