from flask import Flask, render_template, request, send_file, redirect, url_for, session
import pandas as pd
import numpy as np
import pickle
import os
import sqlite3
from datetime import datetime
from io import BytesIO
from functools import wraps
import hashlib

app = Flask(__name__)
app.secret_key = "fraudshield_2024"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DB_PATH    = os.path.join(BASE_DIR, "fraud_history.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

df_ref = pd.read_csv("creditcard.csv")
if "Time" in df_ref.columns:
    df_ref.drop("Time", axis=1, inplace=True)
X_columns = df_ref.drop("Class", axis=1).columns

# ── HELPERS ────────────────────────────────────────────
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ── DATABASE ───────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            email     TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            role      TEXT DEFAULT 'user',
            created   TEXT
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            source      TEXT,
            amount      REAL,
            result      TEXT,
            probability REAL,
            filename    TEXT,
            username    TEXT
        )
    ''')

    admin_exists = conn.execute(
        "SELECT id FROM users WHERE username='admin'"
    ).fetchone()

    if not admin_exists:
        conn.execute('''
            INSERT INTO users (username, email, password, role, created)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            "admin",
            "admin@fraudshield.com",
            hash_password("fraud123"),
            "admin",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

    conn.commit()
    conn.close()


def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    user = conn.execute(
        "SELECT * FROM users WHERE username=?", (username,)
    ).fetchone()
    conn.close()
    return user


def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    user = conn.execute(
        "SELECT * FROM users WHERE email=?", (email,)
    ).fetchone()
    conn.close()
    return user


def create_user(username, email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
            INSERT INTO users (username, email, password, role, created)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            username, email,
            hash_password(password),
            "user",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    users = conn.execute(
        "SELECT id, username, email, role, created FROM users ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return users


def save_many(rows):
    conn = sqlite3.connect(DB_PATH)
    conn.executemany('''
        INSERT INTO transactions
        (timestamp, source, amount, result, probability, filename, username)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', rows)
    conn.commit()
    conn.close()


def get_rows():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM transactions ORDER BY id DESC LIMIT 500"
    ).fetchall()
    conn.close()
    return rows


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    total  = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud  = conn.execute("SELECT COUNT(*) FROM transactions WHERE result='FRAUD'").fetchone()[0]
    normal = conn.execute("SELECT COUNT(*) FROM transactions WHERE result='NORMAL'").fetchone()[0]
    conn.close()
    return total, fraud, normal


def get_analytics():
    conn = sqlite3.connect(DB_PATH)

    daily = conn.execute("""
        SELECT DATE(timestamp),
               COUNT(*),
               SUM(CASE WHEN result='FRAUD' THEN 1 ELSE 0 END)
        FROM transactions
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp) DESC
        LIMIT 14
    """).fetchall()

    amount_ranges = conn.execute("""
        SELECT
            CASE
                WHEN amount < 100  THEN 'Under 100'
                WHEN amount < 500  THEN '100-500'
                WHEN amount < 1000 THEN '500-1000'
                WHEN amount < 5000 THEN '1000-5000'
                ELSE 'Above 5000'
            END,
            COUNT(*)
        FROM transactions
        WHERE result='FRAUD'
        GROUP BY 1
    """).fetchall()

    sources = conn.execute("""
        SELECT source, COUNT(*) FROM transactions GROUP BY source
    """).fetchall()

    conn.close()
    return daily, amount_ranges, sources


init_db()


# ── AUTH DECORATORS ────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            return render_template("error.html",
                error="Access denied. Admins only.")
        return f(*args, **kwargs)
    return decorated


# ── LOGIN ──────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = get_user(username)
        if user and user[3] == hash_password(password):
            session["logged_in"] = True
            session["username"]  = user[1]
            session["email"]     = user[2]
            session["role"]      = user[4]
            return redirect(url_for("home"))
        error = "❌ Invalid username or password."
    return render_template("login.html", error=error)


# ── SIGNUP ─────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("logged_in"):
        return redirect(url_for("home"))
    error   = None
    success = None
    if request.method == "POST":
        username  = request.form.get("username", "").strip()
        email     = request.form.get("email", "").strip()
        password  = request.form.get("password", "")
        password2 = request.form.get("password2", "")

        if not username or not email or not password:
            error = "❌ All fields are required."
        elif len(username) < 3:
            error = "❌ Username must be at least 3 characters."
        elif len(password) < 6:
            error = "❌ Password must be at least 6 characters."
        elif password != password2:
            error = "❌ Passwords do not match."
        elif get_user(username):
            error = "❌ Username already taken."
        elif get_user_by_email(email):
            error = "❌ Email already registered."
        else:
            if create_user(username, email, password):
                success = "✅ Account created successfully! You can now login."
            else:
                error = "❌ Registration failed. Please try again."

    return render_template("signup.html", error=error, success=success)


# ── LOGOUT ─────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── HOME ───────────────────────────────────────────────
@app.route("/")
@login_required
def home():
    total, fraud, normal = get_stats()
    rows = get_rows()
    return render_template("index.html",
        db_total  = total,
        db_fraud  = fraud,
        db_normal = normal,
        rows      = rows,
        username  = session.get("username"),
        role      = session.get("role"))


# ── BATCH PREDICT ──────────────────────────────────────
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        file = request.files["file"]
        if not file.filename:
            return redirect(url_for("home"))

        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)

        if file.filename.lower().endswith(".csv"):
            data = pd.read_csv(filepath)
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            data = pd.read_excel(filepath)
        else:
            return render_template("error.html",
                error="Only CSV or Excel files are supported.")

        amounts = data["Amount"].values.copy() if "Amount" in data.columns else np.zeros(len(data))

        if "Amount" in data.columns:
            data["Amount"] = scaler.transform(data["Amount"].values.reshape(-1, 1))
        if "Time"  in data.columns:
            data.drop("Time",  axis=1, inplace=True)
        if "Class" in data.columns:
            data.drop("Class", axis=1, inplace=True)

        data  = data[X_columns]
        preds = model.predict(data)
        probs = model.predict_proba(data)[:, 1]

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db_rows = [
            (ts, "batch", round(float(amounts[i]), 2),
             "FRAUD" if preds[i] == 1 else "NORMAL",
             round(float(probs[i]) * 100, 2),
             file.filename,
             session.get("username"))
            for i in range(len(preds))
        ]
        save_many(db_rows)

        result_df = data.copy()
        result_df["Prediction"]  = preds
        result_df["Probability"] = np.round(probs * 100, 2)
        result_df.to_csv(
            os.path.join(RESULT_DIR, "prediction_results.csv"), index=False
        )

        total      = len(preds)
        fraud      = int(preds.sum())
        normal     = total - fraud
        percentage = round(fraud / total * 100, 2)

        fraud_table = result_df[result_df["Prediction"] == 1].head(20).to_html(
            classes="table table-striped table-hover table-sm", index=False
        )

        return render_template("results.html",
            total      = total,
            fraud      = fraud,
            normal     = normal,
            percentage = percentage,
            table      = fraud_table,
            username   = session.get("username"))

    except Exception as e:
        return render_template("error.html", error=str(e))


# ── SINGLE PREDICT ─────────────────────────────────────
@app.route("/predict_single", methods=["POST"])
@login_required
def predict_single():
    try:
        vals   = [float(request.form.get(f"V{i}", 0)) for i in range(1, 29)]
        amount = float(request.form.get("Amount", 0))
        vals.append(scaler.transform([[amount]])[0][0])

        df     = pd.DataFrame([vals], columns=X_columns)
        pred   = model.predict(df)[0]
        prob   = round(float(model.predict_proba(df)[0][1]) * 100, 2)
        result = "FRAUD" if pred == 1 else "NORMAL"

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_many([(ts, "manual", amount, result, prob,
                    "manual entry", session.get("username"))])

        return render_template("single_result.html",
            result      = "🚨 FRAUD DETECTED" if pred == 1 else "✅ LEGITIMATE TRANSACTION",
            color       = "danger" if pred == 1 else "success",
            probability = prob,
            amount      = amount,
            username    = session.get("username"))

    except Exception as e:
        return render_template("error.html", error=str(e))


# ── HOW IT WORKS ───────────────────────────────────────
@app.route("/how_it_works")
@login_required
def how_it_works():
    return render_template("how_it_works.html",
        username=session.get("username"))


# ── ANALYTICS ──────────────────────────────────────────
@app.route("/analytics")
@login_required
def analytics():
    total, fraud, normal = get_stats()
    daily, amount_ranges, sources = get_analytics()
    return render_template("analytics.html",
        total         = total,
        fraud         = fraud,
        normal        = normal,
        daily         = daily,
        amount_ranges = amount_ranges,
        sources       = sources,
        username      = session.get("username"))


# ── ADMIN USERS ────────────────────────────────────────
@app.route("/admin/users")
@admin_required
def admin_users():
    users = get_all_users()
    return render_template("admin_users.html",
        users    = users,
        username = session.get("username"))


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "DELETE FROM users WHERE id=? AND username != 'admin'", (user_id,)
    )
    conn.commit()
    conn.close()
    return redirect(url_for("admin_users"))


# ── EXPORT EXCEL ───────────────────────────────────────
@app.route("/export_excel")
@login_required
def export_excel():
    rows = get_rows()
    df   = pd.DataFrame(rows, columns=[
        "ID","Timestamp","Source","Amount",
        "Result","Probability","Filename","User"
    ])
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="History")
    out.seek(0)
    fname = f"fraud_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(out, as_attachment=True, download_name=fname,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ── CLEAR HISTORY ──────────────────────────────────────
@app.route("/clear_history", methods=["POST"])
@login_required
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()
    return redirect(url_for("home"))


# ── DOWNLOAD CSV ───────────────────────────────────────
@app.route("/download")
@login_required
def download():
    return send_file(
        os.path.join(RESULT_DIR, "prediction_results.csv"),
        as_attachment=True
    )


if __name__ == "__main__":
    print("🚀 http://127.0.0.1:5000")
    app.run(debug=False)