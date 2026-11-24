from flask import Flask, render_template, request, send_file, redirect, session, url_for
import pandas as pd
import numpy as np
import sqlite3
import os
import plotly.express as px
import plotly.io as pio
from fpdf import FPDF
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
DB_PATH = os.path.join(BASE_DIR, "users.db")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS admins
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       username TEXT UNIQUE,
                       password TEXT)""")

    default_user = os.environ.get("ADMIN_USERNAME")
    default_pass = os.environ.get("ADMIN_PASSWORD")
    if default_user and default_pass:
        try:
            cursor.execute(
                "INSERT INTO admins (username, password) VALUES (?, ?)",
                (default_user, default_pass),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass

    conn.close()


init_db()


class ManualNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def train(self, X, y, epochs=10000, lr=0.05):
        for _ in range(epochs):
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.sigmoid(z2)
            error = a2 - y
            d_z2 = error * (a2 * (1 - a2))
            d_W2 = np.dot(a1.T, d_z2)
            error_hidden = np.dot(d_z2, self.W2.T)
            d_z1 = error_hidden * (a1 * (1 - a1))
            d_W1 = np.dot(X.T, d_z1)
            self.W2 -= lr * d_W2
            self.W1 -= lr * d_W1

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return self.sigmoid(z2)


def generate_plot(labels, scores):
    clean_labels = [l.split("|")[0] for l in labels]
    fig = px.bar(
        x=clean_labels,
        y=scores,
        labels={"x": "المواد", "y": "نسبة الثقة"},
        title="تحليل احتمالات التنبؤ (Neural Network Analysis)",
        color=scores,
        color_continuous_scale="Viridis",
    )
    return pio.to_html(fig, full_html=False)


def simple_vectorizer(texts, vocab=None):
    if vocab is None:
        vocab = sorted(list(set(" ".join(texts).split())))
    vectors = []
    for text in texts:
        words = str(text).split()
        vector = [1 if word in words else 0 for word in vocab]
        vectors.append(vector)
    return np.array(vectors), vocab


def train_model():
    global df, X_raw, y_labels, X_train, vocabulary, unique_labels, y_train, nn
    if not os.path.exists(DATASET_PATH):
        df = pd.DataFrame({"item": ["test"], "category": ["test"], "instruction": ["test"]})
    else:
        df = pd.read_csv(DATASET_PATH)

    X_raw = df["item"].values
    y_labels = (df["category"] + "|" + df["instruction"]).values
    X_train, vocabulary = simple_vectorizer(X_raw)
    unique_labels = sorted(list(set(y_labels)))
    y_train = np.array([[1 if label == l else 0 for l in unique_labels] for label in y_labels])
    nn = ManualNeuralNetwork(input_size=len(vocabulary), hidden_size=20, output_size=len(unique_labels))
    nn.train(X_train, y_train)


search_history = []
train_model()


@app.route("/")
def home():
    return render_template("index.html", history=search_history)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        user_in = request.form.get("username")
        pass_in = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM admins WHERE username=? AND password=?", (user_in, pass_in))
        admin = cur.fetchone()
        conn.close()

        if admin:
            session["logged_in"] = True
            session["username"] = user_in
            return redirect(url_for("admin_dashboard"))
        error = "خطأ في بيانات الدخول!"
    return render_template("login.html", error=error)


@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("admin.html", total_data=len(df), history=search_history)


@app.route("/generate", methods=["POST"])
def generate():
    user_input = request.form.get("ingredients", "").strip()
    input_vec, _ = simple_vectorizer([user_input], vocab=vocabulary)

    if np.sum(input_vec) == 0:
        return render_template(
            "index.html",
            recipe="⚠️ المادة غير مدعومة في التدريب الحالي.",
            old_ingredients=user_input,
            history=search_history,
        )

    prediction = nn.predict(input_vec)[0]
    max_idx = np.argmax(prediction)
    max_score = prediction[max_idx]

    if max_score < 0.3:
        return render_template(
            "index.html",
            recipe="⚠️ ثقة التنبؤ منخفضة جداً.",
            old_ingredients=user_input,
            history=search_history,
        )

    category, instruction = unique_labels[max_idx].split("|")
    result_data = {
        "item": user_input,
        "category": category,
        "steps": [s.strip() for s in instruction.split(".") if len(s.strip()) > 5],
        "confidence": round(max_score * 100, 2),
        "graph": generate_plot(unique_labels, prediction),
        "instruction": instruction,
    }
    search_history.insert(0, {"item": user_input, "result": category, "score": result_data["confidence"]})
    return render_template("index.html", result=result_data, old_ingredients=user_input, history=search_history)


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    file = request.files.get("file")
    if file and file.filename.endswith(".csv"):
        file.save(DATASET_PATH)
        train_model()
        return redirect(url_for("admin_dashboard"))
    return "ملف غير صالح"


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/download_report", methods=["POST"])
def download_report():
    item = request.form.get("item", "N/A")
    category = request.form.get("category", "N/A")
    confidence = request.form.get("confidence", "0")
    instruction = request.form.get("instruction", "")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="EcoSave AI Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    clean = lambda t: t.encode("latin-1", "ignore").decode("latin-1")
    pdf.cell(200, 10, txt=f"Material: {clean(item)}", ln=True)
    pdf.cell(200, 10, txt=f"Category: {clean(category)}", ln=True)
    pdf.cell(200, 10, txt=f"Score: {confidence}%", ln=True)
    pdf.multi_cell(0, 10, txt=f"Steps: {clean(instruction)}")
    output = io.BytesIO()
    pdf_string = pdf.output(dest="S")
    output.write(pdf_string.encode("latin-1") if isinstance(pdf_string, str) else pdf_string)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="Report.pdf", mimetype="application/pdf")


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)