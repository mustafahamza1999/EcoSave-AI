# EcoSave AI

A Flask-based intelligent waste and recyclable material classification web application powered by a manually implemented Multi-Layer Perceptron (MLP) neural network (`ManualNeuralNetwork`), interactive Plotly visualizations, an admin dashboard for dataset updates and model retraining, and automated PDF report generation via `FPDF`.

## Key Features

- **Manual MLP Neural Network:** Custom neural network implementation built from scratch for material classification and confidence scoring.
- **Interactive Analytics:** Visualizes prediction insights using Plotly charts alongside recent session search history.
- **Admin Dashboard:** Supports authenticated admin access, dataset statistics, and CSV uploads to trigger model retraining.
- **Automated PDF Reports:** Generates downloadable PDF summaries for classification results and suggested recycling/reuse steps.

## Setup & Usage

1. Clone the repository:
   git clone https://github.com/mustafahamza1999/EcoSave-AI.git
   cd EcoSave-AI

2. Create a virtual environment and install dependencies:
   python -m venv venv
   venv\Scripts\activate
   pip install flask pandas numpy plotly fpdf2

3. Configure environment variables (`FLASK_SECRET_KEY`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`) before running the application.

4. Run the application:
   python app.py

## Note

This project was developed as an AI Engineering graduation project for educational and demonstration purposes. Avoid committing real credentials, `.env` files, or local SQLite databases (`*.db`) to version control.