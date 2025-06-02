from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models
diabetes_model = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\ML project\model\diabetes_model.pkl")
heart_model = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\ML project\model\heart_model.pkl")  
liver_model = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\ML project\model\liver_disease_model.pkl")  

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")

# Diabetes Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        prediction = diabetes_model.predict([np.array(data)])[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

# Heart Disease Prediction Route
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        data = [float(x) for x in request.form.values()]
        prediction = heart_model.predict([np.array(data)])[0]
        result = "Heart Problem" if prediction == 1 else "No Problem"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

# Liver Disease Prediction Route
@app.route("/predict_liver", methods=["POST"])
def predict_liver():
    try:
        req_data = request.get_json()

        data = [
            float(req_data["age"]),
            float(req_data["gender"]),
            float(req_data["total_bilirubin"]),
            float(req_data["direct_bilirubin"]),
            float(req_data["alk_phos"]),
            float(req_data["alt"]),
            float(req_data["ast"]),
            float(req_data["total_protein"]),
            float(req_data["albumin"]),
            float(req_data["agr"])
        ]

        prediction = liver_model.predict([np.array(data)])[0]
        result = "Has Liver Problem" if prediction == 2 else "No Liver Problem"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
