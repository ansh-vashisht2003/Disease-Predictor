from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models
diabetes_model = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\ML project\model\diabetes_model.pkl")
heart_model = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\ML project\model\heart_model.pkl")  
liver_model = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\ML project\model\liver_disease_model.pkl")  

# Routes to render HTML pages
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


# ================================
# ✅ DIABETES Prediction Route
# ================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        prediction = diabetes_model.predict([np.array(data)])[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

# ================================
# ✅ HEART Disease Prediction Route
# ================================
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        data = [float(x) for x in request.form.values()]
        prediction = heart_model.predict([np.array(data)])[0]
        result = "Heart Problem" if prediction == 1 else "No Problem"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

# ================================
# ✅ LIVER Disease Prediction Route (fixed to use form data)
# ================================
@app.route("/predict_liver", methods=["POST"])
def predict_liver():
    try:
        data = [float(request.form[key]) for key in [
            "age", "gender", "total_bilirubin", "direct_bilirubin",
            "alk_phos", "alt", "ast", "total_protein", "albumin", "agr"
        ]]
        prediction = liver_model.predict([np.array(data)])[0]
        result = "Has Liver Problem" if prediction == 2 else "No Liver Problem"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})


# ================================
# ✅ Run the Flask app
# ================================
if __name__ == "__main__":
    app.run(debug=True)
