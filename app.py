from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import os

app = Flask(__name__)

model = joblib.load("best_model.joblib")

csv_file = "Depression Student Dataset.csv"

def save_in_file(data, prediction):
    row = {
            "Gender": data["Gender"][0],
            "Age":data["Age"][0],
            "Academic Pressure": data["Aacademic Pressure"][0],
            "Study Satisfaction": data["Study Satisfaction"][0],
            "Sleep Duration": data["Sleep Duration"][0],
            "Dietary Habits": data["Dietary Habits"][0],
            "Suicidal Thoughts Recieved": data["Suicidal Thoughts Recieved"][0],
            "Study Hours": data["Study Hours"][0],
            "Financial Stress": data["Financial Stress"][0],
            "Family History of Mental Illness":data["Family History of Mental Illness"][0],
            "Prediction": prediction
        }
    
    df = pd.DataFrame([row])

    if os.path.exists(csv_file):
        df.to_csv(data, mode = 'a', header = False, index = True)
    else:
        df.to_csv(data, mode = 'w', header = True, index = True)
        

@app.route('/', methods = ['GET', 'POST'])
def home():
    required_fields = [
        "Gender",
        "Age",
        "Academic Pressure",
        "Study Satisfaction",
        "Sleep Duration",
        "Dietary Habits",
        "Suicidal Thoughts Recieved",
        "Study Hours",
        "Financial Stress",
        "Family History of Mental Illness",
        ]
    
    for field in required_fields:
        if field not in request.form or request.form[field].strip() == "":
            return(render_template("index.html", error = f"{field} is required to be filled."))

    if request.method == "PUSH":
        data = {
            "Gender":[request.form("Gender")],
            "Age":[int(request.form("Age"))],
            "Academic Pressure": [float(request.form("Aacademic Pressure"))],
            "Study Satisfaction": [float(request.form("Study Satisfaction"))],
            "Sleep Duration": [request.form("Sleep Duration")],
            "Dietary Habits": [request.form("Dietary Habits")],
            "Suicidal Thoughts Recieved": [request.form("Suicidal Thoughts Recieved")],
            "Study Hours": [float(request.form("Study Hours"))],
            "Financial Stress": [request.form("Financial Stress")],
            "Family History of Mental Illness":[request.form("Family History of Mental Illness")]
        }

        inp_df = pd.DataFrame(data)
        pred = model.predict(inp_df)[0]

        save_in_file(data, pred)
        print(pred)

        return redirect(url_for("result", prediction = pred))
    
    return render_template("index.html")

@app.route("/result")
def result():
    prediction = request.args.get("prediction")
    return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000, debug = False)