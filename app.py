from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/Inputs", methods=['POST'])
def Home():
    if request.method == 'POST':
        Username = request.form['Username']
        Password = request.form['Password']

        
        if Username == Password:

            return render_template("index.html")
        else:
            return render_template("login.html")

    return render_template("login.html")

@app.route('/Validate', methods=['POST'])
def Next():
    Data1 = float(request.form['Data1'])
    Data2 = float(request.form['Data2'])
    Data3 = float(request.form['Data3'])
    Data4 = float(request.form['Data4'])
    Data5 = float(request.form['Data5'])
    Data6 = float(request.form['Data6'])
    Data7 = float(request.form['Data7'])

    Arr = np.array([[Data1, Data2, Data3, Data4, Data5, Data6, Data7]])

    Prediction = model.predict(Arr)
    prediction_str = str(int(Prediction[0]))  # Convert prediction to int and then to str
    return render_template('output.html', prediction=prediction_str)

@app.route('/Back', methods=['POST'])
def BacktoInput():
      return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
