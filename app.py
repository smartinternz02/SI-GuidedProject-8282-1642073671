from flask import Flask, render_template,request,url_for
import pickle 
import numpy as np

app = Flask('__name__')
model = pickle.load(open('diabetes.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediction',methods=['POST'])
def predict():
    features = [[float(i) for i in request.form.values()]]
    features = np.array(features)
    pred_val = model.predict(features)
    if(pred_val[0]==1):
        results = "Result : Positive"
    else:
        results = "Result : Negative"

    return render_template('index.html',results = results)

if __name__ =="__main__":
    app.run(debug=True)
