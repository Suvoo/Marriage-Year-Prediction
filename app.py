import flask
from flask import request
import joblib

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# main index page route
@app.route('/')
def home():
    return '<h1>API is working.. </h1>'

@app.route('/predict',methods=['GET'])
def predict():
    model = joblib.load('marriage_age_predict_model.ml')
    '''age_predict = model.predict([[0,2,5,6,5,25,56,175]])
    return str(age_predict)'''
    predicted_age_of_marriage = model.predict([[int(request.args['gender']),
                            int(request.args['religion']),
                            int(request.args['caste']),
                            int(request.args['mother_tongue']),
                            int(request.args['country']),
                            int(request.args['height_cms']),
                           ]])
    return str(round(predicted_age_of_marriage[0],2))

if __name__ == "__main__":
    app.run(debug=True)

# http://127.0.0.1:5000/predict?gender=0&religion=2&caste=5&mother_tongue=6&country=5&profession=25&location=56&height_cms=175