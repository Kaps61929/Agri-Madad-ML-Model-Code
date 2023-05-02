import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, redirect, flash, session, escape
import numpy as np
import pandas as pd
from joblib import load
app = Flask(__name__)

#Load the model
potato_model = pickle.load(open("potato_model.pkl", "rb"))
wheat_model = pickle.load(open("wheat_model.pkl", "rb"))
# transformer = pickle.load(open("potato_tranformer.pkl", "rb"))
columns = pickle.load(open("crop_columns.pkl", "rb"))
potato_transformer = load(filename="potato_transformer.joblib")
wheat_transformer = load(open("wheat_transformer.pkl", "rb"))


class SchemaValidator(object):
    def __init__(self, response = {}):
        self.response = response
    
    def isValidated(self):
        errorMessages = []

        try:
            crop_type = self.response.get("CROP TYPE", None)
            if crop_type is None or len(crop_type) == 0:
                raise Exception("Error")
        except Exception as e : errorMessages.append("crop type is required")

        try:
            soil_type = self.response.get("SOIL TYPE", None)
            if soil_type is None or len(soil_type) == 0:
                raise Exception("Error")
        except Exception as e : errorMessages.append("soil type is required")

        try:
            region = self.response.get("REGION", None)
            if region is None or len(region) == 0:
                raise Exception("Error")
        except Exception as e : errorMessages.append("region is not defined")


        try:
            weather_condition = self.response.get("WEATHER CONDITION", None)
            if weather_condition is None or len(weather_condition) == 0:
                raise Exception("Error")
        except Exception as e : errorMessages.append("weather condition is required")

        try:
            temp_min = self.response.get("TEMP MIN", None)
            print(temp_min)
            
            if temp_min is None:
                raise Exception("Error")
    
        except Exception as e : errorMessages.append("temp min is required")

        try:
            temp_max = self.response.get("TEMP MAX", None)
            if temp_max is None:
                raise Exception("Error")
        except Exception as e : errorMessages.append("temp max is required")

        return errorMessages
    



@app.route("/")
def home():
    return render_template("home.html")

# @app.route("/predict_api", methods = ["POST"])
# def predict_api():
#     data = request.json["data"]
#     print(data)
#     #converting to list
#     data = list(data.values())
#     crop_type = data[0]
#     data = data[1:]
#     data = np.array(data)
#     print(data)
#     # reshaping 
#     data = data.reshape(1, -1)
#     df = pd.DataFrame(data, columns=columns)
#     print(df)

#     crop_type = crop_type.upper()
#     output = 0
#     match crop_type:
#         case "POTATO":
#             transformed_potato = potato_transformer.transform(df)
#             output = potato_model.predict(transformed_potato)[0]
#         case "WHEAT":
#             transformed_wheat = wheat_transformer.transform(df)
#             output = wheat_model.predict(transformed_wheat)[0]
#         case _ :
#             print("Not valid crop")

#     # transformed_df = potato_transformer.transform(df)
#     # #predicting the output
#     # output = potato_model.predict(transformed_df)

#     return jsonify(output)

@app.route("/predict_api", methods = ["POST"])
def predict_api():

    data = request.json["data"]
    print(data)
    _instance = SchemaValidator(response=data)
    response = _instance.isValidated()
    if len(response) > 0:
        _ = {
            "status" : "failed",
            "message" : response
        }
        return _
    else:
        crop_type = data.get("CROP TYPE").upper()
        soil_type = data.get("SOIL TYPE").upper()
        region = data.get("REGION").upper()
        weather_condition = data.get("WEATHER CONDITION").upper()
        temp_min = data.get("TEMP MIN")
        temp_max = data.get("TEMP MAX")

        values = [soil_type, region, weather_condition, temp_min, temp_max]
        values = np.array(values)
        values = values.reshape(1, -1)
        df = pd.DataFrame(values, columns=columns)
        print(df)
        output = 0
        match crop_type:
            case "POTATO":
                transformed_potato = potato_transformer.transform(df)
                output = potato_model.predict(transformed_potato)[0]
            case "WHEAT":
                transformed_wheat = wheat_transformer.transform(df)
                output = wheat_model.predict(transformed_wheat)[0]
            case _ :
                print("invalid crop type")
        
        return jsonify(output)

@app.route("/predict", methods=["POST"])
def predict():
    data = [x for x in request.form.values()]
    print(data)
    crop_type = data[0]
    data = data[1:]
    data = np.array(data)
    #reshaping
    data = data.reshape(1, -1)
    #creating dataframe
    df = pd.DataFrame(data, columns=columns)
    #selecting crop type
    crop_type = crop_type.upper()

    output = 0
    match crop_type:
        case "POTATO":
            transformed_potato = potato_transformer.transform(df)
            output = potato_model.predict(transformed_potato)[0]
        case "WHEAT":
            transformed_wheat = wheat_transformer.transform(df)
            output = wheat_model.predict(transformed_wheat)[0]
        case _ : 
            print("Not Valid Crop")
        
    # transformed_df = potato_transformer.transform(df)
    # #predicting output
    # output = potato_model.predict(transformed_df)[0]
    print(output)

    return render_template("home.html", prediction_text = f"Water Required : {output}")

if __name__ == "__main__":
    app.run(debug=True)