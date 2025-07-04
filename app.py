from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


time_mapping = {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4, 'Late_Night': 5}
day_mapping = {'Monday': 0, 'Tuesday': 1, 'W`ednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday':6}
stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
class_mapping = {'Economy': 0, 'Business': 1}


dummy_columns = ['airline_Air_Asia','airline_Air_India', 'airline_Go_First', 'airline_Indigo', 'airline_SpiceJet',
                 'airline_Vistara', 'source_city_Bangalore', 'source_city_Chennai', 'source_city_Delhi',
                 'source_city_Hyderabad', 'source_city_Kolkata', 'source_city_Mumbai',
                 'destination_city_Bangalore', 'destination_city_Chennai', 'destination_city_Delhi',
                 'destination_city_Hyderabad', 'destination_city_Kolkata', 'destination_city_Mumbai']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form = request.form

        duration = float(form['duration'])
        days_left = int(form['days_left'])
        stops = stops_mapping[form['stops']]
        day = day_mapping[form['day']]
        dep = time_mapping[form['departure_time']]
        arr = time_mapping[form['arrival_time']]
        cls = class_mapping[form['class']]


        encoded = dict.fromkeys(dummy_columns, 0)
        encoded[f'airline_{form["airline"]}'] = 1
        encoded[f'source_city_{form["source_city"]}'] = 1
        encoded[f'destination_city_{form["destination_city"]}'] = 1

        scaled = scaler.transform([[days_left, duration]])[0]

        features = [scaled[0], scaled[1], stops, dep, arr, cls, day] + list(encoded.values())
        prediction = model.predict([features])[0]
        return render_template('index.html', prediction=round(prediction, 2))
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
