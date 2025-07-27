
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Prophet API is running!'

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    
    past_time = data['past_time']
    past_values = data['past_values']
    
    df = pd.DataFrame({'ds': past_time, 'y': past_values})
    
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=5, freq='D')
    forecast = model.predict(future)

    return jsonify(forecast[['ds', 'yhat']].tail(5).to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
