from fastapi import FastAPI
import joblib
import numpy as np

from pyngrok import ngrok
import os

# Open an ngrok tunnel to port web_port.
web_port = int(os.environ.get("WEB_PORT"))
print(f'web_port: {web_port}')
# authtoken = "31hJVxly22Ll452wEszMNTTfssf_y31L2BXNBpaPHrYAbGcp"
authtoken = "31hYhuo5WYgo57T6dH7N716k1EE_2BJYnoWToy9mA9PB3MxcP"
    # Sign up for a free account here: https://ngrok.com/signup
    # Create a AuthToken and assign the python variable authtoken above to this value.
ngrok.set_auth_token(authtoken)
public_url = ngrok.connect(web_port)
print("Public URL:", public_url)

model = joblib.load('app/model.joblib')

class_names = np.array(['setosa', 'versicolor', 'virginica'])

app = FastAPI()

@app.get('/')
def read_root():
    # return {'message': f'Iris model API, public_url: {public_url}'}
    return {'message': f'Iris model API'}

@app.post('/predict')
def predict(data: dict):
    """
    Predicts the class of a given set of features.

    Args:
        data (dict): A dictionary containing the features to predict.
        e.g. {"features": [1, 2, 3, 4]}

    Returns:
        dict: A dictionary containing the predicted class.
    """        
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted_class': class_name}


