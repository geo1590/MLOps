from flask import Flask, render_template, request
import logging
import os
from transformers import pipeline, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

from pyngrok import ngrok
import os
import mlflow

# Open an ngrok tunnel to port web_port.
web_port = int(os.environ.get("WEB_PORT"))
print(f'web_port: {web_port}')
# authtoken = "31hJVxly22Ll452wEszMNTTfssf_y31L2BXNBpaPHrYAbGcp"
authtoken = "31hYhuo5WYgo57T6dH7N716k1EE_2BJYnoWToy9mA9PB3MxcP"
    # Sign up for a free account here: https://ngrok.com/signup
    # Create a AuthToken and assign the python variable authtoken above to this value.
ngrok.set_auth_token(authtoken)
public_url = ngrok.connect(web_port)
print("--------------------> Public URL:", public_url)


# Disable symlinks warning and set cache directory
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_CACHE'] = './model_cache'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# save_model_name = '/workspaces/MLOps/app2/saved_model'
save_model_name = './saved_model'

# Global variables
translator = None

try:
    logger.info("Loading translation pipeline...")
    # Using a different, publicly available model
    model_name = "t5-small"  # Small and publicly available
    
    # Create translation pipeline
    translator = pipeline(
        "translation_en_to_de",  # English to German
        model=model_name,
        device=-1  # Force CPU
    )
    logger.info("Translation pipeline loaded successfully!")


    logger.info("Saving the model")
    os.system(f'ls -al .')
    # translator.save_pretrained(save_model_name)

    mlflow.set_tracking_uri('https://f004bc311bc4.ngrok-free.app')
    with mlflow.start_run() as run:
        mlflow.transformers.log_model(
            transformers_model=translator,
            name="my_hf_model",
            task="translation_en_to_de",
            input_example="Hello world!"
        )



    os.system(f'ls -al .')
    os.system(f'ls -al {save_model_name}')

    '''
    translator = ''
    logger.info("Loading the model")
    tokenizer = T5Tokenizer.from_pretrained(save_model_name)
    model = T5ForConditionalGeneration.from_pretrained(save_model_name)
    translator = pipeline("translation_en_to_de", model=model, tokenizer=tokenizer)
    logger.info("Done loading the model")
    '''


except Exception as e:
    logger.error(f"Error during initialization: {str(e)}", exc_info=True)
    raise

def translate_text(text):
    try:
        if not translator:
            return "Translator not initialized"
        result = translator(text, max_length=512)
        return result[0]['translation_text']
    
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return f"Translation failed: {str(e)}"

'''
This is needed for the HTML GUI page.
'''
@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        result = ""
        if request.method == 'POST':
            user_input = request.form.get('user_input', '')
            if user_input:
                result = translate_text(user_input)
        return render_template('index.html', result=result)
    
    except Exception as e:
        logger.error(f"Route error: {str(e)}")
        return f"An error occurred: {str(e)}", 500


'''
This is for accessing via the curl command.
Curl Example:
    curl -X POST https://c324e66dd827.ngrok-free.app/api -H "Content-Type: application/json" -d '{"user_input": "what time is it"}'
        # Must replace the URL to the one ngrok gives you when you run this script.
'''
@app.route('/api', methods=['GET', 'POST'])
def api():
    # logger.info(f'In /api')
    try:
        result = ""
        if request.method == 'POST':
            # logger.info("POST")
            data = request.get_json(silent=True)
            user_input = data.get('user_input', '') if data else ''
            # logger.info(f'user_input: {user_input}')
            if user_input:
                result = translate_text(user_input)
                # logger.info(f'result: {result}')
        return {'results': result}

    except Exception as e:
        logger.error(f"Route error: {str(e)}")
        return f"An error occurred: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
