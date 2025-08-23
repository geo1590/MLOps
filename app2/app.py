from flask import Flask, render_template, request
import logging
import os
from transformers import pipeline, AutoTokenizer

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

if __name__ == '__main__':
    app.run(debug=True)
