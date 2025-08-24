# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt if you have it
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | tee /etc/apt/sources.list.d/ngrok.list \
  && apt update \
  && apt install ngrok



# Copy the rest of the application code
COPY . .

# Ensure a writable cache directory for Hugging Face models
RUN mkdir -p /data && chmod -R 777 /data

# Set Hugging Face cache environment variables
ENV HF_HOME=/data
ENV TRANSFORMERS_CACHE=/data
ENV HF_DATASETS_CACHE=/data

# Expose the port Hugging Face Spaces expects
ENV WEB_PORT=7860
EXPOSE $WEB_PORT

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=$WEB_PORT
# force Flask to use the right port

# Run the Flask app
CMD ["flask", "run"]
