FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

RUN curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | tee /etc/apt/sources.list.d/ngrok.list \
  && apt update \
  && apt install ngrok

COPY ./app /code/app

ENV WEB_PORT=8000

EXPOSE $WEB_PORT

ENTRYPOINT ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port $WEB_PORT"]

CMD ["sh", "-c", "echo", "A123"]
