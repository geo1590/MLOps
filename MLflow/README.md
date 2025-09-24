# MLOps
Project: MLflow framework<br>
Author: George Barrinuevo<br>
Date: 09/23/2025<br>

## Purpose
The purpose of this repository is to demonstrate how the MLflow AI/ML framework can be implemented in a production environment. Also, I completed an MLOps certification and am building some MLOps personal projects. This is meant to be for educational purposes.

## What is MLflow?

MLflow is an open-source framework for managing AI/ML projects. Here are the tasks it performs:

 - Tracking
 You can log metrics, parameters, artifacts (plots, models, datasets) and code versioning.
 
 - Projects
 Can package AI/ML code with dependencies so the environment can be replicated and re-run on another location.
 
 - Models
 Can package and save models in different formats (sickit-learn, TensorFlow, PyTorch, XGBoost). These models can be versioned.
 
 - Model Registry
 A repository that can save models.
 
 - Deployments
 Can deploy models to different platforms (AWS SageMaker, Azure ML, or custom servers).

## My Implementation

 My implementation of MLflow is an example of how MLops can be developed and is meant for educational purposes. This implementation consists of several docker containers that talk to each other and provide the functionality where MLflow is the main framework. These are the docker containers that can run all inside one host machine.
 

 - MLflow Server
 This docker container runs the main MLflow framework. Other services like PostGres, Web API, FastAPI, and etc will talk to this MLflow frameowk.
 
 - MinIO
 Another docker container is an S3 object storage open source. It connects to MLflow to store artifacts like datasets. The S3 (Simple Storage Service) is an API for object storage. It is similar to Amazon S3. It is a de facto standard for object storage. One advantage of the S3 standard is the ability to swap MinIO S3 to Amazon S3 with minimal code changes.
 
 - PostGres
 PostGres is an SQL database is a relational database and is open source. It is an enterprise grade database that is also open source.
 
 - FastAPI for MLflow
 FastAPI is a web framework which has python APIs. This allows accessing internal services via using HTTP. It is often used to create microservices.
 
 - FastAPI for Docker
 This is similar to the FastAPI above, but is meant for accessing docker services. The reason for splitting the FastAPI in to 2 sections is that this version requires a python environment that is incompatible with the above FastAPI for MLflow.
 - Web API
 This is a Web application that uses gradio for the UI. It allows accessing some of the functionality of MLflow and services connected to it via a user friendly web app.

## My Setup
This implementation can be setup using 3 different methods. It can be set up using the 'docker run' command. It can also be set up using a Dockerfile method. And lastly, it can be set up using the docker-compose.yml file. All of these docker containers is intended to be run under the same host machine. But, can relocate some of them to another host machine. The easiest method is the docker-compose.yml version. The other methods is there for study and educational purposes. All of the files can be found in this repository.

 - Using 'docker run'
 Here is the procedure to set up this implementation using just the 'docker run' command.
 ```bash

my MinIO Server Docker setup
------------------------------------------------
% sudo mkdir /minio-data
% sudo chmod 755 -R /minio-data

% pip install minio
% pip install boto3

% docker run -d  \
  --name minio-server \
  --network mlflow-network \
  -v /minio-data:/data \
  -e "AWS_ACCESS_KEY_ID=minioadmin" \
  -e "AWS_SECRET_ACCESS_KEY=minioadmin" \
  -p 9000:9000 -p 9001:9001-p 9000:9000 -p 9001:9001 \
  quay.io/minio/minio server /data --console-address ":9001"
  
Data is persisted in ./minio-data on your host.
Access web console: http://localhost:9001, http://127.0.0.1:9001
Access S3 API: http://localhost:9000, http://127.0.0.1:9000


my MLflow Server Docker setup
------------------------------------------------

% sudo mkdir /minio-data
% sudo chmod 755 -R /minio-data

% docker run -d \
  --name mlflow-server \
  --network mlflow-network \
  -v /mlflow-mlruns:/mlflow-mlruns \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(which docker):/usr/bin/docker \
  -p 5000:5000 \
  -e BACKEND_STORE_URI=postgresql://mlflow:mlflowpass@postgres-mlflow:5432/mlflowdb \
  -e ARTIFACT_ROOT=/mlflow-mlruns \
  python:3.12 bash -c "\
    apt-get update && \
    apt-get install -y vim-tiny && \
    pip install --no-cache-dir mlflow psycopg2-binary vim-tiny && \
    mlflow server \
      --backend-store-uri \$BACKEND_STORE_URI \
      --default-artifact-root \$ARTIFACT_ROOT \
      --host 0.0.0.0 --port 5000"

# Options:
  -e BACKEND_STORE_URI=sqlite:///mlflow.db
		# Use SQLite. No additional docker container is needed for SQLite.
		
my PostGres (SQL) form MLflow Docker setup
------------------------------------------------
#This is needed if using PostGres within MLflow server.

% docker run -d \
  --name postgres-mlflow \
  --network mlflow-network \
  -e POSTGRES_USER=mlflow \
  -e POSTGRES_PASSWORD=mlflowpass \
  -e POSTGRES_DB=mlflowdb \
  -p 5432:5432 \
  postgres:15


my fastapi-mlflow docker setup
------------------------------------------------
# Both fastapi-mlflow along and fastapi-docker docker containers are needed since these
# environments needed to run various parts of the script can not exist in one environment.
# So, we have 2 dockers for the 2 environments.

% sudo mkdir /app
% sudo chmod 755 -R /app
% cp my_mlflow.py /app
% cp my_mlflow_fast_api.py /app
% cp my_mlflow_web.py /app

% docker run -d \
  --name fastapi-mlflow \
  --network mlflow-network \
  -w /app \
  -v /app:/app \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(which docker):/usr/bin/docker \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  -e DOCKER_HOST=unix:///var/run/docker.sock \
  -p 8030:8030 \
  python:3.11-slim bash -c "\
    apt-get update && \
    apt-get install -y vim-tiny curl iputils-ping iproute2 && rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
	uvicorn /app/my_mlflow_fast_api:app --host 0.0.0.0 --port 8030"
	  

my fastapi-docker docker setup
------------------------------------------------
# Both fastapi-mlflow along and fastapi-docker docker containers are needed since these
# environments needed to run various parts of the script can not exist in one environment.
# So, we have 2 dockers for the 2 environments.

% sudo mkdir /app
% sudo chmod 755 -R /app
% cp my_mlflow.py /app
% cp my_mlflow_fast_api.py /app
% cp my_mlflow_web.py /app

% docker run -d \
  --name fastapi-docker \
  --network mlflow-network \
  -w /app \
  -v /app:/app \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(which docker):/usr/bin/docker \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  -e DOCKER_HOST=unix:///var/run/docker.sock \
  -p 8020:8020 \
  python:3.11-slim bash -c "\
    apt-get update && \
    apt-get install -y vim-tiny curl iputils-ping iproute2 && rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && RUN pip uninstall docker docker-py docker-pycreds requests urllib3 -y && \
	pip install docker==6.1.3 && \
	uvicorn /app/my_mlflow_fast_api:app --host 0.0.0.0 --port 8020"
	
	
 my web-mlflow-docker docker setup
------------------------------------------------

% sudo mkdir /app
% sudo chmod 755 -R /app
% cp my_mlflow_web.py /app

% docker run -d \
  --name web-mlflow-docker \
  --network mlflow-network \
  -w /app \
  -v /app:/app \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(which docker):/usr/bin/docker \
  -e FASTAPI_MLFLOW=http://fastapi-mlflow:8030 \
  -e FASTAPI_DOCKER=http://fastapi-docker:8020 \
  -p 8045:8045 \
  python:3.11-slim bash -c "\
    apt-get update && \
    apt-get install -y vim-tiny curl iputils-ping iproute2 && rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && RUN pip uninstall docker docker-py docker-pycreds requests urllib3 -y && \
	pip install docker==6.1.3 && \
	./my_mlflow_web.py"

```

 - Dockerfile method
 This method uses the Dockerfile, one for each docker container.
 ```bash
my fastapi-docker Dockerfile setup
------------------------------------------------

File: Dockerfile-fastAPI-docker
-----------------------
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y vim-tiny curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y iputils-ping iproute2
RUN pip install minio boto3
RUN pip install --upgrade pip

RUN pip uninstall docker docker-py docker-pycreds requests urllib3 -y
RUN pip install docker==6.1.3 requests==2.31.0 urllib3==1.26.18
RUN pip list | grep docker

COPY my_mlflow.py /app
COPY my_mlflow_fast_api.py /app

EXPOSE 8020

ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV DOCKER_HOST=unix:///var/run/docker.sock

CMD ["uvicorn", "my_mlflow_fast_api:app", "--host", "0.0.0.0", "--port", "8020"]


Build & Run from Dockerfile
-----------------------
% docker build --no-cache -f Dockerfile-fastAPI-docker -t fastapi-docker .

% docker run -d \
  --name fastapi-docker \
  --network mlflow-network \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(which docker):/usr/bin/docker \
  -p 8020:8020 \
  fastapi-docker
 
 
 
my fastapi-mlflow Dockerfile setup
------------------------------------------------

File: Dockerfile-fastAPI-MLflow
-----------------------
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y vim-tiny curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y iputils-ping iproute2
RUN pip install minio boto3
RUN pip install --upgrade pip

COPY my_mlflow.py /app
COPY my_mlflow_fast_api.py /app

EXPOSE 8030

ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV DOCKER_HOST=unix:///var/run/docker.sock

ENV MINIO_HTTP=http://minio-server:9000
ENV MINIO_ROOT_USER=minioadmin
ENV MINIO_ROOT_PASSWORD=minioadmin

CMD ["uvicorn", "my_mlflow_fast_api:app", "--host", "0.0.0.0", "--port", "8030"]


Build & Run from Dockerfile
-----------------------
% docker build --no-cache -f Dockerfile-fastAPI-MLflow -t fastapi-mlflow .

% docker run -d \
  --name fastapi-mlflow \
  --network mlflow-network \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(which docker):/usr/bin/docker \
  -p 8030:8030 \
  fastapi-mlflow
 


my mlflow-server Dockerfile setup
------------------------------------------------

File: Dockerfile-MLflow-server
-----------------------
FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y vim-tiny && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y iputils-ping iproute2
RUN pip install mlflow psycopg2-binary
RUN pip install boto3
RUN pip install --upgrade pip

EXPOSE 5000

ENV BACKEND_STORE_URI=postgresql://mlflow:mlflowpass@postgres-mlflow:5432/mlflowdb
ENV ARTIFACT_ROOT=/mlflow/mlruns
ENV DEFAULT-ARTIFACT-ROOT=s3://mlflow-artifacts

ENV MLFLOW_S3_ENDPOINT_URL=http://minio-server:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin

ENTRYPOINT ["mlflow", "server", "--backend-store-uri", "$BACKEND_STORE_URI", \
            "--default-artifact-root", "$ARTIFACT_ROOT", "--host", "0.0.0.0"]
			

Build & Run from Dockerfile
-----------------------
% docker build --no-cache -f Dockerfile-MLflow -t mlflow-server .

% docker run -d \
  --name mlflow-server \
  --network mlflow-network \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(which docker):/usr/bin/docker \
  -p 5000:5000 \
  mlflow-server
  


my postgres-mlflow Dockerfile setup
------------------------------------------------

File: Dockerfile-web-MLflow-docker
-----------------------
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y vim-tiny curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y iputils-ping iproute2
RUN pip install --upgrade pip

COPY my_mlflow_web.py /app

EXPOSE 8045

ENV FASTAPI_MLFLOW=http://fastapi-mlflow:8030
ENV FASTAPI_DOCKER=http://fastapi-docker:8020

CMD ["./my_mlflow_web.py"]


Build & Run from Dockerfile
-----------------------
% docker build --no-cache -f Dockerfile-web-MLflow-docker -t web-mlflow-docker .

% docker run -d \
  --name web-mlflow-docker \
  --network mlflow-network \
  -p 5432:5432 \
  web-mlflow-docker



my web-mlflow-docker Dockerfile setup
------------------------------------------------

File: Dockerfile-web-MLflow-docker
-----------------------
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y vim-tiny curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y iputils-ping iproute2
RUN pip install --upgrade pip

COPY my_mlflow_web.py /app

EXPOSE 8045

ENV FASTAPI_MLFLOW=http://fastapi-mlflow:8030
ENV FASTAPI_DOCKER=http://fastapi-docker:8020

CMD ["./my_mlflow_web.py"]


Build & Run from Dockerfile
-----------------------
% docker build --no-cache -f Dockerfile-web-MLflow-docker -t web-mlflow-docker .

% docker run -d \
  --name web-mlflow-docker \
  --network mlflow-network \
  -p 8045:8045 \
  web-mlflow-docker



my minio_server Dockerfile setup
------------------------------------------------

File: Dockerfile-minio-server 
-----------------------
FROM alpine:latest

ENV MINIO_ACCESS_KEY=minioadmin
ENV MINIO_SECRET_KEY=minioadmin

RUN apk add --no-cache wget ca-certificates && \
    wget https://dl.min.io/server/minio/release/linux-amd64/minio -O /usr/local/bin/minio && \
    chmod +x /usr/local/bin/minio

RUN mkdir /data

EXPOSE 9000 9001

CMD ["minio", "server", "/data", "--address", ":9000", "--console-address", ":9001"]


Build & Run from Dockerfile
-----------------------
% docker build --no-cache -f Dockerfile-minio-server -t minio_server .

% docker run -d \
  --name minio_server \
  --network mlflow-network \
  -p 9000:9000 -p 9001:9001 \
  minio_server



requirements.txt file
------------------------------------------------
click==8.2.1
datasets==4.0.0
docker_py==1.10.6
fastapi==0.116.2
gradio==5.46.0
mlflow==3.3.2
mlflow_skinny==3.3.2
mlflow_tracing==3.3.2
numpy==2.3.3
pandas==2.3.2
Requests==2.32.5
scikit_learn==1.7.2
transformers==4.56.1
uvicorn==0.35.0
torch==2.8.0
pycurl

```

 - docker-compose.yml method
 ```bash
my MLflow 'docker-compose' docker setup
------------------------------------------------

File: docker-compose.yml
-----------------------
version: '3.9'  # Specify the Compose file version

services:
  web:
    container_name: web-mlflow-docker
    build:
      context: .                                            # Folder containing your Dockerfile
      dockerfile: Dockerfile-web-MLflow-docker              # <-- specify the filename here
    volumes:
      - /volume:/volume
    working_dir: /app
    ports:
      - "8045:8045"
    environment:
      - NOTHING1=123
    networks:
      - my-mlflow-network
    depends_on:
      - my_fastapi_docker
      - my_fastapi_mlflow
      - my_mlflow_server
      - my_postgres
      - my_minio_server

  my_fastapi_docker:
    container_name: fastapi-docker
    build:
      context: .                                            # Folder containing your Dockerfile
      dockerfile: Dockerfile-fastAPI-docker                 # <-- specify the filename here
    volumes:
      - /volume:/volume
    working_dir: /app
    ports:
      - "8020:8020"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /usr/bin/docker:/usr/bin/docker
    environment:
      - NOTHING2=123
    networks:
      - my-mlflow-network

  my_fastapi_mlflow:
    container_name: fastapi-mlflow
    build:
      context: .                                            # Folder containing your Dockerfile
      dockerfile: Dockerfile-fastAPI-MLflow                 # <-- specify the filename here
    volumes:
      - /volume:/volume
    working_dir: /app
    ports:
      - "8030:8030"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /usr/bin/docker:/usr/bin/docker
    environment:
      - NOTHING2=123
    networks:
      - my-mlflow-network

  my_mlflow_server:
    container_name: mlflow-server
    build:
      context: .                                            # Folder containing your Dockerfile
      dockerfile: Dockerfile-MLflow-server                  # <-- specify the filename here
    volumes:
      - /volume:/volume
    working_dir: /app
    ports:
      - "5000:5000"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /usr/bin/docker:/usr/bin/docker
    environment:
      - NOTHING2=123
    networks:
      - my-mlflow-network

  my_postgres:
    container_name: postgres-mlflow
    build:
      context: .                                            # Folder containing your Dockerfile
      dockerfile: Dockerfile-PostGres                       # <-- specify the filename here
    volumes:
      - /volume:/volume
    working_dir: /app
    ports:
      - "5432:5432"
    volumes:
      - /volume:/volume
    environment:
      - NOTHING2=123
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflowpass
      - POSTGRES_DB=mlflowdb
    networks:
      - my-mlflow-network

  my_minio_server:
    container_name: minio-server
    build:
      context: .                                            # Folder containing your Dockerfile
      dockerfile: Dockerfile-minio-server                   # <-- specify the filename here
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_HTTP: http://minio-server:9000
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - /minio-data:/data
    networks:
      - my-mlflow-network

networks:
  my-mlflow-network:
    driver: bridge                                         # This will create a new network.
    name: my-mlflow-network                                # This is the name of the docker network.
	

Build & Run from docker-compose.yml file
-----------------------
% docker image ls
	# Should delete old images. Else, you will have to re-build these images.

% docker ps -al
	# Should delete old container's.

% docker-compose up -d
	# Start all services from the docker-compose.yml file.
	
% docker-compose ps
	# Check running containers
	
% docker-compose logs -f web
	# See logs.

% docker logs <container-ID>
	# See logs.


Access the main web page
-----------------------
# Go to a web browser on the host PC, and use this URL:
#	http://127.0.0.1:8045


Misc commands
-----------------------
# Here are some commands. Use them if you need them.

% docker-compose down
	# Remove the docker-compose containers.



```

## Troubleshooting
This section shows some troubleshooting commands in case something does not work.

```bash
Verify all is working	
-----------------------
% docker network inspect my-mlflow-network
	# Verify you see web-mlflow-docker, fastapi-docker, fastapi-mlflow
% docker ps -a

% docker exec -it <fastapi-app> bash
	# Enter the container.
	# You will need the mlflow-server docker container running.
	
container> ss -tuln
	# Verify you see this:
	#	0.0.0.0:<port-number>
	#	# Verify the port number matches. This is the port number the container is listening to.

container> ping web-mlflow-docker
container> ping fastapi-docker
container> ping fastapi-mlflow
container> ping mlflow-server

# Try the URL out on a local web browser.
# Must use the last '/', or change the FastAPI route from @app.get("/show_containers/") to @app.get("/show_containers").

container> curl http://fastapi-docker:8020/show_containers/
container> curl http://fastapi-mlflow:8030/evaluate_function/
container> curl http://fastapi-docker:8020/show_containers/
container> curl http://fastapi-mlflow:8030/evaluate_function/
container> curl http://mlflow-server:5000/
container> curl http://web-mlflow-docker:8045/
container> curl http://fastapi-mlflow:8045/
container> ./my_mlflow.py --cmd show_containers
container> ./my_mlflow.py --cmd show_images

% http://127.0.0.1:8045/



Docker Network
-----------------------
docker-compose down
	# Stop and remove the containers
% docker network create my-mlflow-network
% docker network connect my-mlflow-network <docker-name>
% docker network ls
% docker network inspect my-mlflow-network
	# Verify you see all of the <docker-name>


docker-compose
-----------------------
docker image ls
	# Should delete old images. Else, you will have to re-build these images.
	
docker-compose up -d
	# Start/update the services
	
docker-compose ps
	# Check running containers
	
docker-compose logs -f web
	# See logs.

docker logs <container-ID>
	# See logs.
	
```

