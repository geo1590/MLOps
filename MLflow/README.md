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

## Screenshots

Here are some screenshots of the MLflow framework.<br>

[Evaluate Dataset] (https://github.com/geo1590/MLOps/blob/main/MLflow/screenshots/web_evaluate_dataset.png)<br>
[Evaluate Function] (https://github.com/geo1590/MLOps/blob/main/MLflow/screenshots/web_evaluate_function.png)<br>
[MinIO Load Artifacts] (https://github.com/geo1590/MLOps/blob/main/MLflow/screenshots/web_minio_load_artifacts.png)<br>
[Open Dataset] (https://github.com/geo1590/MLOps/blob/main/MLflow/screenshots/web_open_dataset.png)<br>
[Docker Show Containers] (https://github.com/geo1590/MLOps/blob/main/MLflow/screenshots/web_show_containers.png)<br>
[Docker Show Images] (https://github.com/geo1590/MLOps/blob/main/MLflow/screenshots/web_show_images.png)<br>

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

Here are the files you will need:<br>
[Dockerfile-MLflow-server](https://github.com/geo1590/MLOps/blob/main/MLflow/Dockerfile-MLflow-server)<br>
[Dockerfile-PostGres](https://github.com/geo1590/MLOps/blob/main/MLflow/Dockerfile-PostGres)<br>
[Dockerfile-fastAPI-MLflow](https://github.com/geo1590/MLOps/blob/main/MLflow/Dockerfile-fastAPI-MLflow)<br>
[Dockerfile-fastAPI-docker](https://github.com/geo1590/MLOps/blob/main/MLflow/Dockerfile-fastAPI-docker)<br>
[Dockerfile-minio-server](https://github.com/geo1590/MLOps/blob/main/MLflow/Dockerfile-minio-server)<br>
[Dockerfile-web-MLflow-docker](https://github.com/geo1590/MLOps/blob/main/MLflow/Dockerfile-web-MLflow-docker)<br>
[docker-compose.yml](https://github.com/geo1590/MLOps/blob/main/MLflow/docker-compose.yml)<br>
[requirements.txt](https://github.com/geo1590/MLOps/blob/main/MLflow/requirements.txt)<br>

 - 'docker run' method
---
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
 ---
 This method uses the Dockerfile, one for each docker container.
 ```bash
my fastapi-docker Dockerfile setup
------------------------------------------------

File: Dockerfile-fastAPI-docker
-----------------------
See link to file in above link.


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
See link to file in above link.


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
See link to file in above link.
			

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

File: Dockerfile-PostGres
-----------------------
See link to file in above link.


Build & Run from Dockerfile
-----------------------
% docker build --no-cache -f Dockerfile-PostGres -t postgres .

% docker run -d \
  --name postgres \
  --network mlflow-network \
  -p 5432:5432 \
  postgres



my web-mlflow-docker Dockerfile setup
------------------------------------------------

File: Dockerfile-web-MLflow-docker
-----------------------
See link to file in above link.


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
See link to file in above link.


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
See link to file in above link.


```

 - docker-compose.yml method
 ---
 ```bash
my MLflow 'docker-compose' docker setup
------------------------------------------------

File: docker-compose.yml
-----------------------
See link to file in above link.
	

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

