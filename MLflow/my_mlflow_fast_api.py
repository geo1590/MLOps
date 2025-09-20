#!/usr/bin/env python3

'''
# Run the Web Server.
% uvicorn my_mlflow_fast_api:app --reload
	# Make sure you have a script 'main.py' in the current directory.
	# Open the http address displayed in a web browser.


'''


'''
my_mlflow.py --cmd show_containers
http://127.0.0.1:8000/show_containers

my_mlflow.py --cmd show_images
http://127.0.0.1:8000/show_images

my_mlflow.py.py --cmd container_rm --name
my_mlflow.py.py --cmd container_rm --name mlflow_model_38592
http://127.0.0.1:8000/container_rm/?name=mlflow_model_38592

my_mlflow.py --cmd image_rm --name <name> --version <version>
my_mlflow.py --cmd image_rm --name huggingface_model --version latest
http://127.0.0.1:8000/image_rm/?name=mlflow_model_38592&version=latest

my_mlflow.py --cmd save_model --name <name>
my_mlflow.py --cmd save_model --name huggingface_model
http://127.0.0.1:8000/save_model/?name=huggingface_model

my_mlflow.py --cmd load_model --base_uri <> --name <name> 
my_mlflow.py --cmd load_model --base_uri 7b00661e141343ed9a437d7f43cfa94c --name huggingface_model
http://127.0.0.1:8000/load_model/?name=huggingface_model&model_base_uri=7b00661e141343ed9a437d7f43cfa94c

my_mlflow.py --cmd register_model --base_uri <> --name <name> --base_uri <>
my_mlflow.py --cmd register_model --base_uri 7b00661e141343ed9a437d7f43cfa94c --name huggingface_model
http://127.0.0.1:8000/register_model/?name=huggingface_model&model_base_uri=7b00661e141343ed9a437d7f43cfa94c

my_mlflow.py --cmd build_docker_image --name <name> --version <>
my_mlflow.py --cmd build_docker_image --name huggingface_model
http://127.0.0.1:8000/build_docker_image/?name=huggingface_model&version=latest

my_mlflow.py --cmd run_docker_image --model_name <> --container_name <>
my_mlflow.py --cmd run_docker_image --model_name huggingface_model --container_name mlflow_model_38592
http://127.0.0.1:8000/run_docker_image/?model_name=huggingface_model&container_name=mlflow_model_38592

my_mlflow.py --cmd stop_container --name <>
my_mlflow.py --cmd stop_container --name mlflow_model_38592
http://127.0.0.1:8000/stop_container/?name=mlflow_model_38592

my_mlflow.py --cmd start_container --name <>
my_mlflow.py --cmd start_container --name mlflow_model_38592
http://127.0.0.1:8000/start_container/?name=mlflow_model_38592

my_mlflow.py --cmd call_model_serve 
my_mlflow.py --cmd call_model_serve
http://127.0.0.1:8000/call_model_serve/

my_mlflow.py --cmd cleanup --model_name <> --container_name <> --version <>
my_mlflow.py --cmd cleanup --model_name huggingface_model --container_name mlflow_model_38592 --version latest
http://127.0.0.1:8000/cleanup/?model_name=huggingface_model&container_name=mlflow_model_38592&version=latest

my_mlflow.py --cmd evaluate_dataset
http://127.0.0.1:8000/evaluate_dataset

my_mlflow.py --cmd evaluate_function
http://127.0.0.1:8000/evaluate_function
'''

from pprint import pprint
from fastapi import FastAPI
import json

import my_mlflow


# Create FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome, this is my_mlflow.py!"}

@app.get("/show_containers/")
def do_show_containers():
    print(f'my_mlflow_fast_api.py(): do_show_containers(): called')
    results = my_mlflow.show_containers()
    # results = json.dumps(results)
    return results

@app.get("/show_images/")
def do_show_images():
    print(f'my_mlflow_fast_api.py(): do_show_images(): called')
    results = my_mlflow.show_images()
    return results

@app.get("/container_rm/")
def do_container_rm(name: str):
    results = my_mlflow.docker_rm(name)
    return results

@app.get("/image_rm/")
def do_image_rm(name: str, version: str):
    results = my_mlflow.docker_image_rm(name, image_version=version)
    return results

@app.get("/save_model/")
def do_save_model(name: str):
    results = my_mlflow.save_model(name)
    return results

@app.get("/load_model/")
def do_save_model(model_base_uri: str, name: str):
    results = my_mlflow.load_model(model_base_uri, name)
    return results

@app.get("/register_model/")
def do_register_model(model_base_uri: str, name: str):
    results = my_mlflow.register_model(model_base_uri, name)
    return results

@app.get("/build_docker_image/")
def do_build_docker_image(name: str, version: str):
    results = my_mlflow.deploy_build_docker_image(name, model_version=version)
    return results

@app.get("/run_docker_image/")
def do_run_docker_image(model_name: str, container_name: str):
    results = my_mlflow.run_docker_image(model_name, container_name)
    return results

@app.get("/stop_container/")
def do_stop_container(name: str):
    results = my_mlflow.docker_stop(name)
    return results

@app.get("/start_container/")
def do_start_container(name: str):
    results = my_mlflow.docker_start(name)
    return results

@app.get("/call_model_serve/")
def do_call_model_serve():
    results = my_mlflow.call_model_serve()
    return results

@app.get("/cleanup/")
def do_cleanup(model_name: str, container_name: str, version: str):
    results = my_mlflow.cleanup(model_name, container_name, version)
    return results

@app.get("/evaluate_function/")
def get_data1():
    results = my_mlflow.evaluate_function()
    return results

@app.get("/evaluate_dataset/")
def get_data2():
    results = my_mlflow.evaluate_dataset()
    return results







