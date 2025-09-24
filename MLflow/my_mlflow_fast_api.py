#!/usr/bin/env python3

'''
# Run the Web Server.
% uvicorn my_mlflow_fast_api:app --reload
	# Make sure you have a script 'main.py' in the current directory.
	# Open the http address displayed in a web browser.
'''


from pprint import pprint
from fastapi import FastAPI
import json
import my_mlflow

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome, this is my_mlflow.py!"}

@app.get("/show_containers/")
def do_show_containers():
    # print(f'my_mlflow_fast_api.py(): do_show_containers(): called')
    results = my_mlflow.show_containers()
    # results = json.dumps(results)
    return results

@app.get("/show_images/")
def do_show_images():
    # print(f'my_mlflow_fast_api.py(): do_show_images(): called')
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
def do_load_model(model_base_uri: str, name: str):
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

@app.get("/save_dataset/")
def do_save_dataset():
    results = my_mlflow.save_dataset()
    return results

@app.get("/open_dataset/")
def do_open_dataset():
    results = my_mlflow.open_dataset()
    return results

@app.get("/minio_save_artifacts/")
def do_minio_save_artifacts(experiment_name: str):
    results = my_mlflow.minio_save_artifacts(experiment_name)
    return results

@app.get("/minio_load_artifacts/")
def do_minio_load_artifacts(experiment_name: str):
    results = my_mlflow.minio_load_artifacts(experiment_name)
    return results
