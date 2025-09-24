#!/usr/bin/env python3

'''
my_mlflow.py.py --cmd container_rm --name
my_mlflow.py.py --cmd container_rm --name mlflow_model_38592
http://127.0.0.1:8020/container_rm/?name=mlflow_model_38592

my_mlflow.py --cmd image_rm --name <name> --version <version>
my_mlflow.py --cmd image_rm --name huggingface_model --version latest
http://127.0.0.1:8020/image_rm/?name=mlflow_model_38592&version=latest

my_mlflow.py --cmd save_model --name <name>
my_mlflow.py --cmd save_model --name huggingface_model
http://127.0.0.1:8030/save_model/?name=huggingface_model

my_mlflow.py --cmd load_model --base_uri <> --name <name>
my_mlflow.py --cmd load_model --base_uri 7b00661e141343ed9a437d7f43cfa94c --name huggingface_model
http://127.0.0.1:8030/load_model/?name=huggingface_model&model_base_uri=7b00661e141343ed9a437d7f43cfa94c

my_mlflow.py --cmd register_model --base_uri <> --name <name> --base_uri <>
my_mlflow.py --cmd register_model --base_uri 7b00661e141343ed9a437d7f43cfa94c --name huggingface_model
http://127.0.0.1:8030/register_model/?name=huggingface_model&model_base_uri=7b00661e141343ed9a437d7f43cfa94c

my_mlflow.py --cmd build_docker_image --name <name> --version <>
my_mlflow.py --cmd build_docker_image --name huggingface_model
http://127.0.0.1:8030/build_docker_image/?name=huggingface_model&version=latest

my_mlflow.py --cmd run_docker_image --model_name <> --container_name <>
my_mlflow.py --cmd run_docker_image --model_name huggingface_model --container_name mlflow_model_38592
http://127.0.0.1:8020/run_docker_image/?model_name=huggingface_model&container_name=mlflow_model_38592

my_mlflow.py --cmd stop_container --name <>
my_mlflow.py --cmd stop_container --name mlflow_model_38592
http://127.0.0.1:8020/stop_container/?name=mlflow_model_38592

my_mlflow.py --cmd start_container --name <>
my_mlflow.py --cmd start_container --name mlflow_model_38592
http://127.0.0.1:8020/start_container/?name=mlflow_model_38592

my_mlflow.py --cmd call_model_serve
my_mlflow.py --cmd call_model_serve
http://127.0.0.1:8030/call_model_serve/

my_mlflow.py --cmd cleanup --model_name <> --container_name <> --version <>
my_mlflow.py --cmd cleanup --model_name huggingface_model --container_name mlflow_model_38592 --version latest
http://127.0.0.1:8020/cleanup/?model_name=huggingface_model&container_name=mlflow_model_38592&version=latest

my_mlflow.py --cmd evaluate_dataset
http://127.0.0.1:8030/evaluate_dataset/

my_mlflow.py --cmd evaluate_function
http://127.0.0.1:8030/evaluate_function/

my_mlflow.py --cmd save_dataset
http://127.0.0.1:8030/save_dataset

my_mlflow.py --cmd open_dataset
http://127.0.0.1:8030/open_dataset

my_mlflow.py --cmd minio_save_artifacts --experiment_name experiment_1324
http://127.0.0.1:8030/minio_save_artifacts/?experiment_name=experiment_1324

my_mlflow.py --cmd minio_load_artifacts --experiment_name experiment_1324
http://127.0.0.1:8030/minio_load_artifacts/?experiment_name=experiment_1324
'''

from pprint import pprint
import mlflow
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import docker
import requests
import json
import time
import click
import os
import pickle
import random
import traceback
import re
from datetime import datetime, timezone
from minio import Minio
from minio.error import S3Error
import tempfile

@click.command()
@click.option('--cmd', help='Command')
@click.option('--name', type=str, default='', help='name')
@click.option('--version', type=str, default='latest', help='version')
@click.option('--base_uri', type=str, default='', help='base URI')
@click.option('--model_name', type=str, default='', help='model name')
@click.option('--container_name', type=str, default='', help='container name')
@click.option('--experiment_name', type=str, default='', help='experiment name')
def script_args(cmd, name, version, base_uri, model_name, container_name, experiment_name):
    pass
    # click.echo(f"Hello, {name}!")
    if cmd == 'stop_container':
        docker_stop(name)
    elif cmd == 'start_container':
        docker_start(name)
    elif cmd == 'call_model_serve':
        call_model_serve()
    elif cmd == 'container_rm':
        docker_rm(name)
    elif cmd == 'image_rm':
        docker_image_rm(name, image_version=version)
    elif cmd == 'save_model':
        model_base_uri, model_name = save_model(name)
        # print(f'model_base_uri: {model_base_uri}')
        # print(f'model_name: {model_name}')
    elif cmd == 'load_model':
        load_model(base_uri, name)
    elif cmd == 'register_model':
        register_model(base_uri, name)
    elif cmd == 'build_docker_image':
        deploy_build_docker_image(name, version)
    elif cmd == 'run_docker_image':
        run_docker_image(model_name, container_name)
    elif cmd == 'cleanup':
        cleanup(model_name, container_name, version)
    elif cmd == 'evaluate_dataset':
        evaluate_dataset()
    elif cmd == 'evaluate_function':
        evaluate_function()
    elif cmd == 'show_containers':
        show_containers()
    elif cmd == 'show_images':
        show_images()
    elif cmd == 'save_dataset':
        save_dataset()
    elif cmd == 'open_dataset':
        open_dataset()
    elif cmd == 'minio_save_artifacts':
        minio_save_artifacts(experiment_name)
    elif cmd == 'minio_load_artifacts':
        minio_load_artifacts(experiment_name)
    else:
        print(f'ERROR: Unknown --cmd')
        raise('ERROR')
    
def init():
    global minio_http
    global minio_root_user
    global minio_root_password
    global mlflow_tracking_uri
    global mlflow_experiment_name

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow_experiment_name = 'my_test_experiment'
    mlflow.set_experiment(mlflow_experiment_name)

    # minio_http = os.getenv("MINIO_HTTP", "http://127.0.0.1:9000")
    minio_http = os.getenv("MINIO_HTTP", "http://localhost:9000")
    minio_http = re.sub(r'https?://', '', minio_http)
    minio_root_user = os.getenv("MINIO_ROOT_USER", "minioadmin")
    minio_root_password = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")

    print(f"minio_http: '{minio_http}'")
    print(f"minio_root_user: '{minio_root_user}'")
    print(f"minio_root_password: '{minio_root_password}'")

def predict_fn(input_df):
    global clf

    texts = input_df["text"].tolist()
    preds = clf(texts, truncation=True)
    # print(f'------------ ::predict_fn(): preds: {preds}')
    # return [1 if p["label"] == "LABEL_1" else 0 for p in preds]
    # 1: positive, 0: negative
    return [1 if p["label"] == "POSITIVE" else 0 for p in preds]

class HFTextClassifier(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline_model):
        self.pipeline_model = pipeline_model

    def predict(self, context, model_input):
        texts = model_input["text"].tolist()
        preds = self.pipeline_model(texts, truncation=True)
        # print(f'------------ ::predict(): preds: {preds}')
        # 1: positive, 0: negative
        return [1 if p["label"] == "POSITIVE" else 0 for p in preds]

def predict_fn2(input_df):
    global clf

    texts = input_df["text"].tolist()
    preds = clf(texts, truncation=True)
    # print(f'------------ ::predict_fn(): preds: {preds}')
    # 1: positive, 0: negative
    positive_threshold = 0.85
    return [1 if p['label'] == 'POSITIVE' and p['score'] > positive_threshold else 0 for p in preds]

def human_elapsed(created_dt):
    now = datetime.now(timezone.utc)
    delta = now - created_dt
    days = delta.days
    seconds = delta.seconds
    if days > 0:
        return f"{days} day{'s' if days>1 else ''} ago"
    hours = seconds // 3600
    if hours > 0:
        return f"{hours} hour{'s' if hours>1 else ''} ago"
    minutes = (seconds % 3600) // 60
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes>1 else ''} ago"
    return f"{seconds} seconds ago"

def human_size(num_bytes):
    """Convert bytes to human-readable string like Docker CLI."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0 or unit == 'TB':
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def show_containers():
    # print(f'show_containers(): called')
    client = docker.from_env()

    containers = client.containers.list(all=True)

    text1 = 'Short ID'
    text2 = 'Name'
    text3 = 'Status'
    border1 = '-'*len(text1)
    border2 = '-'*len(text2)
    border3 = '-'*len(text3)
    t_all_str = f'{text1:25s} {text2:25s} {text3:20s}'
    t_all_str += '\n' + f"{border1:25s} {border2:25s} {border3:20s}"
    for container in containers:
        # print(f'dir(): {dir(container)}')
        t_str = f"{container.short_id:25s} {container.name:25s} {container.status:20s}"
        # print(f't_str: {t_str}')
        t_all_str += '\n' + t_str

    # print(f't_all_str:\n{t_all_str}')
    t_dict = {'output': t_all_str}
    return t_dict

def show_images():
    client = docker.from_env()

    images = client.images.list()

    text1 = 'Image ID'
    text2 = 'Repository:Version-Tag'
    text3 = 'Created'
    text4 = 'Size'
    border1 = '-'*len(text1)
    border2 = '-'*len(text2)
    border3 = '-'*len(text3)
    border4 = '-'*len(text4)
    t_all_str = f'{text1:15s} {text2:70s} {text3:20s} {text4:25s}'
    t_all_str += '\n' + f"{border1:15s} {border2:70s} {border3:20s} {border4:25s}"
    for image in images:
        # print(f'dir(): {dir(image)}')
        # print(f'image.tags: {image.tags}')
        tags = image.tags or ["None:None"]
        # print(f'tags: {tags}')
        image_id = image.short_id.replace('sha256:', '')
        created_iso = image.attrs['Created']  # e.g. '2023-09-17T12:34:56.789Z'
        created_dt = datetime.fromisoformat(created_iso.replace('Z', '+00:00'))
        created_elapsed = human_elapsed(created_dt)
        size_bytes = image.attrs['Size']   # or img['Size'] from low-level API
        size_human = human_size(size_bytes)
        t_str = f"{image_id:15s} {', '.join(tags):70s} {created_elapsed:20s} {size_human:20s}"
        # print(f't_str: {t_str}')
        t_all_str += '\n' + t_str

    # print(f't_all_str:\n{t_all_str}')
    t_dict = {'output': t_all_str}
    return t_dict

def save_model(model_name):
    out_dict = {}
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))  # smaller sample
    test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
    
    tokenizer_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(tokenizer_model_name, num_labels=2)

    # This must be defined after the 'tokenizer' variable has been defined.
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        eval_strategy="epoch"  # works in recent versions
    )
    
    with mlflow.start_run() as run:
        mlflow.log_param("tokenizer_model_name", tokenizer_model_name)
        mlflow.log_param("num_train_epochs", 1)
        mlflow.log_param("train_batch_size", 8)
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
    
        trainer.train()
    
        metrics = trainer.evaluate()
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
        clf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
        mlflow.transformers.log_model(
            transformers_model=clf_pipeline,
            name=model_name, 
            tokenizer=tokenizer)
    
        print(metrics)
        model_base_uri = run.info.run_id
        model_uri = f"runs:/{run.info.run_id}/{model_name}"
        print(f'model_base_uri: {model_base_uri}')
        print(f'model_name: {model_name}')
        print(f'model_uri: {model_uri}')

        out_dict['status'] = 0
        out_dict['model_base_uri'] = model_base_uri
        out_dict['model_name'] = model_name
        return out_dict

def load_model(model_base_uri, model_name):
    out_dict = {}
    out_dict['review_result'] = []

    model_uri = f"runs:/{model_base_uri}/{model_name}"
    print(f'model_uri: {model_uri}')

    loaded_model = mlflow.pyfunc.load_model(model_uri)
    movie_reviews = [
		['This is the best movie ever.'],
		['This is the worst movie ever.'],
	]
    preds = loaded_model.predict(movie_reviews)
    # print(f'preds: {preds}')

    idx = -1
    for row in preds.itertuples():
        idx += 1
        review = movie_reviews[idx]
        pos_or_neg = row.label
        if pos_or_neg == 'LABEL_1':
            pos_or_neg = 'positive'
        else:
            pos_or_neg = 'negative'

        score = row.score
      
        
        t_str = f'Review: {review}, result: {pos_or_neg}, score: {score}' 
        out_dict['review_result'].append(t_str) 
        # print(t_str)

    out_dict['status'] = 0
    return out_dict

def register_model(model_base_uri, model_name):
    out_dict = {}
    model_uri = f'runs:/{model_base_uri}/{model_name}'
    # print(f"model_uri: '{model_uri}'")
    out_dict['model_base_uri'] = model_base_uri
    out_dict['model_name'] = model_name
    results = mlflow.register_model(model_uri=model_uri, name=model_name)
    out_dict['results'] = results
    out_dict['status'] = 0
    # print('out_dict:')
    # pprint(out_dict)
    return out_dict

def deploy_build_docker_image(model_name, model_version='latest'):
    out_dict = {}
    results = mlflow.models.build_docker(
        model_uri=f"models:/{model_name}/{model_version}",
        name=model_name,
        env_manager="conda",  # or "conda", "virtualenv", "local"
        # mlflow_home="/path/to/mlflow",  # optional
        install_mlflow=True  # install MLflow in container
    )

    out_dict['results'] = results
    out_dict['status'] = 0
    # print(out_dict)
    return out_dict

def run_docker_image(model_name, container_name):
    out_dict = {}
    client = docker.from_env()
    
    for container in client.containers.list(all=True):
        print(container.name, container.status)

    container = client.containers.run(
        model_name,        # image name
        detach=True,                    # run in background (like -d)
        name=container_name,       # container name
        ports={'8080/tcp': 5100}        # container_port: host_port
    )
    
    print(f"Started container: {container.name} (ID: {container.short_id})")

    container = None
    client = None

    out_dict['status'] = 0
    # print(out_dict)
    return out_dict

def docker_start(name):
    out_dict = {}
    client = docker.from_env()

    try:
        container = client.containers.get(name)
    except:
        print(f"Could not find the container name '{name}'.")
        out_dict['status'] = 1
        # print(out_dict)

    container.start()

    out_dict['status'] = 0
    # print(out_dict)
    return out_dict

def docker_stop(name):
    out_dict = {}
    client = docker.from_env()
    
    try:
        container = client.containers.get(name)
    except:
        print(f"Could not find the container name '{name}'.")
        out_dict['status'] = 1
        # print(out_dict)
        return out_dict
    
    container.stop()

    out_dict['status'] = 0
    # print(out_dict)
    return out_dict

def docker_rm(name):
    out_dict = {}
    client = docker.from_env()
    
    try:
        container = client.containers.get(name)
    except:
        pass
        print(f"Could not find the container name '{name}'.")
        out_dict['status'] = 1
        return out_dict
    
    container.remove(force=True)

    out_dict['status'] = 0
    return out_dict

def docker_image_rm(image_name, image_version='latest'):
    out_dict = {}
    client = docker.from_env()

    name = f'{image_name}:{image_version}'
    try:
        images = client.images.list(name=name)
    except:
        pass
        print(f"Could not find the container name '{name}'.")
        out_dict['status'] = 1
        return out_dict

    if len(images) != 1:
        print(f"The images '{images}' should have 1 image only.")
        out_dict['status'] = 1
        return out_dict

    for img in images:
        print(f"Removing image: {img.tags}")
        client.images.remove(image=img.id, force=True)  # equivalent to docker image rm

    out_dict['status'] = 0
    return out_dict

def call_model_serve():
    out_dict = {}

    url = "http://localhost:5100/invocations"
    data = {
        "inputs": [
            ["This is the best movie ever."],
            ["This is the worst movie ever."]
        ]
    }
    
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
    
    # print("Status Code:", response.status_code)
    # print("Response Body:", response.text)
    t_dict = json.loads(response.text)
    out_dict['response'] = t_dict
    t1 = t_dict['predictions']
    t_list = []
    for one_dict in t1:
        if one_dict['label'] == 'LABEL_1':
            one_dict['label'] = 'positive'
        elif one_dict['label'] == 'LABEL_0':
            one_dict['label'] = 'negative'

    if response.status_code == 200 and \
       t1[0]['label'] == 'positive' and \
       t1[1]['label'] == 'negative':
       # print(f'Passed')
       out_dict['status'] = 0
       return out_dict
    else:
       # print(f'Failed')
       out_dict['status'] = 1
       return out_dict

def cleanup(model_name, container_name, version):
   docker_rm(container_name)
   # print(f'-----------------------')
   docker_image_rm(model_name, image_version=version)
   return {'status': 0}

def evaluate_dataset():
    global clf

    dataset = load_dataset("imdb", split="test")

    # Sample 100 positive and 100 negative examples
    df_pos = pd.DataFrame(dataset.filter(lambda x: x["label"] == 1)[:100])
    df_neg = pd.DataFrame(dataset.filter(lambda x: x["label"] == 0)[:100])

    # Combine and shuffle
    df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Label distribution:\n", df["label"].value_counts())

    clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


    pyfunc_model = HFTextClassifier(clf)

    with mlflow.start_run():
        eval_data = mlflow.data.from_pandas(df, name="imdb_eval_subset")
        mlflow.log_input(eval_data, context="evaluation")

        # t1=df.head()
        # print(t1)

        mlflow.log_params(
            {
                "batch_size": len(df),
                "batch_date": df.get("text", "unknown").iloc[0]
                if len(df) > 0
                else "unknown",
                "data_source": 'imdb',
            }
        )

        results = mlflow.evaluate(
            model=predict_fn,
            data=df,
            targets="label",
            model_type="classifier",
            evaluators=["default"]
        )

        out_dict = {}
        out_dict['evaluation_metrics'] = {}
        print("\nEvaluation metrics:")
        for metric, value in results.metrics.items():
            out_dict['evaluation_metrics'][metric] = f'{value}'
            print(f"{metric}: {value}")

        active_run = mlflow.active_run()
        run_ui = f'{mlflow.get_tracking_uri()}/#/experiments/{active_run.info.experiment_id}/runs/{active_run.info.run_id}'
        out_dict['run_ui'] = run_ui
        print(f"\nRun UI: {run_ui}")

    out_dict['status'] = 0
    return out_dict

def evaluate_function():
    global clf

    dataset = load_dataset("imdb", split="test")

    # Sample 100 positive and 100 negative examples
    df_pos = pd.DataFrame(dataset.filter(lambda x: x["label"] == 1)[:100])
    df_neg = pd.DataFrame(dataset.filter(lambda x: x["label"] == 0)[:100])

    # Combine and shuffle
    df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Label distribution:\n", df["label"].value_counts())

    clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    with mlflow.start_run():
        eval_data = mlflow.data.from_pandas(df, name="imdb_eval_subset")
        mlflow.log_input(eval_data, context="evaluation")

        # t1=df.head()
        # print(t1)

        results = mlflow.evaluate(
            model=predict_fn2,
            data=df,
            targets="label",
            model_type="classifier",
            evaluators=["default"]
        )

        out_dict = {}
        out_dict['evaluation_metrics'] = {}
        print("\nEvaluation metrics:")
        for metric, value in results.metrics.items():
            out_dict['evaluation_metrics'][metric] = f'{value}'
            print(f"{metric}: {value}")

        active_run = mlflow.active_run()
        run_ui = f'{mlflow.get_tracking_uri()}/#/experiments/{active_run.info.experiment_id}/runs/{active_run.info.run_id}'
        out_dict['run_ui'] = run_ui
        print(f"\nRun UI: {run_ui}")

    out_dict['status'] = 0
    return out_dict


def save_dataset():
    global minio_http
    global minio_root_user
    global minio_root_password

    out_dict = {}

    dataset = load_dataset("imdb", split="train")  # example dataset
   
    tmp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    dataset.to_parquet(tmp_file.name)

    # print(f"----------------- minio_http: '{minio_http}'")
    # print(f"----------------- minio_root_user: '{minio_root_user}'")
    # print(f"----------------- minio_root_password: '{minio_root_password}'")
    client = Minio(
        minio_http,              # In the format: 'minio-server:9000'. Do NOT use 'http[s]://'.
        access_key=f"{minio_root_user}",
        secret_key=f"{minio_root_password}",
        secure=False
    )
    
    bucket_name = "huggingface-datasets"
    object_name = "imdb/train.parquet"

    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
  
    try:
        client.fput_object(bucket_name, object_name, tmp_file.name)
        print(f"Uploaded HuggingFace dataset to s3://{bucket_name}/{object_name}")
    except S3Error as err:
        print("Error occurred:", err)

    out_dict['status'] = 0
    return out_dict
    
'''
NOTES:
    -- Do NOT rename to 'load_dataset' since there is already a library function with that same name.
'''
def open_dataset():
    global minio_http
    global minio_root_user
    global minio_root_password

    out_dict = {}

    client = Minio(
        minio_http,              # In the format: 'minio-server:9000'. Do NOT use 'http[s]://'.
        access_key=f"{minio_root_user}",
        secret_key=f"{minio_root_password}",
        secure=False
    )

    bucket_name = "huggingface-datasets"
    object_name = "imdb/train.parquet"
    
    tmp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    client.fget_object(bucket_name, object_name, tmp_file.name)
    
    dataset = Dataset.from_parquet(tmp_file.name)
    
    # print(dataset)
    # print(dataset[0])  # example row
    
    out_dict['example_row'] = dataset[0]
    out_dict['status'] = 0
    return out_dict

def minio_save_artifacts(experiment_name):
    out_dict = {}

    try:
        out_dict = minio_save_artifacts_function(experiment_name)
    except:
        print(f'Error calling minio_save_artifacts_function()')
        traceback.print_exc()  # prints full traceback

    if mlflow.active_run():
        mlflow.end_run()
        # print(f'(2) ----------------- mlflow.end_run()')

    out_dict['status'] = 0
    return out_dict

def minio_save_artifacts_function(experiment_name):
    global minio_http
    global minio_root_user
    global minio_root_password
    global mlflow_tracking_uri
    global mlflow_experiment_name

    if mlflow.active_run():
        mlflow.end_run()
        # print(f'----------------- mlflow.end_run()')

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f'http://{minio_http}'
    os.environ["AWS_ACCESS_KEY_ID"] = minio_root_user
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_root_password

    print(f"mlflow_tracking_uri: '{mlflow_tracking_uri}'")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    artifact_bucket = "mlflow-artifacts"
            # This bucket name must already exist.
            # It must be set to 'mlflow-artifacts', which matches DEFAULT-ARTIFACT-ROOT environment variable.

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=f"s3://{artifact_bucket}/mlflow-artifacts/{experiment_name}"
        )
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        print("Location in MinIO: ", experiment.artifact_location)
        experiment_id = experiment.experiment_id
    else:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        print("Location in MinIO: ", experiment.artifact_location)
           # If you see /app/??/, then delete the experiment manually via the MLflow UI or CLI and recreate it.
           # Or create a new experiment with a different name and correct artifact location.
    
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("num_epochs", 5)
    
        mlflow.log_metric("accuracy", 0.92)
        model = {"weights": []}
        for i in range(0, 10):
            number = random.randint(0, 254)
            model['weights'].append(number)

        model_path = "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    
        mlflow.log_artifact(model_path)  # Uploads to MinIO
        # print("Artifacts logged to MinIO successfully!")
   
        if mlflow.active_run():
            mlflow.end_run()
            # print(f'(2) ----------------- mlflow.end_run()')
 
        return model
    return {} 

def minio_load_artifacts(experiment_name):
    out_dict = {}

    try:
        out_dict = minio_load_artifacts_function(experiment_name)
    except:
        print(f'Error calling minio_load_artifacts_function()')
        traceback.print_exc()  # prints full traceback

    if mlflow.active_run():
        mlflow.end_run()
        # print(f'(2) ----------------- mlflow.end_run()')

    out_dict['status'] = 0
    return out_dict

def minio_load_artifacts_function(experiment_name):
    global minio_http
    global minio_root_user
    global minio_root_password
    global mlflow_tracking_uri

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f'http://{minio_http}'
    os.environ["AWS_ACCESS_KEY_ID"] = minio_root_user
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_root_password

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"❌ Experiment '{experiment_name}' not found.")
        return {}

    print(f"Artifact location: {experiment.artifact_location}")
    experiment_id = experiment.experiment_id

    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        print(f"❌ No runs found in experiment '{experiment_name}'.")
        return {}

    run_id = runs.iloc[0]["run_id"]
    print(f"Latest run ID: {run_id}")

    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model.pkl")
        print(f"Downloaded artifact to: {local_path}")

        with open(local_path, "rb") as f:
            model = pickle.load(f)

        print(f"Model loaded: {model}")
        return model

    except Exception as e:
        import traceback
        print("Error downloading artifact:")
        traceback.print_exc()
        return {}
    return {}

init()

# ------------------------------------------
if __name__ == '__main__':
    script_args()




