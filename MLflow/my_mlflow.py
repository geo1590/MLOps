#!/usr/bin/env python3

'''

'''


'''
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
import mlflow
from datasets import load_dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import docker
import requests
import json
import time
import click

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("my_test_experiment")

@click.command()
@click.option('--cmd', help='Command')
@click.option('--name', type=str, default='', help='name')
@click.option('--version', type=str, default='latest', help='version')
@click.option('--base_uri', type=str, default='', help='base URI')
@click.option('--model_name', type=str, default='', help='model name')
@click.option('--container_name', type=str, default='', help='container name')
def script_args(cmd, name, version, base_uri, model_name, container_name):
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
        print(f'model_base_uri: {model_base_uri}')
        print(f'model_name: {model_name}')
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
    else:
        print(f'ERROR: Unknown --cmd')
        raise('ERROR')
    

# ----------------------------
# Define predict function for MLflow
# ----------------------------
def predict_fn(input_df):
    global clf

    texts = input_df["text"].tolist()
    preds = clf(texts, truncation=True)
    print(f'------------ ::predict_fn(): preds: {preds}')
    # return [1 if p["label"] == "LABEL_1" else 0 for p in preds]
    # 1: positive, 0: negative
    return [1 if p["label"] == "POSITIVE" else 0 for p in preds]

# ----------------------------
# Define MLflow Python model wrapper
# ----------------------------
class HFTextClassifier(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline_model):
        self.pipeline_model = pipeline_model

    def predict(self, context, model_input):
        texts = model_input["text"].tolist()
        preds = self.pipeline_model(texts, truncation=True)
        print(f'------------ ::predict(): preds: {preds}')
        # 1: positive, 0: negative
        return [1 if p["label"] == "POSITIVE" else 0 for p in preds]

def predict_fn2(input_df):
    global clf

    texts = input_df["text"].tolist()
    preds = clf(texts, truncation=True)
    print(f'------------ ::predict_fn(): preds: {preds}')
    # 1: positive, 0: negative
    positive_threshold = 0.85
    return [1 if p['label'] == 'POSITIVE' and p['score'] > positive_threshold else 0 for p in preds]



# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def save_model(model_name):
    out_dict = {}
    # Load dataset
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))  # smaller sample
    test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
    
    # Load tokenizer and model
    tokenizer_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(tokenizer_model_name, num_labels=2)

    # Tokenize dataset
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
    
    # MLflow run
    with mlflow.start_run() as run:
    
        # Log parameters
        mlflow.log_param("tokenizer_model_name", tokenizer_model_name)
        mlflow.log_param("num_train_epochs", 1)
        mlflow.log_param("train_batch_size", 8)
    
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
    
        # Train model
        trainer.train()
    
        # Evaluate model
        metrics = trainer.evaluate()
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
        # Create a pipeline with your trained model + tokenizer
        clf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
        # Save model
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
        print(t_str)

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
    print('out_dict:')
    pprint(out_dict)
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
    print(out_dict)
    return out_dict

'''
This is equivalent to 'docker run'.
'''    
def run_docker_image(model_name, container_name):
    out_dict = {}
    # Connect to Docker (default local socket)
    client = docker.from_env()
    
    # List containers
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
    print(out_dict)
    return out_dict


'''
This is equivalent to 'docker stop'.
'''
def docker_start(name):
    out_dict = {}
    # Initialize the Docker client
    client = docker.from_env()

    # Get a container by name or ID
    try:
        container = client.containers.get(name)
    except:
        print(f"Could not find the container name '{name}'.")
        out_dict['status'] = 1
        print(out_dict)

    # Start the container again (like `docker start <container>`)
    container.start()

    out_dict['status'] = 0
    print(out_dict)
    return out_dict

 
'''
This is equivalent to 'docker stop'.
''' 
def docker_stop(name):
    out_dict = {}
    # Initialize the Docker client
    client = docker.from_env()
    
    # Get a container by name or ID
    try:
        container = client.containers.get(name)
    except:
        print(f"Could not find the container name '{name}'.")
        out_dict['status'] = 1
        print(out_dict)
        return out_dict
    
    # Stop the container (like `docker stop <container>`)
    container.stop()

    out_dict['status'] = 0
    print(out_dict)
    return out_dict


    

'''
This is equivalent to 'docker rm'.
'''
def docker_rm(name):
    out_dict = {}
    # Initialize the Docker client
    client = docker.from_env()
    
    # Get a container by name or ID
    try:
        container = client.containers.get(name)
    except:
        pass
        print(f"Could not find the container name '{name}'.")
        out_dict['status'] = 1
        return out_dict
    
    # Remove the container
    container.remove(force=True)

    out_dict['status'] = 0
    return out_dict




'''
This is equivalent to 'docker image rm'.
'''
def docker_image_rm(image_name, image_version='latest'):
    out_dict = {}
    # Connect to Docker (default local socket)
    client = docker.from_env()

    # Get a container by name or ID
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
        client.images.remove(image=img.id)  # equivalent to docker image rm

    out_dict['status'] = 0
    return out_dict




'''
Status Code: 200
Response Body: {"predictions": [{"label": "LABEL_1", "score": 0.9567124843597412}, {"label": "LABEL_0", "score": 0.9516234397888184}]}
LABEL_1: Positive
LABEL_0: Negative
'''
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
    
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)
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
       print(f'Passed')
       out_dict['status'] = 0
       return out_dict
    else:
       print(f'Failed')
       out_dict['status'] = 1
       return out_dict


def cleanup(model_name, container_name, version):
   docker_rm(container_name)
   print(f'-----------------------')
   docker_image_rm(model_name, image_version=version)
   return {'status': 0}

'''
Given a dataset where the Y come from model training and predictions of X, both X and Y are stored in the dataset.
The dataset evaluation can take this dataset and produce metrics without consulting the model (e.g. predictions from
model).
'''

def evaluate_dataset():
    global clf

    # ----------------------------
    # Load the IMDb dataset
    # ----------------------------
    dataset = load_dataset("imdb", split="test")

    # Sample 100 positive and 100 negative examples
    df_pos = pd.DataFrame(dataset.filter(lambda x: x["label"] == 1)[:100])
    df_neg = pd.DataFrame(dataset.filter(lambda x: x["label"] == 0)[:100])

    # Combine and shuffle
    df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Label distribution:\n", df["label"].value_counts())

    # ----------------------------
    # Create Hugging Face pipeline
    # ----------------------------
    clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


    # Wrap pipeline in MLflow PyFunc model
    pyfunc_model = HFTextClassifier(clf)


    # ----------------------------
    # Start MLflow run and log dataset
    # ----------------------------
    with mlflow.start_run():
        # Log evaluation dataset
        eval_data = mlflow.data.from_pandas(df, name="imdb_eval_subset")
        mlflow.log_input(eval_data, context="evaluation")

        t1=df.head()
        print(t1)

        # ----------------------------
        # Log Params
        # ----------------------------
        # Log batch metadata
        mlflow.log_params(
            {
                "batch_size": len(df),
                "batch_date": df.get("text", "unknown").iloc[0]
                if len(df) > 0
                else "unknown",
                "data_source": 'imdb',
            }
        )


        # ----------------------------
        # Evaluate dataset
        # ----------------------------
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

        # Get run info using the active run
        active_run = mlflow.active_run()
        run_ui = f'{mlflow.get_tracking_uri()}/#/experiments/{active_run.info.experiment_id}/runs/{active_run.info.run_id}'
        out_dict['run_ui'] = run_ui
        print(f"\nRun UI: {run_ui}")

    out_dict['status'] = 0
    return out_dict


def evaluate_function():
    global clf

    # ----------------------------
    # Load the IMDb dataset
    # ----------------------------
    dataset = load_dataset("imdb", split="test")

    # Sample 100 positive and 100 negative examples
    df_pos = pd.DataFrame(dataset.filter(lambda x: x["label"] == 1)[:100])
    df_neg = pd.DataFrame(dataset.filter(lambda x: x["label"] == 0)[:100])

    # Combine and shuffle
    df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Label distribution:\n", df["label"].value_counts())

    # ----------------------------
    # Create Hugging Face pipeline
    # ----------------------------
    clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


    # ----------------------------
    # Start MLflow run and log dataset
    # ----------------------------
    with mlflow.start_run():
        # Log evaluation dataset
        eval_data = mlflow.data.from_pandas(df, name="imdb_eval_subset")
        mlflow.log_input(eval_data, context="evaluation")

        t1=df.head()
        print(t1)

        # ----------------------------
        # Evaluate function
        # ----------------------------
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

        # Get run info using the active run
        active_run = mlflow.active_run()
        run_ui = f'{mlflow.get_tracking_uri()}/#/experiments/{active_run.info.experiment_id}/runs/{active_run.info.run_id}'
        out_dict['run_ui'] = run_ui
        print(f"\nRun UI: {run_ui}")

    out_dict['status'] = 0
    return out_dict


# ------------------------------------------
if __name__ == '__main__':
    script_args()




