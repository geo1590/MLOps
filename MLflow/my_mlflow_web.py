#!/usr/bin/env python3


from pprint import pprint, pformat
import gradio as gr
import requests
import json
import time
import os

labels_map = {
    'show_containers': [('None', ''), ('None', ''), ('None', '')],
    'show_images': [('None', ''), ('None', ''), ('None', '')],
    'container_remove': [('container_name', ''), ('None', ''), ('None', '')],    
    'image_remove': [('image_name', ''), ('version', 'latest'), ('None', '')],
    'save_model': [('model_name', ''), ('None', ''), ('None', '')],
    'load_model': [('base_uri', ''), ('model_name', ''), ('None', '')],
    'register_model': [('base_uri', ''), ('model_name', ''), ('None', '')],
    'build_docker_image': [('model_name', ''), ('version', 'latest'), ('None', '')],
    'run_docker_image': [('model_name', ''), ('container_name', ''), ('None', '')],
    'stop_container': [('container_name', ''), ('None', ''), ('None', '')],
    'start_container': [('container_name', ''), ('None', ''), ('None', '')],
    'call_model_serve': [('None', ''), ('None', ''), ('None', '')],
    'cleanup': [('model_name', ''), ('container_name', ''), ('version', 'latest')],
    'evaluate_function': [('None', ''), ('None', ''), ('None', '')],
    'evaluate_dataset': [('None', ''), ('None', ''), ('None', '')],
}

def call_url_for_dict(url):
    response = requests.get(url)   # Send GET request
    print(f'response.content:')
    pprint(response.content)

    t_content = response.content.decode()        # Raw bytes (good for images)
    print(f'(1) type(t_content): {type(t_content)}')
    print(f'(1) t_content:')
    pprint(t_content)

    t_str_dict = pformat(json.loads(t_content))
    return t_str_dict

def call_url_for_text_output(url):
    response = requests.get(url)   # Send GET request
    t_content = response.content.decode()        # Raw bytes (good for images)
    t_content = json.loads(t_content)           # Convert from JSON to python Dictionary
    output = t_content['output']

    out_str = ''
    for line in output.splitlines():
        out_str += '\n' + line

    out_str = out_str.replace('\n', '<br>')
    out_str = '<pre>' + out_str + '</pre>'
    return out_str

def call_url_and_print_dict(url):
    response = requests.get(url)   # Send GET request
    t_content = response.content.decode()        # Raw bytes (good for images)
    t_content = json.loads(t_content)           # Convert from JSON to python Dictionary
    output = pformat(t_content)
    output = '<pre>' + output + '</pre>'
    return output

def update_labels(choice):
    t1 = labels_map[choice]
    visible1 = False if t1[0][0] == 'None' else True
    visible2 = False if t1[1][0] == 'None' else True
    visible3 = False if t1[2][0] == 'None' else True
    return gr.update(label=t1[0][0], value=t1[0][1], visible=visible1), \
           gr.update(label=t1[1][0], value=t1[1][1], visible=visible2), \
           gr.update(label=t1[2][0], value=t1[2][1], visible=visible3)

def process(choice, input1, input2, input3):
    yield f"Running!"
    
    '''
    If running this script outside a docker container, then do this on the host machine:
	% unset FASTAPI_MLFLOW
	% unset FASTAPI_DOCKER
    If running this script inside a docker container, then do this on the host machine (below). These URLs are
    docker container names which can also be used as web URLs.
    	% export FASTAPI_MLFLOW=http://fastapi-mlflow:8030
    	% export FASTAPI_DOCKER=http://fastapi-docker:8020
    ''' 
    
    all_env = os.environ 
    docker_functions = all_env['FASTAPI_DOCKER'] if 'FASTAPI_DOCKER' in all_env else 'http://127.0.0.1:8020'
    mlflow_functions = all_env['FASTAPI_MLFLOW'] if 'FASTAPI_MLFLOW' in all_env else 'http://127.0.0.1:8030'

    if choice == 'show_containers':
        url = f'{docker_functions}/show_containers/'
        text = call_url_for_text_output(url)
        print(f'text:\n{text}')
        yield text
    elif choice == 'show_images':
        url = f'{docker_functions}/show_images/'
        yield call_url_for_text_output(url)
    elif choice == 'container_remove':
        url = f'{docker_functions}/container_rm/?name={input1}'
        yield call_url_for_dict(url)
    elif choice == 'image_remove':
        url = f'{docker_functions}/image_rm/?name={input1}&version={input2}'
        yield call_url_for_dict(url)
    elif choice == 'save_model':
        url = f'{mlflow_functions}/save_model/?name={input1}'
        yield call_url_for_dict(url)
    elif choice == 'load_model':
        url = f'{mlflow_functions}/load_model/?name={input2}&model_base_uri={input1}'
        yield call_url_for_dict(url)
    elif choice == 'register_model':
        url = f'{mlflow_functions}/register_model/?name={input2}&model_base_uri={input1}'
        yield call_url_for_dict(url)
    elif choice == 'build_docker_image':
        url = f'{mlflow_functions}/build_docker_image/?name={input1}&version={input2}'
        yield call_url_for_dict(url)
    elif choice == 'run_docker_image':
        url = f'{docker_functions}/run_docker_image/?model_name={input1}&container_name={input2}'
        yield call_url_for_dict(url)
    elif choice == 'stop_container':
        url = f'{docker_functions}/stop_container/?name={input1}'
        yield call_url_for_dict(url)
    elif choice == 'start_container':
        url = f'{docker_functions}/start_container/?name={input1}'
        yield call_url_for_dict(url)
    elif choice == 'call_model_serve':
        url = f'{mlflow_functions}/call_model_serve/'
        yield call_url_for_dict(url)
    elif choice == 'cleanup':
        url = f'{docker_functions}/cleanup/?model_name={input1}&container_name={input2}&version={input3}'
        yield call_url_for_dict(url)
    elif choice == 'evaluate_dataset':
        url = f'{mlflow_functions}/evaluate_dataset'
        yield call_url_and_print_dict(url)
    elif choice == 'evaluate_function':
        url = f'{mlflow_functions}/evaluate_function'
        yield call_url_and_print_dict(url)
    else:
        yield "Invalid choice"

    return

def main():

    with gr.Blocks() as demo:
        # Markdown title at the very top:
        gr.Markdown("# MLflow & Docker Example")

        with gr.Row():
            dropdown = gr.Dropdown(
                choices=list(labels_map.keys()),
                value="Operation",
                label="Choose Operation"
            )
        with gr.Row():
            t1 = labels_map['container_remove']
            input1 = gr.Textbox(label=t1[0][0], value=t1[0][1])
            input2 = gr.Textbox(label=t1[1][0], value=t1[1][1])
            input3 = gr.Textbox(label=t1[2][0], value=t1[2][1])
        with gr.Row():
            gr.Markdown("### Output")
        with gr.Row():
            output = gr.HTML(
                label="",
                visible=True,
            )
    
        dropdown.change(fn=update_labels, inputs=dropdown, outputs=[input1, input2, input3])
        btn = gr.Button(value='Run')
        btn.click(fn=process, inputs=[dropdown, input1, input2, input3], outputs=output)

 
    demo.launch(share=True, server_name="0.0.0.0", server_port=8045)


# --------------------------------
main()









