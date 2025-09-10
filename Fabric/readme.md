# Python Fabric + Paramiko module for device configurations.
These set of python files uses the Fabric and Paramiko python modules to configure devices like linux VMs via the CLI shell. This implementation adds a YAML file to specify the CLI command and device info organized as recipes (a set of CLI commands) to perform tasks like configure the Docker Container Registry, install Ubuntu or Python files, and etc. 

Some of the example recipes I have include configuring and setting up:
- Docker Container Registry using mTLS security on the server and client side.
- Basic Ubuntu Linux VM setup.
- Various docker tasks.
- Setup MLflow server.

Here is a quick description of what the files do:
- my_docker_devices.yaml
-- Define the devices (e.g. Ubuntu VMs) username and passwords. You must modify this file to use your PC/VMs hostname, usernamd, and password.
- my_docker.env
-- Put custom variables here that can be referenced in the YAML file.
- my_docker.yaml
-- The recipes are contained here. Add/modify recipes in this file.
- my_docker_data.py
-- This will import the recipes in my_docker.yaml and custom variables in my_docker.env. This file will then be sourced by the client script.
- my_docker_lib.py
-- This is the custom library that will use the Fabric and Paramiko Python libraries.
- my_docker_mTLS_run.py
-- This is the client script. Enable the recipe you want to run, then run this script.
