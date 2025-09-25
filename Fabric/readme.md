# Fabric & Paramiko
Project: CLI commands framework<br>
Author: George Barrinuevo<br>
Date: 09/24/2025<br>

# Python Fabric + Paramiko module for device configurations.
These set of python files uses the Fabric and Paramiko python modules to configure devices like linux VMs via the CLI shell. This implementation adds a YAML file to specify the CLI command and device info organized as recipes (a set of CLI commands) to perform tasks like configure the Docker Container Registry, install Ubuntu or Python files, and etc. 

Screenshots:
- [Install-Fabric](https://github.com/geo1590/MLOps/blob/main/Fabric/screenshots/install%20Fabric.png)

Some of the example recipes I have included used for configuring and setting up services:
- Install/Uninstall Docker Server
- Install Docker Registry Server using mTLS security
- Install Docker Registry Client using mTLS security
- Few recipes to test the Docker connection/funcionality
- Tests using multiple recipes inside one recipe.

Here is a quick description of what the files do:
- my_docker_devices.yaml
  - Define the devices (e.g. Ubuntu VMs) username and passwords. You must modify this file to use your PC/VMs hostname, usernamd, and password.
- my_docker.env
  - Put custom variables here that can be referenced in the YAML file.
- my_docker.yaml
  - The recipes are contained here. Add/modify recipes in this file.
- my_docker_data.py
  - This will import the recipes in my_docker.yaml and custom variables in my_docker.env. This file will then be sourced by the client script.
- my_docker_lib.py
  - This is the custom library that will use the Fabric and Paramiko Python libraries.
- my_docker_mTLS_run.py
  - This is the client script. Enable the recipe you want to run, then run this script.

Some notes of the recipes:
- If using 'icmd:', then this will keep the ssh connection in persistent mode so that it is using the same environment for the entire recipes. And the stdout will display as it available line by line instead of displaying at the end of the command. The 'cmd:' does NOT use persistant mode.
- Can run one recipe that consists of multiple recipes.

