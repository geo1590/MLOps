# Docker Registry
Project: Docker Registry server/client install<br>
Author: George Barrinuevo<br>
Date: 09/24/2025<br>

## Docker Registry server/client install
These scripts will install the Docker Registry for server and client. If installing with Basic Auth, then client and server should both be on the same host machine. Else, the mTLS Authentication must be used.

See the newer Fabric & Paramiko version:
- This is the older version. The newer version uses Fabric & Paramiko method. You can see it here:
  - [Fabric & Paramiko](https://github.com/geo1590/MLOps/tree/main/Fabric)
- These are the specific recipes to look at:
  - [my_docker.yaml](https://github.com/geo1590/MLOps/blob/main/Fabric/my_docker.yaml)<br>
    See recipes: server_install_docker_registry and client_install_docker_registry:
