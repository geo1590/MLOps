#!/usr/bin/env python3

'''
'''

from fabric import Connection
from invoke import Responder
from pprint import pprint
import re
import random
import my_docker_lib

docker_obj = my_docker_lib.my_docker()

# Import libs and global variables in to the current namepace scope.
exec( docker_obj.get_source_libs('my_docker_lib.py') )
exec( docker_obj.get_source_libs('my_docker_data.py') )

# -----------------
print(f"="*50)

docker_obj.process_cmds('uninstall_docker_service')
docker_obj.process_cmds('install_docker_service')

docker_obj.process_cmds('server_install_docker_registry')

docker_obj.process_cmds('client_install_docker_registry')
docker_obj.process_cmds('client_test_docker_ip_install')
docker_obj.process_cmds('client_test_docker_dns_install')
docker_obj.process_cmds('client_docker_delete_all_images')





