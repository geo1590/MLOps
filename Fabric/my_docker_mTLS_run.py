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
# print(f"="*50)

# docker_obj.process_main('uninstall_docker_service')
# docker_obj.process_main('install_docker_service')

# docker_obj.process_main('server_install_docker_registry')

# docker_obj.process_main('client_install_docker_registry')
# docker_obj.process_main('client_test_docker_ip_install')

# docker_obj.process_main('client_test_docker_dns_install')
# docker_obj.process_main('client_test_docker_dns_install_icmd')

# docker_obj.process_main('client_docker_delete_all_images')

# docker_obj.process_main('docker_container_test_recipe_02')
# docker_obj.process_main('docker_container_test_recipes')

# ------------------
# docker_obj.process_main('docker_container_test_response_01')
docker_obj.process_main('docker_container_setup_01')

# docker_obj.process_main('client_docker_delete_all_images')
# docker_obj.process_main('server_install_docker_registry')
# docker_obj.process_main('client_install_docker_registry')







