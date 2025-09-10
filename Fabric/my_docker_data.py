#!/usr/bin/env python3

'''
NOTES:
    -- The DNS names used to access the docker container registry server must NOT have a '_' character.
       Docker registry hostnames must only use lowercase letters, digits, '-', and '.'.
'''

yaml_files = [
        'my_docker_devices.yaml',
        'my_docker.yaml'
    ]

docker_obj = my_docker()
docker_obj.import_yaml_files(yaml_files)

docker_obj.update_user_pass_info(user_pass)

exec( docker_obj.get_source_libs('my_docker.env') )

docker_obj.do_variable_substitution()    # This must come after the global variables are defined.

# print(f'--------------------- docker_container_test_response_01')
# pprint(globals()['docker_container_test_response_01'])





