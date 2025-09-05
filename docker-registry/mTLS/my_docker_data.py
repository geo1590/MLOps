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


# Pre-process data.
docker_obj.update_user_pass_info(user_pass)
u1 = user_pass['user1:west-vm2']    # docker client
u2 = user_pass['user1:west-vm1']    # docker container registry server

# Global variables
g_c = {}    # This stores the connection (e.g. ssh) info. The 'g_' means it lives in global scope.
g_id_func = id    # Re-map the id(), since will be using the variable id= as a parameter.
my_rand_str1 = str(random.randint(10000, 99999))    # Used to create random numbers in strings, e.g. 'my-ubuntu-12345'.
my_rand_str2 = str(random.randint(10000, 99999))    # Used to create random numbers in strings, e.g. 'my-ubuntu-12345'.
sudo_response = {'pattern': fr'\[sudo\] password for .*:', 'response': f"{u1['password']}\n", 'name': 'p1'}
new_host_response = {'pattern': fr'Are you sure you want to continue connecting .*\?', 'response': f"yes\n", 'name': 'p1'}
overwrite_response  = {'pattern': fr'Overwrite\? [(]y/N[)]', 'response': f"y\n", 'name': 'p1'}

my_client_path1 = f'/etc/docker/certs.d/{u2["ip"]}:5000'
my_client_path2 = f'/etc/docker/certs.d/{u2["dns"]}:5000'

my_server_server_ext = '''cat > /tmp/server.ext << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = IP:$DOCKER_IP,DNS:$DOCKER_DNS
EOF'''

my_server_config_yml = '''version: 0.1
log:
  level: debug
http:
  addr: :5000
  tls:
    certificate: /certs/server.crt
    key: /certs/server.key
    clientcas:
      - /certs/ca.crt
    clientcert: required
storage:
  filesystem:
    rootdirectory: /var/lib/registry'''

my_server_docker_run = '''docker run -d -p 5000:5000 --restart=always --name registry \
-v /registry/VM/server:/certs \
-v /registry/VM/server/config.yml:/etc/docker/registry/config.yml \
registry:2'''

my_server_dir = '/registry/VM/server'
my_client_dir = '/registry/VM/client'

docker_obj.do_variable_substitution()    # This must come after the global variables are defined.


