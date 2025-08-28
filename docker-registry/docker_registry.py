#!/usr/bin/env python3


'''
-- This script needs user to manually issue the below command and enter the sudo password
   before running this script. You only have to enter your sudo password once and subsequent
   calls to sudo will get this password from cache, but this cache has a expiry period.
   This is needed since this script uses the 'sudo' command.
   Here is an example:
       % sudo pwd

-- Before running this script, must set the NGROK_URL environment variable (see below).
'''

import subprocess
import time
import re
import os

global g_param
g_param = {}
g_param["ngrok_url"] = os.getenv('NGROK_URL')
	# In the shell, use this command to set this variable. Use your own ngrok public URL.
	#	export NGROK_URL=<your-ngrok-public-url>
	#	export NGROK_URL=export NGROK_URL=candy-car-table.ngrok-free.app

def remove_extra_spaces(in_str):
    out = ''
    while True:
        out = re.sub('  ', ' ', in_str)
        if out == in_str:
            return out

        in_str = out


def do_cli_cmd_output(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
	# If you use 'shell=True', then the command does NOT have to be a list, instead it is a string.
    
    cmd_short = remove_extra_spaces(command)
    print(f'\nCommand: {cmd_short}\n-----------------------')
    print(result.stdout)

    dict_out = {
        'output': result.stdout,
        'returncode': result.returncode,
    }

    print(f'returncode: {dict_out["returncode"]}')

    if dict_out['returncode'] != 0:
        print(f"ERROR: The returncode is \'{dict_out['returncode']}\'.")
        raise('ERROR')

    return dict_out


global process
process = None

def do_cli_commands(commands):
    # global g_process_started
    global process

    if process == None:
        # Start a persistent bash session
        process = subprocess.Popen(
            ["bash"], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # merge stderr into stdout
            text=True,
            bufsize=1
        )

    for cmd in commands:
        sentinel = "__END__"
            # This string signals the completion of running the CLI command.
        
        # Write command + sentinel marker
        process.stdin.write(f"{cmd}; echo {sentinel}\n")
        process.stdin.flush()
      
        cmd_short = remove_extra_spaces(cmd) 
        print(f'\nCommand: {cmd_short}\n-----------------------') 

        # Read output until sentinel shows up
        for line in process.stdout:
            line = line.rstrip()
            if line == sentinel:
                break
            print(line)
   

def do_cli_universal(commands):
    for pair in commands:
        # print(f'pair: {pair}')
        cmd_str = pair[0]
        cmd_method = pair[1]
        # print(f'do_cli_unversal(): cmd_str: {cmd_str}, cmd_method: {cmd_method}')

        if cmd_method == 'use_same_session':
            do_cli_commands([cmd_str]) 
        elif cmd_method == 'check_returncode':
            result = do_cli_cmd_output(cmd_str)
        else:
            print(f'ERROR: unknown cmd_method: {cmd_method}')
            raise('ERROR')


def exit_session():
    global process

    # (optional) exit the shell cleanly
    process.stdin.write("exit\n")
    process.stdin.flush()


def test_01():
    commands = [
        ['echo Hello from shell', 'use_same_session'],
        ['cd /tmp', 'use_same_session'],
        ['pwd', 'use_same_session'],
        ['ls -l', 'use_same_session'],
    ]

    do_cli_universal(commands)


def do_install_docker_registry_basic_auth():
    global g_param

    do_configs = True
    do_local_tests = True

    if do_configs:
        delete_running_registry()
    
        commands = [
            ['id', 'use_same_session'],
    	    ['mkdir -p /registry/certs', 'use_same_session'],
    	    ['cd /registry/certs', 'use_same_session'],
    	    ['pwd', 'use_same_session'],
    	    [f'openssl req \
                 -newkey rsa:4096 -nodes -sha256 -keyout domain.key \
                 -x509 -days 365 -out domain.crt \
                 -subj "/CN={g_param["ngrok_url"]}"', 'use_same_session'],
            ['docker run -d -p 5000:5000 --restart=always --name registry \
                 -v /registry/certs:/certs \
                 -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
                 -e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
                 registry:2', 'use_same_session'],
            ['docker ps -a', 'use_same_session'],
            ['sudo systemctl restart docker', 'use_same_session'],
            ["kill -9 `ps -ef | egrep blessed | egrep -v 'grep' | awk '{print $2}'`", 'use_same_session'],
    	    [f'(nohup ngrok http --domain={g_param["ngrok_url"]} https://localhost:5000 &)', 'use_same_session'],
    	    ['sleep 3', 'use_same_session'],
            ['export NGROK_URL="blessed-kiwi-first.ngrok-free.app"', 'use_same_session'],
            ['echo $NGROK_URL', 'use_same_session'],
        ]
    
        do_cli_universal(commands)

    if do_local_tests:
        commands = [
            ['curl --cacert /registry/certs/domain.crt \
                 https://blessed-kiwi-first.ngrok-free.app/v2/_catalog | egrep "repositories"', 'check_returncode'],
            ['curl https://blessed-kiwi-first.ngrok-free.app/v2/_catalog | egrep "repositories"', 'check_returncode'],
        ]
   
        do_cli_universal(commands)
    

def delete_running_registry():
    command = 'docker ps --format "{{.Image}} {{.ID}} {{.Names}}" | egrep "^registry:"'
    result = do_cli_cmd_output(command)
   
    print(f'result: {result}')
    output = result['output']
 
    if output == '':
        return

    image, id, name = output.split()
    # print(f'(1) id: "{id}"')
    cmd = f'docker rm {id} --force'
    result = do_cli_cmd_output(cmd)

    cmd = f'docker ps -a'
    result = do_cli_cmd_output(cmd)



### ----------------------------------------
# test_01()
do_install_docker_registry_basic_auth()
exit_session()

