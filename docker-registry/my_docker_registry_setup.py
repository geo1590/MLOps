#!/usr/bin/env python3


'''
PURPOSE
==============
This script will configure the Ubuntu Linux PC or VM with docker container registry.
It uses the TLS HTTPS secure method.


INSTRUCTIONS
==============
-- This script needs user to manually issue the below command and enter the sudo password
   before running this script. You only have to enter your sudo password once and subsequent
   calls to sudo will get this password from cache, but this cache has a expiry period.
   This is needed since this script uses the 'sudo' command.
   Here is an example:
       % sudo pwd

-- Before running this script, must set in the shell the ngrok environment variableis. You 
   can get this info from the ngrok.com site. Here are the commands.
        % export NGROK_AUTHTOKEN=<the-ngrok-token>
        % export NGROK_URL=<your-ngrok-public-url>
		# For example:
		#	export NGROK_AUTHTOKEN='6ilxu66bA6bA5XbBozzWA_qWHRDy2ufGQKkw_2EPe'
		#       export NGROK_URL='candy-car-table.ngrok-free.app'
'''

import subprocess
import time
import re
import os

class cli_cmds():
    process = None
    g_param = {}
    g_param["ngrok_url"] = os.getenv('NGROK_URL')

    def __init__(self):
        if os.getenv('NGROK_URL') == None:
            print(f"ERROR: Must set the 'NGROK_URL' environment variable before running this script.")
            raise('ERROR')

        if os.getenv('NGROK_AUTHTOKEN') == None:
            print(f"ERROR: Must set the 'NGROK_AUTHTOKEN' environment variable before running this script.")
            raise('ERROR')

    def remove_extra_spaces(self, in_str):
        out = ''
        while True:
            out = re.sub('  ', ' ', in_str)
            if out == in_str:
                return out
    
            in_str = out
    
    def do_one_cmd_output(self, command):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
    	    # If you use 'shell=True', then the command does NOT have to be a list, instead it is a string.
        
        cmd_short = self.remove_extra_spaces(command)
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
    

    def do_cli_commands(self, commands):
        if self.process == None:
            # Start a persistent bash session
            self.process = subprocess.Popen(
                ["bash"], 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,  # merge stderr into stdout
                text=True,
                bufsize=1,
            )
    
        for cmd in commands:
            sentinel = f'__END_{int(time.time() * 1000000)}__'
                # This string signals the completion of running the CLI command.
            
            self.process.stdin.write(f"{cmd}\n")
            self.process.stdin.flush()
            sentinel = f"DONE_{int(time.time() * 1000000)}"
            self.process.stdin.write(f"echo {sentinel}\n")
            self.process.stdin.flush()
         
            cmd_short = self.remove_extra_spaces(cmd) 
            print(f'\nCommand: {cmd_short}\n-----------------------') 
    
            # Read output until sentinel shows up
            output = []
            while True:
                line = self.process.stdout.readline().strip()
                if line == sentinel:
                    break
                if line:
                    output.append(line)
    
            print('\n'.join(output))
    
    
    def do_cli_universal(self, commands):
        for pair in commands:
            # print(f'pair: {pair}')
            cmd_str = pair[0]
            cmd_method = pair[1]
            # print(f'do_cli_unversal(): cmd_str: {cmd_str}, cmd_method: {cmd_method}')
    
            if cmd_method == 'use_same_session':
                self.do_cli_commands([cmd_str]) 
            elif cmd_method == 'check_returncode':
                result = self.do_one_cmd_output(cmd_str)
            else:
                print(f'ERROR: unknown cmd_method: {cmd_method}')
                raise('ERROR')
    
    def exit_session(self):
        # (optional) exit the shell cleanly
        self.process.stdin.write("exit\n")
        self.process.stdin.flush()
    
    
    def test_01(self):
        commands = [
            ['echo Hello from shell', 'use_same_session'],
            ['cd /tmp', 'use_same_session'],
            ['pwd', 'use_same_session'],
            ['ls -l', 'use_same_session'],
        ]
    
        self.do_cli_universal(commands)
    
    def delete_running_registry(self):
        command = 'docker ps --format "{{.Image}} {{.ID}} {{.Names}}" | egrep "^registry:"'
        result = self.do_one_cmd_output(command)
       
        print(f'result: {result}')
        output = result['output']
     
        if output == '':
            return
    
        image, id, name = output.split()
        # print(f'(1) id: "{id}"')
        cmd = f'docker rm {id} --force'
        result = self.do_one_cmd_output(cmd)
    
        cmd = f'docker ps -a'
        result = self.do_one_cmd_output(cmd)

def do_install_docker_registry_basic_auth():
    obj = cli_cmds()
    do_configs = True
    do_local_tests = True

    if do_configs:
        try:
            obj.delete_running_registry()
        except:
            True

        commands = [
            ['id', 'use_same_session'],
                ['mkdir -p /registry/certs', 'use_same_session'],
                ['cd /registry/certs', 'use_same_session'],
                ['pwd', 'use_same_session'],
                [f'openssl req \
                 -newkey rsa:4096 -nodes -sha256 -keyout domain.key \
                 -x509 -days 365 -out domain.crt \
                 -subj "/CN={obj.g_param["ngrok_url"]}"', 'use_same_session'],
            ['docker run -d -p 5000:5000 --restart=always --name registry \
                 -v /registry/certs:/certs \
                 -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
                 -e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
                 registry:2', 'use_same_session'],
            ['docker ps -a', 'use_same_session'],
            ['sudo systemctl restart docker', 'use_same_session'],
            ["kill -9 `ps -ef | egrep blessed | egrep -v 'grep' | awk '{print $2}'`", 'use_same_session'],
                [f'nohup ngrok http --domain={obj.g_param["ngrok_url"]} https://localhost:5000 &', 'use_same_session'],
                ['sleep 3', 'use_same_session'],
            ['export NGROK_URL="blessed-kiwi-first.ngrok-free.app"', 'use_same_session'],
            ['echo $NGROK_URL', 'use_same_session'],
        ]

        obj.do_cli_universal(commands)

    if do_local_tests:
        commands = [
            ['curl --cacert /registry/certs/domain.crt \
                 https://blessed-kiwi-first.ngrok-free.app/v2/_catalog | egrep "repositories"', 'check_returncode'],
            ['curl https://blessed-kiwi-first.ngrok-free.app/v2/_catalog | egrep "repositories"', 'check_returncode'],
        ]

        obj.do_cli_universal(commands)

    obj.exit_session()
 
### ----------------------------------------
do_install_docker_registry_basic_auth()
