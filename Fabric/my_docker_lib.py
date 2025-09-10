#!/usr/bin/env python3

'''
TO DO
===================


INFO
===================
-- Author:
	George Barrinuevo


INSTALLATION
===================
pip install fabric
pip install pyyaml



PURPOSE
===================


REQUIREMENTS
===================

NOTES
===================
'''

from fabric import Connection
from invoke import Responder
from pprint import pprint
from paramiko.client import SSHClient, AutoAddPolicy
import re
import random
import socket
import yaml
import ipaddress
import copy
import time
import os
import stat

class my_result:
    return_code = 0

class my_Connection(Connection):

    def clean_str(self, in_str):
        cleaned = re.sub(r"\x1B\[[0-9;?]*[a-zA-Z]", "", in_str)
        return cleaned

    def get_recv(self, id, responses=[], timeout=60):
        self.general_prompt_regexp = fr'[^@]+@[^:]+:[^\$\#]+[\$\#] '
        ptr = g_c[id]
        channel = ptr['channel']

        time.sleep(1)

        num_lines_printed = 0
        output = ''
        max_secs = timeout
        start_secs = int(time.time())
        while (int(time.time()) - start_secs) < max_secs:

            while not channel.recv_ready():
                time.sleep(0.5)
                continue

            t_out = channel.recv(65535).decode('utf-8', errors='ignore')
            output += t_out

            output_list = output.splitlines()

            num_blank_lines = 0
            for line in output_list:
                line = self.clean_str(line)
                if len(line) == 0:
                    num_blank_lines += 1
                else:
                    break

            if num_lines_printed < num_blank_lines:
                num_lines_printed = num_blank_lines
                # print(f'----------------- set num_lines_printed to {num_lines_printed}')

            for line in output_list:
                m = re.search(self.general_prompt_regexp, output)
                if m:
                    orig_num_lines_printed = num_lines_printed
                    for idx, line in enumerate(output_list[num_lines_printed::]):
                        idx += orig_num_lines_printed
                        # print(f'A: {idx}: \'{line}\', len: {len(line)}')
                        print(f'{line}')
                        num_lines_printed += 1

                    return output

            for one_response in responses:
                pattern = one_response['pattern']
                response = one_response['response']     # The 'response' here is OK.
                if re.search(pattern, t_out):
                    channel.send(response)
                    break

            orig_num_lines_printed = num_lines_printed
            for idx, line in enumerate(output_list[num_lines_printed::]):
                idx += orig_num_lines_printed
                # print(f'B: {idx}: \'{line}\', len: {len(line)}')
                print(f'{line}')
                num_lines_printed += 1

        print(f'ERROR: Can not get the user prompt')
        raise('ERROR')


    def send_icmd(self, id, cmd, raw_output=False, responses=[]):
        # print(f'----------------------- send_icmd(): responses: {responses}')

        ptr = g_c[id]
        channel = ptr['channel']

        channel.send(f'{cmd}')
        time.sleep(0.1)
        discard_out = channel.recv(65535).decode('utf-8')
        channel.send(f'\n')

        output = self.get_recv(id, responses=responses)

        return_code_str = 'return_code_4387261'
        return_code_cmd = f'echo "{return_code_str}=`echo $?`"'
        return_code = 0
        channel.send(return_code_cmd + '\n')
        while not channel.recv_ready():
            time.sleep(0.5)
            continue

        t_out = channel.recv(65535).decode('utf-8')
        # print(f"----------------- send_icmd(): t_out: '{t_out}'")

        m = re.search(fr'{return_code_str}=(\d+)', t_out)
        if m:
            # print(f'send_icmd(): Found return code')
            return_code = int(m.group(1))
            # print(f'send_icmd(): return_code: {return_code}')

        return (return_code, output)


class my_docker():
    
    def __init__(self):
        dummy_var = 1
        self.check_file_permission('my_docker_devices.yaml')

    def test_01(self):
        global g_user_pass
        # pprint(globals())
        pprint(g_user_pass)

    def get_recipe_list(self):
        g_globals = globals()
 
        return(g_globals['my_recipe_list'])

    def import_yaml_files(self, yaml_files):
        g_globals = globals()
    
        g_globals['my_recipe_list'] = []
        for file in yaml_files:
            t_dict = docker_obj.load_yaml(file)
            # pprint(t_dict)
            for recipe in t_dict:
                # print(f'recipe: {recipe}')
                # print('')
                g_globals['my_recipe_list'].append(recipe)
                g_globals[recipe] = t_dict[recipe]
   
    def get_file_permission(self, file):
        mode = os.stat(file).st_mode
    
        t_dict = {}
        t_dict['owner'] = {'read': bool(mode & stat.S_IRUSR), 'write': bool(mode & stat.S_IWUSR), 'execute': bool(mode & stat.S_IXUSR)}
        t_dict['group'] = {'read': bool(mode & stat.S_IRGRP), 'write': bool(mode & stat.S_IWGRP), 'execute': bool(mode & stat.S_IXGRP)}
        t_dict['other'] = {'read': bool(mode & stat.S_IROTH), 'write': bool(mode & stat.S_IWOTH), 'execute': bool(mode & stat.S_IXOTH)}
        return t_dict
    
    def check_file_permission(self, file):
        my_dict = self.get_file_permission(file)
        # pprint(my_dict)
        if my_dict['owner']['read'] == False or \
           my_dict['owner']['write'] == False or \
           my_dict['other']['read'] == True or \
           my_dict['other']['write'] == True or \
           my_dict['other']['execute'] == True or \
           my_dict['group']['read'] == True or \
           my_dict['group']['write'] == True or \
           my_dict['group']['execute'] == True:
           print(f"ERROR: The file permissions of '{file}' did not pass the security test.")
           raise('ERROR')

        # print(f"The file permissions of '{file}' is OK.")
 
    def do_variable_substitution(self):
        g_globals = globals()
    
        for recipe in g_globals['my_recipe_list']:
            # print(f'do_variable_substitution(): recipe: {recipe}')
            # pprint(g_globals[recipe])
            g_globals[recipe] = docker_obj.variable_substitute(g_globals[recipe])
    

    def print_dict(self, dict_name):
        g_globals = globals()

        print(f'{dict_name}\n' + '-'*30)
        pprint(g_globals[dict_name])
        print('')

    def remove_f_string(self, s):
        if s.startswith("f'") or s.startswith('f"'):
            s = s[1:]

        if re.search('^ *"', s):
            s = re.sub('^ *"', '', s)
            s = re.sub('" *$', '', s)

        if re.search("^ *'", s):
            s = re.sub("^ *'", '', s)
            s = re.sub("' *$", '', s)

        return s

    '''
    NOTES
    ==========
    -- {"__builtins__": {}} removes dangerous built-ins (open, os, etc.).
       The globals() will do the variable substitution from that namespace scope.
    '''
    def safe_eval(self, s):
        s = eval(s, {"__builtins__": {}}, globals())
        return s


    '''
    Do NOT user eval() for variable substitution, since it will also do python command substitution. Instead,
    use this function which does NOT do command substitution.
    TO DO
    ---------
    -- This does not work if the value is a dictionary. Fix it so that the one_cmd['connection'] in variable_substitution()
       will work. This is needed for sudo_repsonse variable too.
    -- The input parameter 's' should be in the form ${var1}. If it is a dictionary, then use self.safe_eval(var1).
    '''
    def var_substitute(self, s):
        t_str = s
        # print(f'(1) t_str: {t_str}')
        t_str = self.remove_f_string(t_str)
        # print(f'(2) t_str: {t_str}')

        pattern = r"\${.+?}"
        found_list = re.findall(pattern, t_str)
        # print(f'found_list: {found_list}')

        for found in found_list:
            # print(f'-'*20)
            s = re.sub(r'^ *\$', '', found)
            s = re.sub(r'^ *\{', '', s)
            s = re.sub(r' *\}$', '', s)
            # print(f"(1) s: '{s}'")
            s = self.safe_eval(s)
            # print(f"(2) var_substitute(): s: '{s}'")
            t_str = re.sub(pattern, s, t_str, count=1)
            # print(f'(after re.sub) var_substitute(): t_str: {t_str}')

        return t_str
 
    def load_yaml(self, file):
        # Load YAML file into a Python dictionary
        with open(file, "r") as file:
            data = yaml.safe_load(file)
            return data
  
    '''
    USAGE:
        exec( docker_obj.get_source_libs('my_docker_lib.py') )
    NOTES:
        -- Using 'from my_docker_lib import *' does not work since it places variables in that file in a different
           scope.
    ''' 
    def get_source_libs(self, file):
        fd = open(file)
        text = fd.read()
        # exec(text)
        fd.close()
        return text
    
    
    def variable_substitute(self, in_dict):
        out_dict = copy.deepcopy(in_dict)
        for one in out_dict:
            if 'connection' in one:
                one['connection_dict'] = self.safe_eval(one['connection'])
    
            if 'cmd' in one:
                one['cmd'] = self.var_substitute(one['cmd'])	# Works

            if 'icmd' in one:
                one['icmd'] = self.var_substitute(one['icmd']) 

            if 'cwd' in one:
                one['cwd'] = self.var_substitute(one['cwd']) 

            if 'responses' in one:
                responses = []
                watchers = []
                for one_response in one['responses']:
                    # print(f'(1) one_response: {one_response}')
                    # print(f'sudo_response: {sudo_response}')
                    one_response = self.safe_eval(one_response)
                    # print(f'(2) one_response: {one_response}')
                    # name = one_response['name']
                    pattern = one_response['pattern']
                    response = one_response['response']		# The 'response' here is OK.
                    responder = Responder(pattern=pattern, response=response)
                    watchers.append(responder)
                    responses.append(one_response)

                one['watchers'] = watchers
                one['responses'] = responses
                # del one['response']
                
        return out_dict
    
    
    def remove_extra_spaces(self, in_str):
        out = ''
        while True:
            out = re.sub('  ', ' ', in_str)
            if out == in_str:
                return out
    
            in_str = out

    def get_ip_from_dns(self, hostname):
        try:
            return socket.gethostbyname(hostname)
        except:
            return None
   
    def update_user_pass_info(self, in_dict):
        for one in in_dict:
            t_dict = in_dict[one]
            host = t_dict['host']
            ip = self.get_ip_from_dns(host)
            t_dict['ip'] = ip
     
    def is_public_ip(self, ip: str) -> bool:
        try:
            ip_obj = ipaddress.ip_address(ip)
            return not (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved or ip_obj.is_multicast or ip_obj.is_link_local)
        except ValueError:
            # Invalid IP string
            return False
   
    def run(self, **kwargs):
        # print(f'run(): ------------------- kwargs:')
        # pprint(kwargs)

        if 'cmd' in kwargs: 
            return(self.run_cmd(**kwargs))
        elif 'icmd' in kwargs:
            return(self.run_icmd(**kwargs))
        else:
            print(f"ERROR: neither 'cmd:' or 'icmd:' provided in the recipe.")
            raise('ERROR')

    def run_cmd(self, **kwargs):
        # print(f'run_cmd(): ------------------- kwargs:')
        # pprint(kwargs)
        
        if 'cwd' in kwargs:
            print(f"ERROR: run_cmd(): the cwd='{kwargs['cwd']}' was used, but is NOT allowed here. Move cwd= to my_cmds variable.")
            raise('ERROR')
    
        cmd_str = self.remove_extra_spaces(kwargs['cmd'])
        ptr = g_c[kwargs["id"]]
        if 'cwd' in ptr:
            cwd = ptr['cwd']
            # print(f'run_cmd(): -------------------> cwd: {cwd}')
        else:
            cwd = '/tmp/'
      
        if re.search('&', kwargs['cmd']):
            kwargs['cmd'] = f'cd {cwd} && ({kwargs["cmd"] })'
        else:
            kwargs['cmd'] = f'cd {cwd} && {kwargs["cmd"] }'
    
        connection_obj = ptr['connection_obj'] 
        u_id = kwargs['u_id']
        cmd = kwargs['cmd']
    
        if not 'warn' in kwargs:
            kwargs['warn'] = True
            # The 'warn=True' means to not error out if the .return_code returns a non-zero value.
    
        print(f'\nConnection: {u_id}, Command: \'{cmd_str}\'\n---------------------------------')
        # print(f'Command: {cmd}')
    
        del kwargs['cmd']
        del kwargs['id']
        if 'cwd' in kwargs:
            del kwargs['cwd'] 
        del kwargs['u_id']

        # t_kwargs = copy.deepcopy(kwargs)
        # del t_kwargs['responses']

        del kwargs['responses']
        # print(f'(2) run_cmd(): ------------------- kwargs:')
        # pprint(kwargs)

        result = connection_obj.run(cmd, **kwargs)
        return_code = result.return_code

        return result
   
    def run_icmd(self, **kwargs):
        # print(f'run_cmd(): ------------------- kwargs:')
        # pprint(kwargs)

        if 'cwd' in kwargs:
            print(f"ERROR: run_icmd(): the cwd='{kwargs['cwd']}' was used, but is NOT allowed here. Move cwd= to my_cmds variable.")
            raise('ERROR')

        cmd_str = self.remove_extra_spaces(kwargs['icmd'])
        ptr = g_c[kwargs["id"]]
        if 'cwd' in ptr:
            cwd = ptr['cwd']
            # print(f'run_icmd(): -------------------> cwd: {cwd}')
        else:
            cwd = ''

        icmd = ''
        if cwd != '':
            icmd = f'cd {cwd} && '

        if re.search('&', kwargs['icmd']):
            icmd += f'({kwargs["icmd"]})'
        else:
            icmd += f'{kwargs["icmd"]}'  

        connection_obj = ptr['connection_obj']
        u_id = kwargs['u_id']
        channel = ptr['channel']

        if not 'warn' in kwargs:
            kwargs['warn'] = True
            # The 'warn=True' means to not error out if the .return_code returns a non-zero value.

        print(f'\nConnection: {u_id}, iCommand: \'{cmd_str}\'\n---------------------------------')

        # print(f'------------------ run_icmd(): responses: {kwargs["responses"]}')
        return_code, output = connection_obj.send_icmd(kwargs['id'], icmd, responses=kwargs["responses"])
        # print(f'<--------------------->')
        # print(f'{output}')		# This is needed, do not comment this out.

        result = my_result()
        result.return_code = return_code
        return result

    def close(self):
        global g_c

        for u_name in g_c:
            if 'connection_obj' in g_c[u_name]:
                # print(f'------------------ close(): closing Paramiko session')
                connection_obj = g_c[u_name]['connection_obj']
                # print(f'------------------ close(): A100')
                connection_obj.client.close()
                # print(f'------------------ close(): A110')
                connection_obj.close()
                # print(f'------------------ close(): A120')
                # del g_c[u_name]
                # print(f'------------------ close(): A130')

            g_c = {}
            # print(f'------------------ close(): A140')
                

    def process_main(self, yaml_name):
        global g_c

        g_globals = globals()
        # print(f'---------------------- process_main(), yaml_name: {yaml_name}')
        # print(f'---------------------- process_main(), globals()[{yaml_name}]:')
        # pprint(g_globals[yaml_name])
        if 'recipes' in g_globals[yaml_name]:
            # print(f'---------------------- process_main(), A100')
            for one_recipe in g_globals[yaml_name]['recipes']:
                # print(f'---------------------- process_main(), one_recipe: {one_recipe}')
                self.process_cmds(one_recipe)
                self.close()

            
        else:
            # print(f'---------------------- process_main(), A200')
            self.process_cmds(yaml_name)
 
    def process_cmds(self, yaml_name):
        global g_c

        g_globals = globals()

        print('')
        print(f'='*50)
        print(f'recipe: {yaml_name}')
        print('')
        my_cmds = g_globals[yaml_name]
        # print(f'--------------------- process_cmds(): my_cmds: ')
        # pprint(my_cmds)

        for one_cmd in my_cmds:
            # print(f'one_cmd: {one_cmd}')
            if 'cmd' in one_cmd:
                cmd = one_cmd['cmd']
                # print(f'cmd: {cmd}')
            elif 'icmd' in one_cmd:
                icmd = one_cmd['icmd']
                # print(f'icmd: {icmd}')
            else:
                print(f"ERROR: neither 'cmd:' or 'icmd:' found.")
                raise('ERROR')

            one_cmd['connection_dict'] = docker_obj.variable_substitute(one_cmd['connection_dict'])
            u = one_cmd['connection_dict']
            u_id = f'{u["user"]}@{u["host"]}'
            u_name = one_cmd['connection']
            if 'options' in one_cmd:
                options = one_cmd['options']
            else:
                options = ''
    
            if not u_name in g_c:
                g_c[u_name] = {}
                ptr = g_c[u_name] 
                ptr['id'] = u_name
                my_kwargs = {}
                my_kwargs['host'] = u['host']
                my_kwargs['user'] = u['user']
                if 'password' in u:
                    my_kwargs['connect_kwargs'] = {'password': u['password']}
                elif 'connect_kwargs' in u:
                    my_kwargs['connect_kwargs'] = u['connect_kwargs']
   
                ptr['connection_obj'] = my_Connection(**my_kwargs)
                ptr['connection_obj'].open()		# Force the connection to be established, for interactive shell.
                client = ptr['connection_obj'].client # This is Paramiko's SSHClient().
                client.set_missing_host_key_policy(AutoAddPolicy())
                ptr['channel'] = client.invoke_shell()
                ptr['channel'].settimeout(10 * 60) 	# In seconds.

    
            if 'cwd' in one_cmd:
                ptr['cwd'] = one_cmd['cwd']

            t_kwargs = {}
            t_kwargs['id'] = u_name
            t_kwargs['u_id'] = u_id
            t_kwargs['pty'] = True

            t_kwargs['watchers'] = []
            if 'watchers' in one_cmd:
                t_kwargs['watchers'] = one_cmd['watchers']

            t_kwargs['responses'] = []
            if 'responses' in one_cmd:
                t_kwargs['responses'] = one_cmd['responses']

            t_kwargs['warn'] = True
            if 'cmd' in one_cmd: 
                t_kwargs['cmd'] = cmd
            elif 'icmd' in one_cmd:
                t_kwargs['icmd'] = icmd
            else:
                print(f"ERROR: neither 'cmd' or 'icmd' found")
                raise('ERROR')

            result = self.run(**t_kwargs)

            return_code = result.return_code
            # print(f'------------------------ process_cmds(): return_code: {return_code}')

            if return_code != 0:
                print(f"ERROR: the return_code '{return_code}'.")
                if not re.search('ignore-errors', options):
                    raise('ERROR')
                else:
                    print(f'This error is ignored since this was enabled in the yaml file:')
                    print(f"    options: 'ignore-errors'")
    
            print('')

# ----------------------------------------------------------


# if __name__ == "__main__":
#     dummy_var = 1

