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
import re
import random
import socket
import yaml
import ipaddress
import copy


class my_docker():
    
    def __init__(self):
        dummy_var = 1

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
                one['connection'] = self.safe_eval(one['connection'])
    
            if 'cmd' in one:
                one['cmd'] = self.var_substitute(one['cmd'])	# Works

            if 'cwd' in one:
                one['cwd'] = self.var_substitute(one['cwd']) 

            if 'response' in one:
                responses = []
                for one_response in one['response']:
                    # print(f'(1) one_response: {one_response}')
                    # print(f'sudo_response: {sudo_response}')
                    one_response = self.safe_eval(one_response)
                    # print(f'(2) one_response: {one_response}')
                    name = one_response['name']
                    pattern = one_response['pattern']
                    response = one_response['response']
                    responder = Responder(pattern=pattern, response=response)
                    responses.append(responder)

                one['response'] = responses
                
    
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
    
 
    def run_in(self, **kwargs):
        # print(f'run_in(): kwargs: {kwargs}')
        if 'cwd' in kwargs:
            print(f"ERROR: run_in(): the cwd='{kwargs['cwd']}' was used, but is NOT allowed here. Move cwd= to my_cmds variable.")
            raise('ERROR')
    
        cmd_str = self.remove_extra_spaces(kwargs['cmd'])
        t1 = kwargs["id"]
        # print(f'g_c[{t1}]: {g_c[t1]}')
        if 'cwd' in g_c[kwargs["id"]]:
            cwd = g_c[kwargs["id"]]['cwd']
            # print(f'run_in(): -------------------> cwd: {cwd}')
        else:
            cwd = '/tmp/'
      
        if re.search('&', kwargs['cmd']):
            kwargs['cmd'] = f'cd {cwd} && ({kwargs["cmd"] })'
        else:
            kwargs['cmd'] = f'cd {cwd} && {kwargs["cmd"] }'
    
        connection = g_c[kwargs["id"]]['connection'] 
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
        result = connection.run(cmd, **kwargs)
        return_code = result.return_code
        # print(f'run_in(): return_code: {return_code}')
        return result
    
    def process_cmds(self, my_cmds):
        g_globals = globals()

        recipe = ''
        if type(my_cmds) is str:
            recipe = my_cmds
            print(f'(2) recipe: {recipe}')
            print('')
            my_cmds = g_globals[my_cmds]

        for one_cmd in my_cmds:
            # print(f'one_cmd: {one_cmd}')
            cmd = one_cmd['cmd']
            # print(f'cmd: {cmd}')
            u = one_cmd['connection']
            # print(f'(1) u: {u}')
            u_id = f'{u["user"]}@{u["host"]}'
            # print(f'u_id: {u_id}')
            t_id = g_id_func(u)    # TO DO: try using this one.
            # t_id = id(u)
            if 'options' in one_cmd:
                options = one_cmd['options']
            else:
                options = ''
    
            # print(f'options: {options}')
            # print(f'\n(2) Connection: {t_id}, Command: \'{cmd}\'\n---------------------------------')
            if not t_id in g_c:
                # print(f'-----------------> Calling Connection()')
                g_c[t_id] = {}
                # g_c[t_id]['id'] = t_id
                g_c[t_id]['id'] = g_id_func(t_id)
                my_kwargs = {}
                my_kwargs['host'] = u['host']
                my_kwargs['user'] = u['user']
                if 'password' in u:
                    my_kwargs['connect_kwargs'] = {'password': u['password']}
                elif 'connect_kwargs' in u:
                    my_kwargs['connect_kwargs'] = u['connect_kwargs']
    
                g_c[t_id]['connection'] = Connection(**my_kwargs)
    
            if 'cwd' in one_cmd:
                # print(f'------------------------> found "cwd" in one_cmd')
                g_c[t_id]['cwd'] = one_cmd['cwd']
   
            responses = []
            if 'response' in one_cmd:
                responses = one_cmd['response']
                # print(f'responses: {responses}')
    
            result = self.run_in(id=t_id, u_id=u_id, cmd=cmd, pty=True, watchers=responses, warn=True)
            return_code = result.return_code
            # print(f'return_code: {return_code}')
            if return_code != 0 and not re.search('ignore-errors', options):
                print(f'return_code: {return_code}')
                print(f"ERROR: the return_code '{return_code}'.")
                raise('ERROR')
    
            print('')

# ----------------------------------------------------------


# if __name__ == "__main__":
#     dummy_var = 1

