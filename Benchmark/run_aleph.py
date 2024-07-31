import sys
import os
import logging
import subprocess
import tempfile
from subprocess import CalledProcessError, TimeoutExpired
from time import perf_counter
import re

RE = "<PROG>\n(.*)<\/PROG>"
CMD = "induce(P),writeln('<PROG>'),numbervars(P,0,_),foreach(member(C,P),(write(C),write('. '))),writeln('</PROG>')"

class Duration:
    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.execution_time = perf_counter() - self.start_time


def get_logger():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger("popperexperiments")

def run_command(cmd, args, timeout = None):
    if not isinstance(cmd, list):
        cmd = [cmd]

    final_args = cmd


    if isinstance(args, dict):
        for (k, v) in args.items():
            final_args.append(k)
            if v != None and v != "":
                final_args.append(v)
    else:
        final_args.extend(args)

    logger = get_logger()
    
    try:
        proc = subprocess.run(
            final_args, 
            encoding="utf-8", 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout)
        
    except TimeoutExpired as timeout:
        logger.debug(f"Timeout running command {cmd} {args} : {repr(timeout)}")
        return ""

    if proc.stdout:
        result = proc.stdout
        logger.debug(result)
    else:
        result = ""

    if proc.stderr:
        logger.debug(proc.stderr)
        print(proc.stderr)

    if proc.returncode < 0:
        raise CalledProcessError(proc.returncode, cmd, proc.stdout + proc.stderr)

    #print(proc.stdout)
    
    return result

def call_prolog(action, files_to_load=[], timeout=None):
    args = ["-g", action, "-t", "halt", "-q"]

    #TODO(Brad): This feels like a huge hack to make Aleph work
    if len(files_to_load) == 1:
        args.append("-s") 
        args.append(files_to_load[0])

        return run_command('swipl', args, timeout=timeout)
    else:
        with tempfile.NamedTemporaryFile() as temp_file:
            files = ', '.join([f'\"{f}\"' for f in files_to_load])
            cmd = f":- load_files([{files}],[silent(true)])."
            with open(temp_file.name, 'w') as f:
                f.write(cmd) 
                f.flush()
            
            args.append("-s") 
            args.append(temp_file.name)

            return run_command('swipl', args, timeout=timeout)

def time_prolog_experiment(action, files_to_load=[], timeout=None):
    with Duration() as d:
        try:
            prog = call_prolog(action, files_to_load=files_to_load, timeout=timeout)
        except Exception as e:
            logger = get_logger()
            logger.error(f"Exception timing prolog {action} :: {repr(e)}")
            prog = ""

    return (prog, d.execution_time)


def learn(aleph_path):

    modes_path = os.path.join(aleph_path, "modes.pl")
    background_path = os.path.join(aleph_path, "background.pl")
    pos_example_path = os.path.join(aleph_path, "pos_example.f")
    neg_example_path = os.path.join(aleph_path, "neg_example.n")
    extra_bg_path = os.path.join(aleph_path, "extra_bg.pl")
    

    with open(modes_path, 'r') as f:
        modes = f.read()
    with open (background_path, 'r') as f:
        background = f.read()
    with open(pos_example_path, 'r') as f:
        pos_example = f.readlines()
    with open(neg_example_path, 'r') as f:
        neg_example = f.readlines()
    # if there is an extra background file, add it to the background
    if os.path.exists(extra_bg_path):
        with open(extra_bg_path, 'r') as f:
            background = f.read() + background

    output_path = os.path.join(aleph_path, "working.pl")
    with open(output_path, 'w') as f:
        f.write(modes + "\n")

        f.write(':-begin_bg.\n')
        f.write(background + "\n")
        f.write(':-end_bg.\n')

        f.write(':-begin_in_pos.\n')
        for ex in pos_example:
            f.write(ex)
        f.write(':-end_in_pos.\n')

       
        f.write(':-begin_in_neg.\n')
        for ex in neg_example:
            f.write(ex)
        f.write(':-end_in_neg.\n')
    
    (output,total_exec_time) = time_prolog_experiment(CMD,[output_path],timeout=600)


    
    code = re.search(RE,output)
    code = code.group(1).replace('. ', '.\n') if code else None
    
    print((code,total_exec_time))
    return 




if __name__ == '__main__':
    # parse arguments from command line
    aleph_path = sys.argv[1]
    
    learn(aleph_path)
