import docker
import time
import os
import pandas as pd
import subprocess


class Tilde:
    
    def __init__(self, dataset_name,relational_path,target) -> None:
        self.dataset_name = dataset_name
        self.relational_path = relational_path
        self.target = target


    def run(self,tilde_input_path):
        # Start the docker container and run the bash script
        client = docker.from_env()
        container = client.containers.get('ace_system')
        if container.status != 'running':
            container.start()
        dir = "/user"
        start = time.time()
        base_path = "docker"
        relative_path = os.path.relpath(tilde_input_path,base_path)
        # bash script requires the path to the logic data
        result = container.exec_run(f"bash run_ace.sh {relative_path}/logic/tilde",workdir=dir)
        print("Tilde training output: ")
        print(result.output.decode('utf-8'))
        end = time.time()
        self.extract_tilde_program(tilde_input_path)
        runtime = end-start
        accuracy = self.test_program(tilde_input_path)

        # the output dataframe should be [ test_acc, runtim, train_acc, system, representation]
        training_results = self.extract_results(os.path.join(tilde_input_path,'logic','tilde'),train=True)
        train_acc = training_results['accuracy'][0]
        representation = os.path.basename(tilde_input_path)
        results = pd.DataFrame([{
            'test_acc': accuracy,
            'runtime': runtime,
            'train_acc': train_acc,
            'system': 'Tilde',
            'representation': representation
        }])

        return results

    def test_program(self,tilde_input_path):
        logic_path = os.path.join(tilde_input_path, "logic")
        program_path = os.path.join(logic_path,"tilde",f"{self.dataset_name}_out.pl")
        # if background exists else None
        background = os.path.join(logic_path,"tilde",f"{self.dataset_name}.bg")
        if not os.path.exists(background):
            background = "None"
        output = subprocess.run(["python","Benchmark/test_hypothesis.py",logic_path,self.dataset_name,self.relational_path,program_path,self.target,background],capture_output=True,text=True)
        #print(output)
        accuracy = float(output.stdout)
        return accuracy


    def extract_tilde_program(self, tilde_folder_path):
        
        result_path = os.path.join(tilde_folder_path,'logic','tilde','tilde',f"{self.dataset_name}.out")
        print(result_path)
        with open(result_path, 'r') as file:
            lines = file.readlines()
        
        start = lines.index("Equivalent prolog program:\n")
        program = []
        index = start + 2
        while lines[index].strip():
            program.append(lines[index])
            index += 2
        
        # map the [pos] and [neg] to pos and neg
        for i in range(len(program)):
            program[i] = program[i].replace("[pos]","pos")
            program[i] = program[i].replace("[neg]","neg")

        # add the styll check singleton
        program.insert(0,":- style_check(-singleton).\n")
        
        # write to a file
        output_file = os.path.join(tilde_folder_path,'logic','tilde',f"{self.dataset_name}_out.pl")
        with open(output_file, "w") as file:
            for line in program:
                file.write(line)
        return

    # Extract tilde results
    def extract_results(self,tilde_folder_path,train=True,test=False) -> pd.DataFrame:
        results_file = os.path.join(tilde_folder_path,f"tilde/{self.dataset_name}.out")
        with open(results_file, 'r') as file:
            lines = file.readlines()
        
        overall_metrics = {}
        start = lines.index(" **/\n")
        
        overall_metrics['time_discretization'] = float(lines[start+2].split()[-1])
        overall_metrics['time_induction'] = float(lines[start+3].split()[-1])

        # Extract the training metrics
        if train:
            training_start = lines.index("Training:\n") + 1
            training_end = lines.index("Testing:\n")
            training_metrics = {}
            training_section = lines[training_start:training_end]

            training_metrics['n_examples'] = int(training_section[0].split()[-1])
            training_metrics['n_pos_examples'] = int(training_section[4].split()[-1])
            training_metrics['n_neg_examples'] = int(training_section[5].split()[-1])
            
            training_metrics['accuracy'] = float(training_section[9].split()[1])
            training_metrics['accuracy_std'] = float(training_section[9].split()[3].strip(','))
            training_metrics['accuracy_default'] = float(training_section[9].split()[-1].strip(')'))
            training_metrics['cramers_coeff'] = float(training_section[10].split()[2])
            training_metrics['TP'] = float(training_section[11].split()[4].strip(','))
            training_metrics['FN'] = float(training_section[11].split()[-1])
            training_metrics['type'] = 'training'

        testing_metrics = {}
        if test:
            # Extract the testing metrics
            testing_start = training_end + 1
            testing_end = lines.index('Compact notation of tree:\n') - 2
            testing_section = lines[testing_start:testing_end]

            testing_metrics['n_examples'] = int(testing_section[0].split()[-1])
            testing_metrics['n_pos_examples'] = int(testing_section[4].split()[-1])
            testing_metrics['n_neg_examples'] = int(testing_section[5].split()[-1])

            testing_metrics['accuracy'] = float(testing_section[9].split()[1])
            testing_metrics['accuracy_std'] = float(testing_section[9].split()[3].strip(','))
            testing_metrics['accuracy_default'] = float(testing_section[9].split()[-1].strip(')'))
            testing_metrics['cramers_coeff'] = float(testing_section[10].split()[2])
            testing_metrics['TP'] = float(testing_section[11].split()[4].strip(','))
            testing_metrics['FN'] = float(testing_section[11].split()[-1])
            testing_metrics['type'] = 'testing'

            training_metrics['total_examples'] = training_metrics['n_examples'] + testing_metrics['n_examples']
            testing_metrics['total_examples'] = training_metrics['n_examples'] + testing_metrics['n_examples']

        # time metrics
        training_metrics['time_total'] = float(overall_metrics['time_discretization']) + float(overall_metrics['time_induction'])
        testing_metrics['time_total'] = float(overall_metrics['time_discretization']) + float(overall_metrics['time_induction'])
        
        df = pd.DataFrame([training_metrics, testing_metrics])
        # should be written to the results directory: Tilde and some identifier
        return df

    
    
        