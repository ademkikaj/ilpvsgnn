from Experiments.experiment import Experiment
from bongardGenerator.bongardGenerator import generate_bongard_example


exp = Experiment(path='Bongard/test',dataset_name='bongard')


rule = generate_bongard_example(
    50,
    object_complexity=5,
    relation_complexity=3,
    rule_complexity=3,
    filename = exp.relational_test_path,
)

generate_bongard_example(
    100,
    object_complexity=5,
    relation_complexity=3,
    rule_complexity=rule,
    filename=exp.relational_path,
)





if __name__ == '__main__':
    exp.run_logic()
