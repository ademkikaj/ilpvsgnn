from Experiments.experiment import Experiment
from bongardGenerator.bongardGenerator import generate_bongard_example
import os


## Bongard Dataset

bongard_relations = {
        'bongard': ['problemId','class'],
        'square': ['problemId','objectId'],
        'circle': ['problemId','objectId'],
        'triangle': ['problemId','objectId'],
        'in': ['problemId','objectId1','objectId2']
}

def scalingExperiment():
    # Scale experiment: change the amount of examples

    scalingExperiment = Experiment(
        path = "Bongard/LogicScalingExperiment",
        logic = False,
        relational=True,
        graph = False,
        dataset_name='bongard',
        gnn_config_path='gnn_config.yaml',
        relations=bongard_relations,
    )

    for i in range(20,105,5):

        num_examples = i
        object_complexity = 5
        relation_complexity = 2
        generate_bongard_example(
            num_examples=num_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=3,
            filename=scalingExperiment.relational_path
        )
        scalingExperiment.run_logic()
        # log additional information
        
    #scalingExperiment.combine_results()

def scalingExperimentgnn():
    # Scale experiment: change the amount of examples

    scalingExperiment = Experiment(
        path = "Bongard/GNNScalingExperiment",
        logic = True,
        graph = False,
        dataset_name='bongard',
        gnn_config_path='gnn_config.yaml',
        relations=bongard_relations,
    )

    for i in range(20,105,5):

        num_examples = i
        object_complexity = 5
        relation_complexity = 2
        generate_bongard_example(
            num_examples=num_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=3,
            filename=scalingExperiment.logic_path + f"/bongard.kb"
        )
        scalingExperiment.run_gnn()
        # log additional information
        
    # scalingExperiment.combine_results()
    
def singleExperiment():
    # Scale experiment: change the amount of examples
    singleExperiment= Experiment(
        path = "Bongard/SingleExperiment",
        logic = False,
        relational=True,
        graph = False,
        dataset_name='bongard',
        gnn_config_path='gnn_config.yaml',
        relations=bongard_relations,
    )
    # generate the test data
    rule = generate_bongard_example(
        num_examples=50,
        object_complexity=5,
        relation_complexity=2,
        rule_complexity=3,
        filename=singleExperiment.relational_path + "/test"
    )

    # generate the training data
    num_examples = 100
    object_complexity = 5
    relation_complexity = 2
    generate_bongard_example(
        num_examples=num_examples,
        object_complexity=object_complexity,
        relation_complexity=relation_complexity,
        rule_complexity=rule,
        filename=singleExperiment.relational_path
    )
    singleExperiment.run_logic()
    # log additional information
    singleExperiment.log_extra_metrics('num_examples',num_examples ,0,gnn=False)
    singleExperiment.log_extra_metrics('object_complexity', object_complexity,0,gnn=False)
    singleExperiment.log_extra_metrics('relation_complexity', relation_complexity,0,gnn=False)

    singleExperiment.index += 1

    singleExperiment.combine_results(gnn=False)


if __name__ == '__main__':
    #scalingExperiment()
    #scalingExperimentgnn()
    singleExperiment()