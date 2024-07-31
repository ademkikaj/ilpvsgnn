from Experiments.experiment import Experiment
from bongardGenerator.bongardGenerator import generate_bongard_example
import os
import numpy as np

## Bongard Dataset

bongard_relations = {
        'bongard': ['problemId','class'],
        'square': ['problemId','objectId'],
        'circle': ['problemId','objectId'],
        'triangle': ['problemId','objectId'],
        'in': ['problemId','objectId1','objectId2']
}

# scale experiment
def scaleExperiment():
    amount_of_test_examples = 50
    object_complexity = 5
    relation_complexity = 3
    rule_complexity = 3
    # Scale experiment: change the number of examples the learners are trained on
    scaling_experiment = Experiment(
        path = "Bongard/ScalingExperiment",
        logic = False,
        relational=True,
        graph = False,
        dataset_name='bongard',
        gnn_config_path='Experiments/gnn_config.yaml',
        relations=bongard_relations
    )
    # generate the test data, the same test set can be used for all number of exapmles
    rule = generate_bongard_example(
        num_examples=amount_of_test_examples,
        object_complexity=object_complexity,
        relation_complexity=relation_complexity,
        rule_complexity=rule_complexity,
        filename=scaling_experiment.relational_path + "/test/"
    )
    values = [10,20,50,100,200,500,1000,2000]
    for index, i in enumerate(values):
        num_examples = i

        generate_bongard_example(
            num_examples=num_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=rule,
            filename=scaling_experiment.relational_path
        )
        scaling_experiment.run()
        # log additional information
        scaling_experiment.log_extra_metrics('num_examples', num_examples,index)
        scaling_experiment.log_extra_metrics('object_complexity', object_complexity,index)
        scaling_experiment.log_extra_metrics('relation_complexity', relation_complexity,index)
        scaling_experiment.log_extra_metrics('rule_complexity', rule_complexity,index)

    scaling_experiment.combine_results()

    return



# exponential scale experiment
def scaleExperimentExponential():
    amount_of_test_examples = 50
    object_complexity = 5
    relation_complexity = 3
    rule_complexity = 3
    # Scale experiment: change the number of examples the learners are trained on
    scaling_experiment = Experiment(
        path = "Bongard/ScalingExperimentExponential",
        logic = False,
        relational=True,
        graph = False,
        dataset_name='bongard',
        gnn_config_path='Experiments/gnn_config.yaml',
        relations=bongard_relations,
    )
    # generate the test data, the same test set can be used for all number of exapmles
    rule = generate_bongard_example(
        num_examples=amount_of_test_examples,
        object_complexity=object_complexity,
        relation_complexity=relation_complexity,
        rule_complexity=rule_complexity,
        filename=scaling_experiment.relational_path + "/test/"
    )
    start_exp = 1
    stop_exp  = 7
    values = np.power(10,np.arange(start_exp,stop_exp,1))
    values = values.tolist()
    print(values)
    for index, i in enumerate(values):
        num_examples = i
        generate_bongard_example(
            num_examples=num_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=rule,
            filename=scaling_experiment.relational_path
        )
        scaling_experiment.run()
        # log additional information
        scaling_experiment.log_extra_metrics('num_examples', num_examples,index)
        scaling_experiment.log_extra_metrics('object_complexity', object_complexity,index)
        scaling_experiment.log_extra_metrics('relation_complexity', relation_complexity,index)
        scaling_experiment.log_extra_metrics('rule_complexity', rule_complexity,index)

    scaling_experiment.combine_results()





# Object complexity experiment: change the number of different objects in a single expample

def object_complexity_experiment():
    amount_of_test_examples = 50
    num_examples = 100
    relation_complexity = 3
    # rule complexity must be smaller than or the object complexity or relation complexity
    rule_complexity = 3
    # Scale experiment: change the number of examples the learners are trained on
    object_experiment = Experiment(
        path = "Bongard/ObjectComplexityExperiment",
        logic = False,
        relational=True,
        graph = False,
        dataset_name='bongard',
        gnn_config_path='Experiments/gnn_config.yaml',
        relations=bongard_relations,
    )
    # Different test set for every object complexity

    for index, i in enumerate(range(5,40,5)):
        object_complexity = i
        # test set
        rule = generate_bongard_example(
            num_examples=amount_of_test_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=rule_complexity,
            filename=object_experiment.relational_path + "/test/"
        )
        # training set
        generate_bongard_example(
            num_examples=num_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=rule,
            filename=object_experiment.relational_path
        )
        object_experiment.run()
        # log additional information
        object_experiment.log_extra_metrics('num_examples', num_examples,index)
        object_experiment.log_extra_metrics('object_complexity', object_complexity,index)
        object_experiment.log_extra_metrics('relation_complexity', relation_complexity,index)
        object_experiment.log_extra_metrics('rule_complexity', rule_complexity,index)

    object_experiment.combine_results()
    return


# Relation complexity experiment: change the number of different relations in a single example
# limitation: the relation complexity must be smaller than the object complexity

def relation_complexity_experiment():
    amount_of_test_examples = 50
    num_examples = 100
    object_complexity = 5
    # rule complexity must be smaller than or the object complexity or relation complexity
    rule_complexity = 5
    # Scale experiment: change the number of examples the learners are trained on
    experiment = Experiment(
        path = "Bongard/RelationComplexityExperiment",
        logic = False,
        relational=True,
        graph = False,
        dataset_name='bongard',
        gnn_config_path='Experiments/gnn_config.yaml',
        relations=bongard_relations,
    )
    # Different test set for every object complexity

    for index, i in enumerate(range(3,object_complexity,1)):
        relation_complexity = i
        # test set
        rule = generate_bongard_example(
            num_examples=amount_of_test_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=rule_complexity,
            filename=experiment.relational_path + "/test/"
        )
        # training set
        generate_bongard_example(
            num_examples=num_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=rule,
            filename=experiment.relational_path
        )
        experiment.run()
        # log additional information
        experiment.log_extra_metrics('num_examples', num_examples,index)
        experiment.log_extra_metrics('object_complexity', object_complexity,index)
        experiment.log_extra_metrics('relation_complexity', relation_complexity,index)
        experiment.log_extra_metrics('rule_complexity', rule_complexity,index)

    experiment.combine_results()
    return


# Rule complexity experiment: change the complexity of the rules
# Take very large object complexity and relation complexity and increase the rule complexity

def rule_complexity_experiment():
    amount_of_test_examples = 50
    num_examples = 100
    object_complexity = 8
    relation_complexity = 6
    # rule complexity must be smaller than or the object complexity or relation complexity
    # Scale experiment: change the number of examples the learners are trained on
    experiment = Experiment(
        path = "Bongard/RuleComplexityExperiment",
        logic = False,
        relational=True,
        graph = False,
        dataset_name='bongard',
        gnn_config_path='Experiments/gnn_config.yaml',
        relations=bongard_relations,
    )
    # Different test set for every object complexity

    for index, i in enumerate(range(2,relation_complexity,1)):
        rule_complexity = i
        # test set
        rule = generate_bongard_example(
            num_examples=amount_of_test_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=rule_complexity,
            filename=experiment.relational_path + "/test/"
        )
        # training set
        generate_bongard_example(
            num_examples=num_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=rule,
            filename=experiment.relational_path
        )
        experiment.run()
        # log additional information
        experiment.log_extra_metrics('num_examples', num_examples,index)
        experiment.log_extra_metrics('object_complexity', object_complexity,index)
        experiment.log_extra_metrics('relation_complexity', relation_complexity,index)
        experiment.log_extra_metrics('rule_complexity', rule_complexity,index)

    experiment.combine_results()
    return


# Object and relation complexity experiment: change the number of different objects and relations in a single example
def object_relation_complexity_experiment():
    amount_of_test_examples = 50
    num_examples = 200
    rule_complexity = 5
    # rule complexity must be smaller than or the object complexity or relation complexity
    # Scale experiment: change the number of examples the learners are trained on
    experiment = Experiment(
        path = "Bongard/ObjectRelationComplexityExperiment",
        logic = False,
        relational=True,
        graph = False,
        dataset_name='bongard',
        gnn_config_path='Experiments/gnn_config.yaml',
        relations=bongard_relations,
    )
    # Different test set for every object complexity

    for index, i in enumerate(range(5,15,1)):
        object_complexity = i
        relation_complexity = i-2
        # test set
        rule = generate_bongard_example(
            num_examples=amount_of_test_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=rule_complexity,
            filename=experiment.relational_path + "/test/"
        )
        # training set
        generate_bongard_example(
            num_examples=num_examples,
            object_complexity=object_complexity,
            relation_complexity=relation_complexity,
            rule_complexity=rule,
            filename=experiment.relational_path
        )
        experiment.run()
        # log additional information
        experiment.log_extra_metrics('num_examples', num_examples,index)
        experiment.log_extra_metrics('object_complexity', object_complexity,index)
        experiment.log_extra_metrics('relation_complexity', relation_complexity,index)
        experiment.log_extra_metrics('rule_complexity', rule_complexity,index)

    experiment.combine_results()
    return


# TO DO:

# Object variety experiment: change the number of different objects




# Robustness experiment: change the number of examples that are incorrect in training



if __name__ == "__main__":
    #scaleExperiment()
    #object_complexity_experiment()
    #rule_complexity_experiment()
    #object_relation_complexity_experiment()
    
    relation_complexity_experiment()
    #scaleExperimentExponential()

