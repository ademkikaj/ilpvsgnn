import os
from runner import Run 
from utils.bongardGenerator.bongardGenerator import generate_bongard_example
from utils.bongardGenerator.bongardGenerator import generate_rule
from Benchmark.bongard.toLogic import toLogic
from Benchmark.bongard.toGraph import toGraph
import pandas as pd
import random


# scalability experiment: increase the number of examples the systems are trained on

ilpSettings = {
            "tilde": {
                "node_only" : [
                    "predict(bongard(+B,-C)).\n",
                    "rmode(triangle(+id,-object)).\n",
                    "rmode(square(+id,-object)).\n",
                    "rmode(circle(+id,-object)).\n",
                    "rmode(in(+id,-object)).\n",
                    "rmode(edge(+id,-object,-object)).\n",
                    "typed_language(yes).\n",
                    "type(bongard(pic,class)).\n",
                    "type(triangle(pic,obj)).\n",
                    "type(square(pic,obj)).\n",
                    "type(circle(pic,obj)).\n",
                    "type(in(pic,obj)).\n",
                    "type(edge(pic,obj,obj)).\n",
                    "auto_lookahead(triangle(Id,Obj),[Obj]).\n",
                    "auto_lookahead(square(Id,Obj),[Obj]).\n",
                    "auto_lookahead(circle(Id,Obj),[Obj]).\n",
                    "auto_lookahead(in(Id,Obj),[Obj]).\n",
                    "auto_lookahead(edge(Id,Obj1,Obj2),[Obj1,Obj2]).\n",
                ],
                "node_edge": [
                    "predict(bongard(+id,-C)).\n",
                    "warmode(triangle(+id,+-S)).\n",
                    "warmode(square(+id,+-S)).\n",
                    "warmode(circle(+id,+-S)).\n",
                    "warmode(edge(+id,+-S,+-S)).\n",
                    "warmode(shape1(+id,+-S)).\n",
                    "warmode(shape2(+id,+-S)).\n",
                    "warmode(shape3(+id,+-S)).\n",
                    "warmode(shape4(+id,+-S)).\n",
                    "warmode(shape5(+id,+-S)).\n",
                    "auto_lookahead(triangle(Id,S),[S]).\n",
                    "auto_lookahead(square(Id,S),[S]).\n",
                    "auto_lookahead(circle(Id,S),[S]).\n",
                    "auto_lookahead(edge(Id,S1,S2),[S1,S2]).\n",
                    "auto_lookahead(shape1(Id,S),[S]).\n",
                    "auto_lookahead(shape2(Id,S),[S]).\n",
                    "auto_lookahead(shape3(Id,S),[S]).\n",
                    "auto_lookahead(shape4(Id,S),[S]).\n",
                    "auto_lookahead(shape5(Id,S),[S]).\n",
                ],
                "edge_based": [
                    "predict(bongard(+B,-C)).\n",
                    "rmode(shape(+P,triangle,+-C)).\n",
                    "rmode(shape(+P,square,+-C)).\n",
                    "rmode(shape(+P,circle,+-C)).\n",
                    "rmode(in(+P,+S1,+-S2)).\n",
                    "rmode(instance(+P,+-C)).\n",
                    "typed_language(yes).\n",
                    "type(bongard(pic,class)).\n",
                    "type(in(pic,obj,obj)).\n",
                    "type(shape(pic,fact,obj)).\n",
                    "type(instance(pic,obj)).\n",
                    "auto_lookahead(shape(Id,T,C),[T,C]).\n",
                    "auto_lookahead(in(Id,S1,S2),[S1,S2]).\n",
                    "auto_lookahead(instance(Id,C),[C]).\n",
                ],
                "Klog": [
                    "predict(bongard(+id,-C)).\n",
                    "warmode(triangle(+id,-object)).\n",
                    "warmode(square(+id,-object)).\n",
                    "warmode(circle(+id,-object)).\n",
                    "warmode(edge(+id,-object,-object)).\n",
                    "warmode(in(+id,-object)).\n",
                    "auto_lookahead(triangle(Id,Obj),[Obj]).\n",
                    "auto_lookahead(square(Id,Obj),[Obj]).\n",
                    "auto_lookahead(circle(Id,Obj),[Obj]).\n",
                    "auto_lookahead(edge(Id,Obj1,Obj2),[Obj1,Obj2]).\n",
                    "auto_lookahead(in(Id,Obj),[Obj]).\n",
                ]
            },
            "popper": {
                "node_only": [
                    "body_pred(square,2).\n",
                    "body_pred(circle,2).\n",
                    "body_pred(triangle,2).\n",
                    "body_pred(in,2).\n",
                    "body_pred(edge,3).\n",
                    "type(bongard,(id,)).\n",
                    "type(square,(id,object)).\n",
                    "type(circle,(id,object)).\n",
                    "type(triangle,(id,object)).\n",
                    "type(in,(id,object)).\n",
                    "type(edge,(id,object,object)).\n",
                ],
                "node_edge": [
                    "body_pred(square,2).\n",
                    "body_pred(circle,2).\n",
                    "body_pred(triangle,2).\n",
                    "body_pred(edge,3).\n",
                    "type(bongard,(id,)).\n",
                    "type(square,(id,object)).\n",
                    "type(circle,(id,object)).\n",
                    "type(triangle,(id,object)).\n",
                    "type(edge,(id,object,object)).\n",
                ],
                "edge_based": [
                    "body_pred(shape,3).\n",
                    "body_pred(in,3).\n",
                    "type(bongard,(id,)).\n",
                    "type(in,(id,object,object)).\n"
                    "type(shape,(id,fact,object)).\n"
                ],
                "Klog": [
                    "body_pred(square,2).\n",
                    "body_pred(circle,2).\n",
                    "body_pred(triangle,2).\n",
                    "body_pred(edge,3).\n",
                    "body_pred(in,2).\n",
                    "type(bongard,(id,)).\n",
                    "type(square,(id,object)).\n",
                    "type(circle,(id,object)).\n",
                    "type(triangle,(id,object)).\n",
                    "type(in,(id,object)).\n",
                    "type(edge,(id,object,object)).\n",
                ],
            },
            "aleph": {
                "node_only": [
                    ":- modeb(*,square(+id, -object)).\n",
                    ":- modeb(*,circle(+id, -object)).\n",
                    ":- modeb(*,triangle(+id, -object)).\n",
                    ":- modeb(*,in(+id, -object)).\n",
                    ":- modeb(*,edge(+id, -object, -object)).\n",
                    ":- determination(bongard/1,square/2).\n",
                    ":- determination(bongard/1,circle/2).\n",
                    ":- determination(bongard/1,triangle/2).\n",
                    ":- determination(bongard/1,in/2).\n",
                    ":- determination(bongard/1,edge/3).\n"
                ],
                "node_edge": [
                    ":- modeb(*,square(+id, -object)).\n",
                    ":- modeb(*,circle(+id, -object)).\n",
                    ":- modeb(*,triangle(+id, -object)).\n",
                    ":- modeb(*,edge(+id, -object, -object)).\n",
                    ":- determination(bongard/1,square/2).\n",
                    ":- determination(bongard/1,circle/2).\n",
                    ":- determination(bongard/1,triangle/2).\n",
                    ":- determination(bongard/1,edge/3).\n"
                ],
                "edge_based": [
                    ":- modeb(1,square(+constant)).\n",
                    ":- modeb(1,circle(+constant)).\n",
                    ":- modeb(1,triangle(+constant)).\n",
                    ":- modeb(*,shape(+id, -constant, -object)).\n",
                    ":- modeb(*,in(+id, -object, -object)).\n",
                    ":- modeb(*,instance(+id, -object)).\n",
                    ":- determination(bongard/1,shape/3).\n",
                    ":- determination(bongard/1,in/3).\n",
                    ":- determination(bongard/1,square/1).\n",
                    ":- determination(bongard/1,circle/1).\n",
                    ":- determination(bongard/1,triangle/1).\n",
                    ":- determination(bongard/1,instance/2).\n"
                ],
                "Klog":[
                    ":- modeb(*,square(+id, -object)).\n",
                    ":- modeb(*,circle(+id, -object)).\n",
                    ":- modeb(*,triangle(+id, -object)).\n",
                    ":- modeb(*,in(+id, -object)).\n",
                    ":- modeb(*,edge(+id, -object, -object)).\n",
                    ":- determination(bongard/1,square/2).\n",
                    ":- determination(bongard/1,circle/2).\n",
                    ":- determination(bongard/1,triangle/2).\n",
                    ":- determination(bongard/1,in/2).\n"
                    ":- determination(bongard/1,edge/3).\n"
                ]
            }
        }

# max bongard examples 392

# rule = generate_rule(2)s
# print("Rule: ", rule)

# representations = ["node_edge"]

# for object_complexity in [4,5,10,15,20,25,30]:

#     # generate the relational data
#     output_path = os.path.join("docker","Experiment","bongard","relational")
#     generate_bongard_example(num_examples=300,
#                              object_complexity=object_complexity,
#                              relation_complexity=object_complexity-2,
#                              rule_complexity=rule,
#                              filename=output_path)

#     # only select the num_examples training examples
#     # df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational","original","bongard.csv"))
#     # df = df.sample(num_examples)
#     # df.to_csv(os.path.join("docker","Experiment","bongard","relational","train","bongard.csv"),index=False)
    
#     bongard_experiment_runner = Run(
#         "bongard",
#         representations,
#         "class",
#         "id",
#         toGraph,
#         toLogic,
#         ilpSettings,
#         "Experiment",
#         object_complexity,
#         split_data=True
#     )
#     bongard_experiment_runner.TILDE = False
#     bongard_experiment_runner.ALEPH = True
#     bongard_experiment_runner.POPPER = False
#     bongard_experiment_runner.GNN = False

    
#     bongard_experiment_runner.run()


#for num_examples in [10,20,50,100,200]:
# for num_examples in [100]:

    
#     # only select the num_examples training examples
#     df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational","original","bongard.csv"))
#     df = df.sample(num_examples)
#     df.to_csv(os.path.join("docker","Experiment","bongard","relational","train","bongard.csv"),index=False)
#     df.to_csv(os.path.join("docker","Experiment","bongard","relational","bongard.csv"),index=False)
    
#     bongard_experiment_runner = Run(
#         "bongard",
#         ["node_edge","edge_based","Klog"],
#         "class",
#         "id",
#         toGraph,
#         toLogic,
#         ilpSettings,
#         "Experiment",
#         num_examples,
#         split_data=False
#     )
#     bongard_experiment_runner.TILDE = True
#     bongard_experiment_runner.ALEPH = True
#     bongard_experiment_runner.POPPER = False
#     bongard_experiment_runner.GNN = False

    
#     bongard_experiment_runner.run()



files = ["bongard.csv","circle.csv","square.csv","triangle.csv","in.csv"]
node_files = ["circle.csv","square.csv","triangle.csv"]


def get_amount_of_nodes(node_files,i):
    amount_of_nodes = 0
    for file in node_files:
        df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational",file))
        current_df = df[df["id"] == i]
        amount_of_nodes += len(current_df)
    return amount_of_nodes


def get_random_node(node_files,i):
    not_zero = True
    while not_zero:
        shape = random.choice(node_files)
        df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational",shape))
        possible_rows = df[df["id"] == i]
        if len(possible_rows) > 0:
            not_zero = False
    row = possible_rows.sample(1)
    id = possible_rows.sample(1)["objectId"]
    return id.iloc[0]

for additional_nodes in [2,3,4,5,6,7,8,9,10]:

    for file in files:
        df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational","original",file))
        df.to_csv(os.path.join("docker","Experiment","bongard","relational",file),index=False)
        if file != "bongard.csv":
            df.to_csv(os.path.join("docker","Experiment","bongard","relational","train",file),index=False)
            df.to_csv(os.path.join("docker","Experiment","bongard","relational","test",file),index=False)

    df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational","original","bongard.csv"))
    ids = df["id"].unique()
    
    for i in ids:
        amount_of_nodes = get_amount_of_nodes(node_files,i)
        for j in range(additional_nodes):
            object = random.choice(node_files)
            df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational",object))
            # add a node for this i example
            new_row = {"id": i, "objectId": "o" + str(amount_of_nodes + j+1)}
            df = pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)
            df.to_csv(os.path.join("docker","Experiment","bongard","relational",object),index=False)

        for j in range(additional_nodes-1):
            # add a relation for this i example
            id1 = get_random_node(node_files,i)
            id2 = get_random_node(node_files,i)
            
            new_row = {"id": i, "objectId1": id1, "objectId2": id2}
            df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational","in.csv"))
            df = pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)
            df.to_csv(os.path.join("docker","Experiment","bongard","relational","in.csv"),index=False)

    bongard_experiment_runner = Run(
        "bongard",
        ["node_only"],
        "class",
        "id",
        toGraph,
        toLogic,
        ilpSettings,
        "Experiment",
        additional_nodes,
        split_data=True
    )
    bongard_experiment_runner.TILDE = True
    bongard_experiment_runner.ALEPH = False
    bongard_experiment_runner.POPPER = False
    bongard_experiment_runner.GNN = True


    bongard_experiment_runner.run()



# files = ["bongard.csv","circle.csv","square.csv","triangle.csv","in.csv"]
# node_files = ["circle.csv","square.csv","triangle.csv"]
# new_shapes = ["shape1.csv","shape2.csv","shape3.csv","shape4.csv","shape5.csv"]

# relations_to_add = 2
# nodes_to_add = 10
# for additional_shapes in [1,2,3,4,5]:
    

#     for file in files:
#         df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational","original",file))
#         df.to_csv(os.path.join("docker","Experiment","bongard","relational",file),index=False)
#         if file != "bongard.csv":
#             df.to_csv(os.path.join("docker","Experiment","bongard","relational","train",file),index=False)
#             df.to_csv(os.path.join("docker","Experiment","bongard","relational","test",file),index=False)
    
#     # delete the new shape csv files if they exist
#     for file in new_shapes:
#         if os.path.exists(os.path.join("docker","Experiment","bongard","relational",file)):
#             os.remove(os.path.join("docker","Experiment","bongard","relational",file))
#         if os.path.exists(os.path.join("docker","Experiment","bongard","relational","train",file)):
#             os.remove(os.path.join("docker","Experiment","bongard","relational","train",file))
#         if os.path.exists(os.path.join("docker","Experiment","bongard","relational","test",file)):
#             os.remove(os.path.join("docker","Experiment","bongard","relational","test",file))

#     df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational","original","bongard.csv"))
#     ids = df["id"].unique()

#     for i in ids:
#         amount_of_nodes = get_amount_of_nodes(node_files,i)
#         for j in range(nodes_to_add):
#             shape = random.choice(new_shapes[:additional_shapes])
#             if os.path.exists(os.path.join("docker","Experiment","bongard","relational",shape)):
#                 df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational",shape))
#                 new_row = {"id": i, "objectId": "o" + str(amount_of_nodes + j+1)}
#                 df = pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)
#                 df.to_csv(os.path.join("docker","Experiment","bongard","relational",shape),index=False)
#             else:
#                 new_row = {"id": i, "objectId": "o" + str(amount_of_nodes + j+1)}
#                 df = pd.DataFrame([new_row])
#                 df.to_csv(os.path.join("docker","Experiment","bongard","relational",shape),index=False)

#         for j in range(relations_to_add):
#             id1 = get_random_node(node_files+new_shapes[:additional_shapes],i)
#             id2 = get_random_node(node_files+new_shapes[:additional_shapes],i)
#             new_row = {"id": i, "objectId1": id1, "objectId2": id2}
#             df = pd.read_csv(os.path.join("docker","Experiment","bongard","relational","in.csv"))
#             df = pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)
#             df.to_csv(os.path.join("docker","Experiment","bongard","relational","in.csv"),index=False)
        

            
#     bongard_experiment_runner = Run(
#             "bongard",
#             ["node_edge"],
#             "class",
#             "id",
#             toGraph,
#             toLogic,
#             ilpSettings,
#             "Experiment",
#             additional_shapes,
#             split_data=True
#         )
#     bongard_experiment_runner.TILDE = True
#     bongard_experiment_runner.ALEPH = False
#     bongard_experiment_runner.POPPER = False
#     bongard_experiment_runner.GNN = True

#     bongard_experiment_runner.run()