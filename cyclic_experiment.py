
import os
from runner import Run 
from Benchmark.cyclic.generate_data_final import generate_relational_data
from Benchmark.cyclic.toLogic import toLogic
from Benchmark.cyclic.toGraph import toGraph
import pandas as pd
import random


# scalability experiment: increase the number of examples the systems are trained on

ilpSettings = {
            "tilde" :{
                "node_only": [
                    "max_lookahead(1).\n",
                    "query_batch_size(50000).\n",
                    "predict(cyclic(+B,-C)).\n",
                    "warmode(edge(+id_int,-id_int)).\n",
                    "warmode(node(+id,+-color)).\n",
                    "warmode(green(+color)).\n",
                    "warmode(red(+color)).\n",
                    "auto_lookahead(edge(Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).\n",
                    "auto_lookahead(node(Node_id,Color),[Node_id,Color]).\n",
                ],
                "node_edge": [
                    "max_lookahead(1).\n",
                    "query_batch_size(50000).\n",
                    "predict(cyclic(+B,-C)).\n",
                    "warmode(edge(+id,-id)).\n",
                    "warmode(node(+id,+-color)).\n",
                    "warmode(green(+color)).\n",
                    "warmode(red(+color)).\n",
                    "auto_lookahead(edge(Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).\n",
                    "auto_lookahead(node(Node_id,Color),[Node_id,Color]).\n",
                ],
                "edge_based": [
                    "max_lookahead(1).\n",
                    "query_batch_size(50000).\n",
                    "predict(cyclic(+B,-C)).\n",
                    "warmode(edge(+id,-id)).\n",
                    "warmode(node(+id,+-color)).\n",
                    "warmode(green(+color)).\n",
                    "warmode(red(+color)).\n",
                    "warmode(instance(+id)).\n",
                    "auto_lookahead(edge(Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).\n",
                    "auto_lookahead(node(Node_id,Color),[Node_id,Color]).\n",
                    "auto_lookahead(instance(Id),[Id]).\n",
                ],
                "Klog": [
                    "max_lookahead(1).\n",
                    "query_batch_size(50000).\n",
                    "predict(cyclic(+B,-C)).\n",
                    "warmode(edge(+id,-id)).\n",
                    "warmode(node(+id,+-color)).\n",
                    "warmode(green(+color)).\n",
                    "warmode(red(+color)).\n",
                    "warmode(klog_edge(+id)).\n",
                    "auto_lookahead(edge(Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).\n",
                    "auto_lookahead(node(Node_id,Color),[Node_id,Color]).\n",
                    "auto_lookahead(klog_edge(Id),[Id]).\n",
                ]   
            },
            "aleph":{
                "node_only": [
                    ":- modeb(*,edge(+id_int,-id_int)).\n",
                    ":- modeb(*,node(+id,-color)).\n",
                    ":- modeb(*,green(+color)).\n",
                    ":- modeb(*,red(+color)).\n",
                    ":- determination(cyclic/1,edge/2).\n",
                    ":- determination(cyclic/1,node/2).\n",
                    ":- determination(cyclic/1,green/1).\n",
                    ":- determination(cyclic/1,red/1).\n",
                ],
                "node_edge": [
                    ":- aleph_set(clauselength,15).\n",
                    ":- modeb(*,edge(+id,-id)).\n",
                    ":- modeb(*,node(+id,-color)).\n",
                    ":- modeb(*,green(+color)).\n",
                    ":- modeb(*,red(+color)).\n",
                    ":- determination(cyclic/1,edge/2).\n",
                    ":- determination(cyclic/1,node/2).\n",
                    ":- determination(cyclic/1,green/1).\n",
                    ":- determination(cyclic/1,red/1).\n",
                ],
                "edge_based": [
                    ":- modeb(*,edge(+id,-id)).\n",
                    ":- modeb(*,node(+id,-color)).\n",
                    ":- modeb(*,green(+color)).\n",
                    ":- modeb(*,red(+color)).\n",
                    ":- modeb(*,instance(+id)).\n",
                    ":- determination(cyclic/1,edge/2).\n",
                    ":- determination(cyclic/1,node/2).\n",
                    ":- determination(cyclic/1,green/1).\n",
                    ":- determination(cyclic/1,red/1).\n",
                    ":- determination(cyclic/1,instance/1).\n",
                ],
                "Klog": [
                    ":- modeb(*,edge(+id,-id)).\n",
                    ":- modeb(*,node(+id,-color)).\n",
                    ":- modeb(*,green(+color)).\n",
                    ":- modeb(*,red(+color)).\n",
                    ":- modeb(*,klog_edge(+id)).\n",
                    ":- determination(cyclic/1,edge/2).\n",
                    ":- determination(cyclic/1,node/2).\n",
                    ":- determination(cyclic/1,green/1).\n",
                    ":- determination(cyclic/1,red/1).\n",
                    ":- determination(cyclic/1,klog_edge/1).\n",
                ]
            },
            "popper":{
                "node_only": [
                    "body_pred(edge,2).\n",
                    "body_pred(node,2).\n",
                    "body_pred(green,1).\n",
                    "body_pred(red,1).\n",
                    "type(cyclic,(id,)).\n",
                    "type(edge,(id_int,id_int)).\n",
                    "type(node,(id,color)).\n",
                    "type(green,(color,)).\n",
                    "type(red,(color,)).\n",
                    "direction(edge,(in,out)).\n",
                    "direction(node,(in,out)).\n",
                    "direction(green,(out,)).\n",
                    "direction(red,(out,)).\n",
                ],
                "node_edge": [
                    "body_pred(edge,2).\n",
                    "body_pred(node,2).\n",
                    "body_pred(green,1).\n",
                    "body_pred(red,1).\n",
                    "type(cyclic,(id,)).\n",
                    "type(node,(id,color)).\n",
                    "type(edge,(id,id)).\n",
                    "type(green,(color,)).\n",
                    "type(red,(color,)).\n",
                ],
                "edge_based": [
                    "body_pred(edge,2).\n",
                    "body_pred(node,2).\n",
                    "body_pred(green,1).\n",
                    "body_pred(red,1).\n",
                    "body_pred(instance,1).\n",
                    "type(cyclic,(id,)).\n",
                    "type(node,(id,color)).\n",
                    "type(edge,(id,id)).\n",
                    "type(instance,(id,)).\n",
                    "type(green,(color,)).\n",
                    "type(red,(color,)).\n",
                    "direction(cyclic,(in,)).\n",
                    "direction(node,(in,out)).\n",
                    "direction(edge,(in,out)).\n",
                    "direction(green,(out,)).\n",
                    "direction(red,(out,)).\n",
                    "direction(instance,(out,)).\n",
                ],
                "Klog": [
                    "body_pred(edge,2).\n",
                    "body_pred(node,2).\n",
                    "body_pred(green,1).\n",
                    "body_pred(red,1).\n",
                    #"body_pred(klog_edge,1).\n",
                    "type(cyclic,(id,)).\n",
                    "type(node,(id,color)).\n",
                    "type(edge,(id,id)).\n",
                    #"type(klog_edge,(id,)).\n",
                    "direction(cyclic,(in,)).\n",
                    "direction(node,(in,out)).\n",
                    "direction(edge,(in,out)).\n",
                    "direction(green,(out,)).\n",
                    "direction(red,(out,)).\n",
                    #"direction(klog_edge,(out,)).\n",
                ]
            }
        }

# dataset = "cyclic"
# for graph_size in [16,20,24]:
    
#     output_path = os.path.join("docker","Experiment",dataset,"relational")   
#     generate_relational_data(output_path=output_path,min_graph=graph_size,max_graph=graph_size,num_graphs=200)
    
#     experiment_runner = Run(
#         dataset,
#         ["node_edge","edge_based","Klog"],
#         "class",
#         "id",
#         toGraph,
#         toLogic,
#         ilpSettings,
#         "Experiment",
#         graph_size,
#         split_data=True
#     )
#     experiment_runner.TILDE = True
#     experiment_runner.ALEPH = True
#     experiment_runner.POPPER = True
#     experiment_runner.GNN = True

#     experiment_runner.run()



### Relational_complexity -> changing graph size

dataset = "cyclic"
for graph_size in [5,10,15,20,25,30]:
    
    output_path = os.path.join("docker","Experiment",dataset,"relational")   
    generate_relational_data(output_path=output_path,min_graph=graph_size,max_graph=graph_size,num_graphs=150,n_color=2)
    
    experiment_runner = Run(
        dataset,
        ["node_edge","edge_based","Klog"],
        "class",
        "id",
        toGraph,
        toLogic,
        ilpSettings,
        "Experiment",
        graph_size,
        split_data=True
    )
    experiment_runner.TILDE = True
    experiment_runner.ALEPH = True
    experiment_runner.POPPER = True
    experiment_runner.GNN = True

    experiment_runner.run()





# def change_colors(relational_path,n_colors):
#     nodes = pd.read_csv(os.path.join(relational_path,"nodes.csv"))
#     colors = ["red","green","blue","yellow"]
#     colors = colors[:n_colors]
#     # take the color column and assign random colors to the nodes
#     nodes["color"] = nodes["color"].apply(lambda x: colors[random.randint(0,len(colors)-1)])
#     nodes.to_csv(os.path.join(relational_path,"nodes.csv"),index=False)
#     return


# cycle_size = 4
# dataset = "cyclic"
# graph_size = 15
# # generate the data once and when adding a color go over randomly and change colors but the graphs must be the same, only changing the colors
# output_path = os.path.join("docker","Experiment",dataset,"relational")   
# generate_relational_data(output_path=output_path,min_graph=graph_size,max_graph=graph_size,dataset_name=dataset,num_graphs=100,cycle_size=cycle_size,n_color=1)
# # original data is generated with a single color
# for n_color in [1,2,3,4]:
#     if n_color == 1:
#         pass
#     else:
#         change_colors(output_path,n_color)

#     experiment_runner = Run(
#         dataset,
#         ["node_only","node_edge","edge_based","Klog"],
#         "class",
#         "id",
#         toGraph,
#         toLogic,
#         ilpSettings,
#         "Experiment",
#         n_color,
#         split_data=True
#     )

#     experiment_runner.TILDE = True
#     experiment_runner.ALEPH = True
#     experiment_runner.POPPER = False
#     experiment_runner.GNN = True

#     experiment_runner.run()



### changing cycle sizes

# cycle_size = 4
# dataset = "cyclic"
# graph_size = 15
# # generate the data once and when adding a color go over randomly and change colors but the graphs must be the same, only changing the colors
# output_path = os.path.join("docker","Experiment",dataset,"relational")   
# generate_relational_data(output_path=output_path,min_graph=graph_size,max_graph=graph_size,dataset_name=dataset,num_graphs=100,cycle_size=cycle_size,n_color=1)
# # original data is generated with a single color
# for n_color in [1,2,3,4]:
#     if n_color == 1:
#         pass
#     else:
#         change_colors(output_path,n_color)

#     experiment_runner = Run(
#         dataset,
#         ["node_only","node_edge","edge_based","Klog"],
#         "class",
#         "id",
#         toGraph,
#         toLogic,
#         ilpSettings,
#         "Experiment",
#         n_color,
#         split_data=True
#     )

#     experiment_runner.TILDE = True
#     experiment_runner.ALEPH = True
#     experiment_runner.POPPER = False
#     experiment_runner.GNN = True

#     experiment_runner.run()