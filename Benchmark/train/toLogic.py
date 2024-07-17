import torch


class toLogic:

    def __init__(self,dataset_name,relational_path,problem_key) -> None:
        self.dataset_name = dataset_name

    def createNodeIds(self,n):
        return {i:'n'+str(i) for i in range(n)}
    
    def node_only(self, graphs, output_path):
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            for j,node in enumerate(graph.x):
                if j == 0:
                    continue
                node_id = nodeIds[j]
                if node[1] == 1.0:
                    relation = "short"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[2] == 1.0:
                    relation = "long"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[3] == 1.0:
                    relation = "two_wheels"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[4] == 1.0:
                    relation = "three_wheels"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[5] == 1.0:
                    relation = "roof_open"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[6] == 1.0:
                    relation = "roof_closed"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[7] == 1.0:
                    relation = "zero_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[8] == 1.0:
                    relation = "one_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[9] == 1.0:
                    relation = "two_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[10] == 1.0:
                    relation = "three_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[11] == 1.0:
                    relation = "circle"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[12] == 1.0:
                    relation = "triangle"
                    example += f"{relation}({problem_id},{node_id}).\n"
            
            # the edge indices are in relations between nodeIds
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j%2 == 0:
                    example += f"edge({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"
                
            examples.append(example)

        #file = self.exp_path + "/logic/edge_based.pl"
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return
    
    def node_edge(self, graphs, output_path):
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            for j,node in enumerate(graph.x):
                if j == 0:
                    continue
                node_id = nodeIds[j]
                if node[1] == 1.0:
                    relation = "short"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[2] == 1.0:
                    relation = "long"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[3] == 1.0:
                    relation = "two_wheels"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[4] == 1.0:
                    relation = "three_wheels"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[5] == 1.0:
                    relation = "roof_open"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[6] == 1.0:
                    relation = "roof_closed"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[7] == 1.0:
                    relation = "zero_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[8] == 1.0:
                    relation = "one_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[9] == 1.0:
                    relation = "two_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[10] == 1.0:
                    relation = "three_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[11] == 1.0:
                    relation = "circle"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[12] == 1.0:
                    relation = "triangle"
                    example += f"{relation}({problem_id},{node_id}).\n"
            
            # the edge indices are in relations between nodeIds
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j%2 == 0:
                    if graph.edge_attr[j][0] == 1.0:
                        relation = "has_car"
                        example += f"{relation}({problem_id},{nodeIds[edge[1]]}).\n"
                    if graph.edge_attr[j][1] == 1.0:
                        relation = "has_load"
                        example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"
                
            examples.append(example)

        #file = self.exp_path + "/logic/edge_based.pl"
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return

    def edge_based(self, graphs, output_path):
        examples = []
        fact_mapping = {0:"train",1:"car",2:"load",3:"short",4:"long",5:"two_wheels",6:"three_wheels",7:"roof_open",8:"roof_closed",9:"zero_load",10:"one_load",11:"two_load",12:"three_load",13:"circle",14:"triangle"}
        edge_mapping = {}
        fact_exam = ""
        for fact in fact_mapping.values():
            fact_exam += f"{fact}({fact}).\n"
        examples.append(fact_exam)
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            for j,node in enumerate(graph.x):
                if j <= 14:
                    pass
                    #example += f"{fact_mapping[j]}.\n"
                else:
                    node_id = nodeIds[j]
                    relation = "instance"
                    example += f"{relation}({problem_id},{node_id}).\n"
                    # add possible relations
                    # non_zero = torch.nonzero(node)
                    # for k in non_zero:
                    #     relation = "edge"
                    #     example += f"{relation}({problem_id},{node_id},{fact_mapping[k.item()]}).\n"
        
            # the edge indices are in relations between nodeIds
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j%2 == 0:
                    if graph.edge_attr[j][0] == 1.0:
                        # relation is edge between the node and the fact
                        relation = "edge"
                        if edge[0] not in fact_mapping:
                            example += f"{relation}({problem_id},{nodeIds[edge[0]]},{fact_mapping[edge[1]]}).\n"
                        else:
                            example += f"{relation}({problem_id},{nodeIds[edge[1]]},{fact_mapping[edge[0]]}).\n"

                    if graph.edge_attr[j][-2] == 1.0:
                        relation = "has_car"
                        # has_car(problem_id,train_id,car_id)
                        example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"
                    if graph.edge_attr[j][-1] == 1.0:
                        relation = "has_load"
                        # has_load(problem_id,car_id,load_id)
                        example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"

                
            examples.append(example)

        #file = self.exp_path + "/logic/edge_based.pl"
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return

    def Klog(self, graphs, output_path):   
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            for j,node in enumerate(graph.x):
                if j == 0:
                    continue
                node_id = nodeIds[j]
                if node[1] == 1.0:
                    relation = "short"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[2] == 1.0:
                    relation = "long"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[3] == 1.0:
                    relation = "two_wheels"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[4] == 1.0:
                    relation = "three_wheels"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[5] == 1.0:
                    relation = "roof_open"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[6] == 1.0:
                    relation = "roof_closed"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[7] == 1.0:
                    relation = "zero_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[8] == 1.0:
                    relation = "one_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[9] == 1.0:
                    relation = "two_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[10] == 1.0:
                    relation = "three_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[11] == 1.0:
                    relation = "circle"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[12] == 1.0:
                    relation = "triangle"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[13] == 1.0:
                    relation = "has_car"
                    example += f"{relation}({problem_id},{node_id}).\n"
                if node[14] == 1.0:
                    relation = "has_load"
                    example += f"{relation}({problem_id},{node_id}).\n"
            
            # the edge indices are in relations between nodeIds
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j%2 == 0:
                    relation = "edge"
                    example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"
                
            examples.append(example)

        #file = self.exp_path + "/logic/edge_based.pl"
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return
