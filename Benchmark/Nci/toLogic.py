import torch


class toLogic:

    def __init__(self,dataset_name,relational_path,problem_key) -> None:
        self.dataset_name = dataset_name
        self.symbol_to_number = {
                        "O": 0,
                        "N": 1,
                        "C": 2,
                        "S": 3,
                        "Cl": 4,
                        "P": 5,
                        "F": 6,
                        "Na": 7,
                        "Sn": 8,
                        "Pt": 9,
                        "Ni": 10,
                        "Zn": 11,
                        "Mn": 12,
                        "Br": 13,
                        "Cu": 14,
                        "Co": 15,
                        "Se": 16,
                        "Au": 17,
                        "Pb": 18,
                        "Ge": 19,
                        "I": 20,
                        "Si": 21,
                        "Fe": 22,
                        "Cr": 23,
                        "Hg": 24,
                        "As": 25,
                        "B": 26,
                        "Ga": 27,
                        "Ti": 28,
                        "Bi": 29,
                        "Y": 30,
                        "Nd": 31,
                        "Eu": 32,
                        "Tl": 33,
                        "Zr": 34,
                        "Hf": 35,
                        "In": 36,
                        "K": 37,
                        "La": 38,
                        "Ce": 39,
                        "Sm": 40,
                        "Gd": 41,
                        "Dy": 42,
                        "U": 43,
                        "Pd": 44,
                        "Ir": 45,
                        "Re": 46,
                        "Li": 47,
                        "Sb": 48,
                        "W": 49,
                        "Mg": 50,
                        "Ru": 51,
                        "Rh": 52,
                        "Os": 53,
                        "Th": 54,
                        "Mo": 55,
                        "Nb": 56,
                        "Ta": 57,
                        "Ag": 58,
                        "Cd": 59,
                        "Er": 60,
                        "V": 61,
                        "Ac": 62,
                        "Te": 63,
                        "Al": 64,
                    }
        self.number_to_symbol = {v: k for k, v in self.symbol_to_number.items()}

    def createNodeIds(self,n):
        return {i:'n'+str(i) for i in range(n)}
    
    
    def node_only(self,graphs,output_path):
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
                node_id = nodeIds[j]
                if not (node[:4] == 0.0).all().item():
                    relation = "drug"
                    example += f"{relation}({problem_id},{node_id},{node[0]},{node[1]},{node[2]},{node[3]}).\n"
                elif not (node[4] == 0.0).all().item():
                    relation = "atom"
                    example += f"{relation}({problem_id},{node_id},{node[4]}).\n"
                elif not (node[5] == 0.0).all().item():
                    relation = "bond"
                    example += f"{relation}({problem_id},{node_id},{node[5]}).\n"
                elif not torch.all(node[6]):
                    relation = "nitro"
                    example += f"{relation}({problem_id},{node_id}).\n"
                elif not torch.all(node[7]):
                    relation = "benzene"
                    example += f"{relation}({problem_id},{node_id}).\n"  
            
            # add all the edges between all nodeIds
            for m in nodeIds:
                for n in nodeIds:
                    if m != n:
                        example += f"edge({problem_id},{m},{n}).\n"
            examples.append(example)
        #file = self.exp_path + "/logic/node_only.pl"
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return
    
    def node_edge(self,graphs,output_path):
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

            # add all atom relations
            for j,node in enumerate(graph.x):
                node_id = nodeIds[j]
                index = torch.nonzero(node).item()
                relation = "atom"
                example += f"{relation}({problem_id},{node_id},{self.number_to_symbol[index].lower()}).\n"
            
            # add all the bonds from the edge_index and edge features
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    node_id1 = nodeIds[edge[0]]
                    node_id2 = nodeIds[edge[1]]
                    edge_feat = graph.edge_attr[j]
                    bond_type = torch.nonzero(edge_feat).item() + 1
                    relation = "bond"
                    example += f"{relation}({problem_id},{node_id1},{node_id2},{bond_type}).\n"
            
            
            examples.append(example)
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return

    def edge_based(self, graphs, output_path):
        examples = []
        for i, graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"

        return

    def Klog(self,graphs,output_path):
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            # add all the 

        return



                