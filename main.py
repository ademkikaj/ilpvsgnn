from runner import Run


def main():


    logic_datasets = ["krk","bongard","train","sameGen","cyclic"]
    real_datasets = ["mutag","nci","cancer","financial","PTC"]

    representations = ["node_only","node_edge","edge_based","Klog"]

    dataset = "cyclic"

    runner = Run(dataset_name=dataset,
                 representations=representations)
    
    runner.TILDE = False
    runner.ALEPH = True
    runner.POPPER = False
    runner.GNN = False
    runner.run()

if __name__ == "__main__":
    main()