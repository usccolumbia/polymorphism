This github repo is for our paper **Data-Driven Topological Analysis of Polymorphic Crystal Structures** </br>

- The dataset is available in dataset folder and its name is dataset.csv.
- To find similar structures,
  - refer to the folder mapping_structures. Here, step1.ipynb is used to process polyhedral graphs of structures. The graph is already stored inside dataset folder. If you want to construct graphs of your own dataset, then step1 will guide you.
  - Step2.ipynb file will guide you to insert structure of your choice and then will return ids of similar structures.
  - view_similar.ipynb should be used then to plot those similar structures.
- The oxidation state analysis code can be found in the folder oxidation_state.
- The space group pair analysis code can be found in the folder space_group_pair_analysis.
