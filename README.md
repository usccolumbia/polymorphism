This github repo is for our paper **Data-Driven Topological Analysis of Polymorphic Crystal Structures** </br>

- The datasets are available in the `dataset` folder:
    - `dataset_polymorphs_mp.csv`: Polymorph data from the Materials Project (MP).
    - `dataset_polymorphs_icsd.csv`: Polymorph data from the Inorganic Crystal Structure Database (ICSD).
- To find similar structures,
    - refer to the folder mapping_structures. Here, step1.ipynb is used to process polyhedral graphs of structures. The graph is already stored inside the dataset folder. If you want to construct graphs of your own dataset, then step1 will guide you.
  - Step2.ipynb file will guide you to insert structure of your choice and then will return ids of similar structures.
  - view_similar.ipynb should be used then to plot those similar structures.
- The oxidation state analysis code can be found in the folder oxidation_state.
- The space group pair analysis code can be found in the folder space_group_pair_analysis.
