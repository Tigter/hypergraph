# HyperGraph and EnzRank Project
## Project Structure
This project is designed to facilitate the training and evaluation of HyperGraph and EnzRank models. Below is an overview of the project structure and the purpose of each file and directory:

Core Files and Directories
1. hyper_graph_brenda.py: 
This is the main entry point for training the HyperGraph model. It contains the necessary code to initialize and train the HyperGraph model.

2. enzrank_model/EnzRank.py: 
This file contains the implementation of the EnzRank model. It includes the model architecture, training procedures, and evaluation metrics.

3. enzrank_model/run.sh: 
This is a shell script for running the EnzRank model. It automates the process of setting up the environment and executing the model training and evaluation.

4. enzrank_model/result_any/: 
This directory includes the results of joint testing for the EnzRank model. It contains various output files and logs generated during the testing process.

5.sh/hyper_graph.sh
This shell script contains the commands for training the HyperGraph model. It simplifies the process of executing the training pipeline by providing a single command to run.


6.sh/rotate.sh: 
This shell script contains the commands for training the KGE (Knowledge Graph Embedding) baseline model. It provides a straightforward way to execute the training process for the KGE baseline.

7. config/
This directory contains configuration files for both the HyperGraph and KGE baseline models. These files include various parameters and settings required for training.