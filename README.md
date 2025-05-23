# Scheduling_Graph

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Graph Theory](https://img.shields.io/badge/Graph_theory-Scheduling-FF6F00)
![NetworkX](https://img.shields.io/badge/NetworkX-2.6+-FFA500)

Analysis and scheduling of tasks using a table of constraints. Oriented graph construction, circuit detection, critical path calculation and visualisation


## 📝 Description

Task scheduling tool based on weighted oriented graphs, allowing:
- Automatic construction of graphs from constraints
- Analysis of structural properties
- Computation of an optimal schedule with identification of critical paths

## 🎯 Main objectives

- Read an array of constraints from a .txt file
- Construct a weighted directed graph based on the constraints
- Check that there are no circuits or negative values
Calculate:
- The rank of each task
- The earliest and latest schedule
- The margins
- The critical path(s)
- Graphically display the graph
- Generate detailed execution traces
  
## 📝 Project structure

1. Main files
   
main.py: Main program in user-interactive mode

main_trace.py: Version with detailed trace generation in .txt files

Graph.py: Main class containing all processing related to the graph, calculations and display

2. Data directories
   
Test_files/: Contains the constraint files to be tested (table 1.txt, table 2.txt, etc.)

Trace_files/: Contains the output files (traces) generated by main_trace.py

3. Implemented functions
   
Reading and displaying the constraints table

Creating the graph (adjacency matrix and value matrix)

Checking graph properties:

A single entry and exit point

Absence of circuits

Absence of arcs with negative values

Calculations:

Task ranks

Earliest/latest dates

Margins

Critical paths

Graphical display with matplotlib and networkx

Step-by-step trace generation (in main_trace. py)

## Directed by

SHANG Jacky

LEE Zhuo Chan Stive

LAZAUCHE Louis

THIRIOT-LESNE Matthew

AYETO Jonathan
