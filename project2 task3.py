import networkx as nx # v2.4
import matplotlib.pyplot as plt
import collections
import numpy as np
from datetime import datetime
import time


Graph = nx.DiGraph() # Directed Graph for the emails

# Reads the edges in from the dataset
with open('532projectdataset.txt', 'r') as data:
    # Loops through each line in the file
    
    # This is a middle data in a dictionary in the form of {(u,v): weight, ...}
    # we'll prepare this middle data first and then make a weighted edge list for netwokx to read from
    # although this is more verbose than try and except, it has better running time
    weighted_edges = {} 

    # Process raw data
    for line in data.readlines():
        line = line.rstrip() # Removes trailing whitespace
        
        # we know every row is in the form (timestamp, email1, email2), so we unpack the variables
        timestamp, u, v = line.split(' ') # Splits line into a list on spaces
        
        datetimeObj = datetime.fromtimestamp(int(timestamp)) # filter out weekend datas
       
        if datetimeObj.strftime('%w') in {'0','6'}: # if the timestamp is in Sunday or Saturday
            continue 

        if (u,v) not in weighted_edges: # if we haven't seen this edge before, weight is 1
            weighted_edges[(u,v)] = 1
        else:                           # otherwise, increment the weight
            weighted_edges[(u,v)] += 1
    
    # Now we're preparing the weighted edge list for networkx to read from
    weighted_edge_list = [] # each item in this list will be in the form (u , v , weight)
    for edge in weighted_edges.keys():
        weighted_edge_list.append((edge[0],edge[1],weighted_edges[edge]))
    
    Graph.add_weighted_edges_from(weighted_edge_list)
#####################################################################################
# Task 3:                                                                           #
#   Consider a temporal representation of the Graph (i.e. a sequence of graph       #
#   snapshots) G ={G_1, G_2, ..., G_t}, where V = U_t(V_t) and E = U_t(E_t), where  #
#   t = 1, ..., t. Each snapshot G_t = (V_t, E_t) represents a dailt sample         #
#   i.e. (excluding weekends), where a directed weighted edge e_uv denotes the      #
#   number of emails senf from node u to node v over that sample                    #
#####################################################################################
class TemporalGraph:
    def __init__(self):
        self.graph = nx.Graph()

strong_componet_sizes = list() # list of the strong componet sizes
weak_componet_sizes = list() # list of the eak componete sizes
densities = list() # list of the densites of each temporal graph
clusterings = list() # list of the clustering coieficents for each temporal graph
G_t = nx.DiGraph() # The directed temporal graph

# Reads the edges in from the dataset file
with open('532projectdataset.txt', 'r') as dataset:
    # Reads in the file line by line
    day = None
    for line in dataset.readlines():
        line = line.rstrip() # removes white space
        line = line.split(' ') # splits the line into a list on spaces

        ts = int(line[0])
        newDay = datetime.utcfromtimestamp(ts).strftime('%d')

        if day == None:
            day = newDay

        # Reads in 724867 lines at a time
        if day == newDay:
            # Same stratagy as before for counting the number of times a edge appears in the dataset
            try:
                G_t[line[1]][line[2]]['weight'] += 1
            except:
                G_t.add_edge(line[1], line[2], weight=1)

        # When 724867 lines have been read fromt he file
        else:
            # Gets the size of the largest strongly connected componete and largest weakly connected componet in the temporal graph
            # adds the componet sizes to the resective list
            strong_componet_sizes.append(len(max(nx.strongly_connected_components(G_t), key=len))) 
            weak_componet_sizes.append(len(max(nx.weakly_connected_components(G_t), key=len)))

            # Gets the density and the average clustering coeifcent for the currant temporal graph
            # Adds them to the respective lists
            densities.append(nx.density(G_t))
            clusterings.append(nx.average_clustering(G_t))

            # Makes a new grpah for the next temporal graph
            # Adds the current edge to it; defualt weight set to 1
            # re-intialzies the count to 1
            G_t = nx.DiGraph()
            G_t.add_edge(line[1], line[2], weight=1)
            day = None

# Computes the componte sizes, density, and average clustering coeificent for the last temporal graph
strong_componet_sizes.append(len(max(nx.strongly_connected_components(G_t), key=len)))
weak_componet_sizes.append(len(max(nx.weakly_connected_components(G_t), key=len)))
densities.append(nx.density(G_t))
clusterings.append(nx.average_clustering(G_t))

#####################################################################################
# Plot the evoultion of the size of the largest connected componet (both in the weak#
# and the strong sense) as a function of time (mesaured in days). In a separte      #
# figure, plot the density of each G_t, over t.                                     #
# TODO:                                                                             #
#   Do you observe a "densification law" or an "undensification" trend?             #
#####################################################################################

# Plots the stong and weak compnent sizes for each temporal graph
plt.plot(range(len(strong_componet_sizes)), weak_componet_sizes, label='Weakly Connected Componet')
plt.plot(range(len(strong_componet_sizes)), strong_componet_sizes, label='Strongly Connected Componet')
plt.xlabel('Days')
plt.ylabel('Largest Componet Size')
plt.legend()
plt.show()

# Plots the density of each temporal Graph
plt.plot(range(len(strong_componet_sizes)), densities)
plt.xlabel('Days')
plt.ylabel('Density')
plt.show()

#####################################################################################
# Repeat the step above for the average clustering coefficent of each G_t.          #
# TODO:                                                                             #
#   What do you observe? What does an increasing (similary, decreasing) trend of    #
#   clustering coefficients mean for changes to the structure of the network over   #
#   time.                                                                           #
#####################################################################################

# Plots the average clustering coeifecints for each temporal graph.
plt.plt(range(len(strong_componet_sizes)), clusterings)
plt.xlabel('Days')
plt.ylabel('Clustering')
plt.show()

#####################################################################################
# TODO Extra Credit:                                                                #
#   Given G, identify time points 2 <= phi <= T such that G_p differs signifigantly #
#   from G_p-1. Your approach should rank the time points in descending order of    #
#   Signifigance in the identified change between graph snapshots. You may use any  #
#   algorithm in the literature (in which case proper citation is required to       #
#   recognize the author(s) of the algorithm), or develop an algorithm of your own  #
#####################################################################################