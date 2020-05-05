import networkx as nx # v2.4
import matplotlib.pyplot as plt
import collections
import numpy as np
from datetime import datetime
import time

def duration(start_time: float) -> str:
    '''return a string format of a duration from the given time up til now.'''
    end_time = time.time() - start_time
    return f"{end_time:.5}"

#####################################################################################
# Task 1:                                                                           #
#   Constructs a weighted, directed graph G(V, E), out of all email correspondence  #
#   in the dataset. Nodes represent email addresses and directed edges depict sent  #
#   and recivied relations. in G, a directed edge e_uv denotes the number of emails #
#   sent from node u to node v over the entire dataset.                             #
#####################################################################################

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
        if datetimeObj.strftime('%w') in {'0','6'}: # if the timestamp is Sunday or Saturday
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
# Summary statistics:                                                               #
# TODO: filter out weekend datas
#   Number of nodes, edges, and bidrectional edges. In, out, and total degree for   #
#   each email and the diameter of the network.                                     #
#####################################################################################
def summary_statistics():
    print(f"\n\n{'《 T A S K  1 》 B E G I N S ':=^52}")
    print(f"{'[ Summary statistics ]':.<52}")
    # Gets number of nodes and edges
    # I store these values so i can use them later for calculations without needing to re-count
    number_of_nodes = Graph.number_of_nodes()
    number_of_edges = Graph.number_of_edges()
    print("- Number of nodes: ", number_of_nodes)
    print("- Number of edges: ", number_of_edges)

    # Counts the number of bidirectional edges
    count = 0
    for edge in Graph.edges:
        # For edge (u, v) if edge (v, u) exists then there is a bidirectional edge
        if Graph.number_of_edges(edge[1], edge[0]) > 0:
            count += 1

    # We have to divid the coutn by 2 because each bidirectional edge is counted twice.
    print("- Number of Bidirectional edges: ", count//2)

    # Calculates the in, out, and total degree for each node in the graph
    in_degree = Graph.in_degree()
    out_degree = Graph.out_degree()
    degree = Graph.degree()

    # Gets the min in, out, and total degree
    # I cast in_degree, out_degree, and degree to dict() so I can use the values() function to get a list of the values and easily get the min, max, and averages.
    min_in = min(dict(in_degree).values())
    min_out =  min(dict(out_degree).values())
    min_degree = min(dict(degree).values())

    # Gets the max in, out, and total degree
    max_in = max(dict(in_degree).values())
    max_out =  max(dict(out_degree).values())
    max_degree = max(dict(degree).values())

    # Gets the average in, out, and total degree
    average_in = sum(dict(in_degree).values()) / number_of_nodes
    average_out =  sum(dict(out_degree).values()) / number_of_nodes
    average_degree = sum(dict(degree).values()) / number_of_nodes

    # Print out a matrix for the above information
    print(f"\n{'[ MATRICE ]':.<52}")
    print(f"{'': ^10}|{'in degree': ^13}|{'out degree': ^13}|{'total degree': ^13}")
    print(f"{'':-<10}+{'':-<13}+{'':-<13}+{'':-<13}")
    print(f"{'min': <10}|{min_in: ^13}|{min_out: ^13}|{min_degree: ^13}")
    print(f"{'average': <10}|{average_in: ^13.6}|{average_out: ^13.6}|{average_degree: ^13.6}")
    print(f"{'max': <10}|{max_in: ^13}|{max_out: ^13}|{max_degree: ^13}")

# summary_statistics()
# breakpoint()
# TODO:
#   Trys to compute the diameter of the Graph
#   When I run this i get an error because there is a node that can not be reached from another node
#   Thus, the diameter of the Graph would be infite.
#   This may be wrong, I'm not really sure

def compute_diameter():
    '''
    this function will compute and print the diameter of the largest components
    '''
    s = time.time() # for timing purposes
    strongly_cc_nodes = max(nx.strongly_connected_components(Graph),key=len)
    scc_edges = list(filter(lambda item: item[0] in strongly_cc_nodes and item[1] in strongly_cc_nodes , weighted_edge_list))
    largest_scc = nx.DiGraph()
    largest_scc.add_weighted_edges_from(scc_edges)
    print(f"compute time: {duration(s)}")

# compute_diameter()

#####################################################################################
# Plot the distrobution of degrees, in-degrees, and out-degrees of the nodes in G   #
# in the same plot on a log-log sclae. For each of the three degree distobutions,   #
# draw the corresponding best fit least-square regression line in the same log-log  #
# plot. Show the coefficients of each fited line in the legened of the plot         #
#####################################################################################
def plot_degree_distributions():
    degree = Graph.degree()
    in_degree = Graph.in_degree()
    out_degree = Graph.out_degree()

    # Calculates the degree distobution
    degree_sequence = sorted([d for n, d in degree], reverse=True) # sorts the degrees in descinding order
    degreeCount = collections.Counter(degree_sequence) # counts the number of times a degree occurs
    del degreeCount[0] # Removes instances when the degree is 0
    deg, cnt = zip(*degreeCount.items()) # gets the list of the degrees and the corisponding count

    # Calculates the in degree distrobution
    in_sequence = sorted([d for n, d in in_degree], reverse=True)
    inCount = collections.Counter(in_sequence)
    del inCount[0]
    in_deg, in_cnt = zip(*inCount.items())

    # Calculates the out degree distrobution
    out_sequence = sorted([d for n, d in out_degree], reverse=True)
    outCount = collections.Counter(out_sequence)
    del outCount[0]
    out_deg, out_cnt = zip(*outCount.items())

    # Calculates the line of best fit for the total degree distrobution
    # Have to take the log of the degree and the count as the ploting will be done on a log-log scale
    m, c = np.polyfit(np.log(deg), np.log(cnt), 1) # gets the slope (m) and the y intercept (C) for the regression line
    deg_fit = np.exp(m*np.log(deg) + c) # gets the y' points for the regression line

    plt.figure(1, [10,5])
    # plots the degree distrobution
    plt.loglog(deg, cnt, color='#FF6666', label='Total Degree')
    plt.loglog(deg, deg_fit,'--', color='#FF9696', label=f"Regression line: log y = {m:+.3} log x + {c:.3}")

    # Calculates the line of best fit for the in degree distrobution
    m, c = np.polyfit(np.log(in_deg), np.log(in_cnt), 1)
    in_deg_fit = np.exp(m*np.log(in_deg) + c)

    # plots the in degree distrobution
    plt.loglog(in_deg, in_cnt, color='#668CFF', label='In Degree')
    plt.loglog(in_deg, in_deg_fit,'--', color='#A3BAFF', label=f"Regression line: log y = {m:+.3} log x + {c:.3} ")

    # Calculates the line of best fit for the out degree distrobution
    m, c = np.polyfit(np.log(out_deg), np.log(out_cnt), 1)
    out_deg_fit = np.exp(m*np.log(out_deg) + c)

    # plots the out degree distrobution
    plt.loglog(out_deg, out_cnt, color='#66FF8C', label='Out Degree')
    plt.loglog(out_deg, out_deg_fit,'--', color='#B3FF66', label=f"regression line: log y = {m:+.3} log x + {c:.3}")

    plt.title('Degree Distributions')
    plt.ylabel('Count')
    plt.xlabel('Degree')
    plt.legend()
    plt.ylim(ymin=1) #sets the min y value to 1 for the graph
    plt.show()

# plot_degree_distributions()

#####################################################################################
# TODO:                                                                             #
#   Compare the fitted power-law for the degree distrobution to the (i) exponential #
#   and (ii) log-normal distobutions. Indictae which model is a better fit, and     #
#   explain why.                                                                    #
#   I'm not really sure what this is asking and I don't think it is a programming   #
#   But more of a analysis question.                                                #
#####################################################################################

