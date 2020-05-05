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
# Task 2:                                                                           #
#   Extract the 1.5 egonetwork, G_u, of every node u in G. Note that for the        #
#   purposes of this task, we will treat G_u as undirected                          #
#####################################################################################

# Task 2 --1
#####################################################################################
# Compute the total number of nodes, V_u, edges, E_u, and total weight, W_u, for    #
# each G_u. Also compute the principal eigenvalue, lamda_wu, of the weighted        #
# adjancency matrix of each egonet G_u.                                             #
#####################################################################################
print(f"\n\n{'《 T A S K  2 》 B E G I N S ':=^52}")
node_counts = list() # List of total number of nodes in each egonet
edge_counts = list() # List of total number of edges in each egonet
weight_totals = list() # List of total weights for each egonet
eigenvalues = list() # List of max egienvalues for each egonet
def compute_1half_egonet():
    for n in Graph:
        pass
        
# Loops through each node in the Graph tp construct the 1.5 egonet for the node
for node in Graph:
    G_u = nx.ego_graph(Graph, node, 1, undirected=True) # Gets the 1 egonet for the node
    G_u = G_u.to_undirected() # changes the egonet from directed to undirected

    # Loops over each node in the egonet twice to check for the exsitence of edges between
    # nodes in the egonet inorder to compute the actual 1.5 egonet
    # as oppose to the 1 egonet computed above
    for n_u in G_u:
        # Dont need to check the node the egonet is made for
        if n_u == node:
            continue
        for v_u in G_u:
            if v_u == node:
                continue
            # if the nodes are the same continue
            if n_u == v_u:
                continue
            # if the edge (n_u, v_u) is in the orginal Graph then add it to the egonet
            if (n_u, v_u) in Graph.edges:
                G_u.add_edge(n_u, v_u, weight=Graph[n_u][v_u]['weight'])

    # Calculates the total weight of the egonet by summing the weights in the edges together
    weight_total = 0
    # Loops over eges in the egonet
    for edge in G_u.edges:
        # Sums the weights of the edges
        weight_total += G_u[edge[0]][edge[1]]['weight']
    
    # Adds the number fo nodes, edges, total weigths and max eigenvalue to their respective lists
    node_count = G_u.number_of_nodes() # Gets number of nodes in the egonet

    # TODO:
    #   Only computes the eigenvalue when the egonet has more than one node
    #   If the ego net only has one node it dose not have an edge and thus no Adjacency matrix to use to compute the eigenvalue
    #   selects the max eigenvalue if edges exstis, otherwise sets the eigenvalue to 0
    #   I am not sure if this method is correct or not
    if node_count > 1:
        node_counts.append(node_count)
        edge_counts.append(G_u.number_of_edges())
        weight_totals.append(weight_total)

        L = nx.normalized_laplacian_matrix(G_u) # Computes the adjacency matrix
        e = np.linalg.eigvals(L.A) # Gets the egienvalues of the adjacency matrix

        # Gets the max eginvalue of the Adjaceny matrix
        # Adds it to the eigenvalues list
        eigenvalues.append(sum(e))
# print(duration(t))
breakpoint()
# Task 2 --2
#####################################################################################
# Plot on a log-log scale:                                                          #
#   (i) E_u versus V_u for every egonet G_u                                         #
#   (ii) the least squares fit on the median values for each bucket of points after #
#   applying logairthmic binning on the x-axis                                      #
#   (iii) two lines of slope 1 and 2, the correspond to the stars and cliques       #
#   respectivly.                                                                    #
#   Additionally, plot (on a separate figure) the value of lamda_wu versus W_u for  #
#   each egonet G_u on a log-log scale.                                             #
#   TODO:                                                                           #
#       Can you fit a power law (e.g. E_u = V_u^alpha)? If so, what is the value of #
#       the power law exponet alpha in each case?                                   #
#####################################################################################

# TODO:
#   Plots E_u vs V_u
#   This may be backwards. i.e. should be edges on the x-axis.
#   I went with this way beacause it just looks nicer on the graph
plt.figure(2,[10,5])
plt.loglog(node_counts, edge_counts, 'o', markersize=5)

# Gets slope and y intercept for the line of best fit
m, c = np.polyfit(np.log(node_counts), np.log(edge_counts), 1)
edge_fit = np.exp(m*np.log(node_counts) + c) # computes the line of best fit
plt.loglog(node_counts, edge_fit, label='m=' + f'{m:.5}' + ',c=' + f'{c:.5}') # Plots the line of best fit

# TODO:
#   Computes the line to represent the stars and cliques
#   I have no cluse what this question was asking in this regard so i just used
#   the slope given in the prompt (1 for starts and 2 for cliques) along with
#   the y intercept computed from the line fo best fit to plot the line
stars = np.exp(1*np.log(node_counts) + c)
cliques = np.exp(2*np.log(node_counts) + c)
plt.loglog(node_counts, stars,'--', label='Stars')
plt.loglog(node_counts, cliques,'--', label='Cliques')

# Does not display plot yet 
# Have to calculate the "out-of-norm" nodes first
plt.ylabel('Edges')
plt.xlabel('Nodes')


# Task 2 --3
#####################################################################################
# Let x_u and y_u denote the observed x and y value for a node u respectivly for a  #
# particular feature pair f(x, y). Given the power law equation y = Cx^a for f(x, y)#
# let the "out-of-the-norm" score u to be                                           #
# o(u) = max(y_u, Cx_u^a)/min(y_u, Cx_u^a)log(|y_u - Cx_u^a|+1). Intuitively, this  #
# metric penalizes each node with both the number of times that y_u deviates from   #
# its expected value Cx_u^a given x_u, and with the logarithm of the amount of      #
# deviation. Thus, o(u) = 0 when y_u is equal to the expected value Cx_u^a. Compute #
# o(u) for all nodes in the Graph for the case of f(V_u, E_u) and f(W_u, lamda_wu)  #
# and sort the nodes according to their scores. Highlight the top 20                #
# "out-of-the-norm" nodes by marking them as triangles in the corresponding plots   #
#####################################################################################

# Calculates o(u) for each node in the graph using the egonets computed before 

# List of dicts containing info on the o(u) of the egonet
# Each dict in the list represents an egonet
o_u_list = list(dict()) 

# Loops over the node and edge counts computed from before for each egonet
for i in range(len(node_counts)):
    x_u = node_counts[i] # x_u for the power law equation
    y_u = edge_counts[i] # y_u for the power law equation

    # TODO:
    #   Computes expected value of y_u using the power law equeation
    #   I assumed the alpha was the slope from the best fit line
    #   and that C was the y intercept from the best fit line
    #   but I am not 100% sure
    c_x = c*(x_u**m)

    o_u_dict = dict() # egonet o(u) info; goes in the o(u) list

    # Computes o(u)
    o_u = (max(y_u, c_x) / min(y_u, c_x)) * np.log(abs(y_u - c_x) + 1)

    # Stores the vlaues needed to plot into the o_u_dict
    o_u_dict['o_u'] = o_u
    o_u_dict['x_u'] = x_u
    o_u_dict['y_u'] = y_u

    # adds the o_u_dict for the egonet to the list of o(u)'s for each egonet
    o_u_list.append(o_u_dict)

# Sorts the list of o(u) dicts in decesnding order by the value o_u in each dict
o_u_list_sorted = sorted(o_u_list, key=lambda x: x['o_u'], reverse=True)
o_u_list = o_u_list_sorted[:20] # takes the top 20 o(u) scores

# Creates lists of the actual y_u values and the given x_u values 
# to be used for ploting from the top 20 o(u) scores
y_u_data = list() 
x_u_data = list()
# Loops through top 20 ego nets o(u) info
for data in o_u_list:
    # Adds the actual y_u to the list of y_u data
    # Adds the x_u of the ego net to the list of x_u data
    y_u_data.append(data['y_u'])
    x_u_data.append(data['x_u'])

# Highlights the top scoring o(u) nodes with triangles 
plt.loglog(x_u_data, y_u_data, '^', label='Top 20 o(u) scores')
plt.legend()
plt.show()
breakpoint()
# TODO:
#   This graph does not seem to have a power distrobution
#   Thus, you can not compute the o(u) score for it
#   I Have no cluse if this is correct My methods or understanding could be wrong
m, c = np.polyfit(np.log(weight_totals), np.log(eigenvalues), 1)
weight_fit = np.exp(m*np.log(weight_totals) + c)
plt.loglog(weight_totals, eigenvalues, 'o')
plt.loglog(weight_totals, weight_fit, label='m=' + str(m) + ',c=' + str(c))
plt.ylabel('Eigen Values')
plt.xlabel('weight_totals')

o_u_list = list(dict()) 

# Loops over the node and edge counts computed from before for each egonet
for i in range(len(node_counts)):
    x_u = weight_totals[i] # x_u for the power law equation
    y_u = eigenvalues[i] # y_u for the power law equation

    # TODO:
    #   Computes expected value of y_u using the power law equeation
    #   I assumed the alpha was the slope from the best fit line
    #   and that C was the y intercept from the best fit line
    #   but I am not 100% sure
    c_x = c*(x_u**m)

    o_u_dict = dict() # egonet o(u) info; goes in the o(u) list

    # Computes o(u)
    o_u = (max(y_u, c_x) / min(y_u, c_x)) * np.log(abs(y_u - c_x) + 1)

    # Stores the vlaues needed to plot into the o_u_dict
    o_u_dict['o_u'] = o_u
    o_u_dict['x_u'] = x_u
    o_u_dict['y_u'] = y_u

    # adds the o_u_dict for the egonet to the list of o(u)'s for each egonet
    o_u_list.append(o_u_dict)

# Sorts the list of o(u) dicts in decesnding order by the value o_u in each dict
o_u_list_sorted = sorted(o_u_list, key=lambda x: x['o_u'], reverse=True)
o_u_list = o_u_list_sorted[:20] # takes the top 20 o(u) scores

# Creates lists of the actual y_u values and the given x_u values 
# to be used for ploting from the top 20 o(u) scores
y_u_data = list() 
x_u_data = list()
# Loops through top 20 ego nets o(u) info
for data in o_u_list:
    # Adds the actual y_u to the list of y_u data
    # Adds the x_u of the ego net to the list of x_u data
    y_u_data.append(data['y_u'])
    x_u_data.append(data['x_u'])

# Highlights the top scoring o(u) nodes with triangles 
plt.loglog(x_u_data, y_u_data, '^', label='Top 20 o(u) scores')
plt.legend()
plt.show()