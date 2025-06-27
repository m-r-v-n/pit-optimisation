############################################################
# Pseudoflow Pit Optimiser
#
# This module requires the Pseudoflow library.
# Pseudoflow is licensed separately for non-commercial use only.
# See https://riot.ieor.berkeley.edu/Applications/Pseudoflow/maxflow.html
#
############################################################

import numpy as np
import pseudoflow as pf
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import sin, cos, tan, radians
from igraph import plot
import igraph as ig
from igraph import Graph
import os

# Sending nodes to Sink or Source
def sendNodes(BM, sink, source, g, BVal):
    Sink = sink
    Source = source
    rows = sink - 1
    start_UPL = time.time()

    edges = []
    consts = []
    mults = []

    for i in range(rows):
        node = i + 1
        capacity = np.abs(np.round(BM[i, BVal], 2))

        if BM[i, BVal] < 0:
            edges.append((node, Sink))
            consts.append(capacity)
            mults.append(-1)
            #print("connect to sink", "node:", node, "Sink", Sink, "Block Value",BM[i,BVal])
        else:
            edges.append((Source, node))
            consts.append(capacity)
            mults.append(1)
            #print("connect to source", "node:", node, "Source:", Source, "Block Value",BM[i,BVal])

    g.add_edges(edges)

    # Assign attributes only to newly added edges
    g.es[-len(edges):]["const"] = consts
    g.es[-len(edges):]["mult"] = mults

    assert all(e["const"] is not None for e in g.es), "Some edges have missing 'const'"
    assert all(e["mult"] is not None for e in g.es), "Some edges have missing 'mult'"
    print(f"Total vertices: {g.vcount()}")
    used_nodes = [e.source for e in g.es] + [e.target for e in g.es]
    print(f"Max node ID used: {max(used_nodes)}")

    print("--> Create external arc time: --%s seconds " % (np.round((time.time() - start_UPL), 2)))
    return g

# Arc Precedence
def createArcPrecedence(BM,
                        idx,
                        xsize,
                        ysize,
                        zsize,
                        xmin,
                        ymin,
                        zmin,
                        xmax,
                        ymax,
                        zmax,
                        xcol,
                        ycol,
                        zcol,
                        slopecol,
                        search_blocks_front,
                        search_blocks_back,
                        search_blocks_left,
                        search_blocks_right,
                        num_blocks_above,
                        g,
                        minWidth):

    start_UPL = time.time()

    BM1 = BM[:, [idx, xcol, ycol, zcol]]

    # Build spatial lookup dictionary
    block_to_value = {(x, y, z): value for value, x, y, z in BM1}

    # Collect edges and attributes for batch insert
    edges = []
    consts = []
    mults = []

    internal_arc = 0
    nodes_with_edge = 0

    BM2 = BM[:, [xcol, ycol, zcol,slopecol]]

    for i, (x_i, y_i, z_i, angle_i) in enumerate(BM2, start=1):
        min_radius = minWidth / 2  # Set to 0 to exclude same-elevation blocks, >0 to include them

        if z_i == zmax and min_radius == 0:
            continue

        cone_height = zsize * num_blocks_above
        cone_radius = cone_height / math.tan(math.radians(angle_i))

        # Generate 3D block search region
        x_range = range(-int(min(((x_i - xmin)/xsize), search_blocks_left)),
                        int(min(((xmax - x_i)/xsize)+1, search_blocks_right+1)))

        y_range = range(-int(min(((y_i - ymin)/ysize), search_blocks_front)),
                        int(min(((ymax - y_i)/ysize)+1, search_blocks_back+1)))

        # Always start from l = 0 so you can optionally include z = z_i
        z_range = range(0, int(min(((zmax - z_i)/zsize)+1, num_blocks_above+1)))

        block_coords = np.array([
            (x_i + j * xsize, y_i + k * ysize, z_i + l * zsize)
            for j in x_range
            for k in y_range
            for l in z_range
        ])

        if block_coords.size == 0:
            continue

        # Compute horizontal distances and vertical heights
        dists = np.sqrt((block_coords[:, 0] - x_i)**2 + (block_coords[:, 1] - y_i)**2)
        heights = block_coords[:, 2] - z_i

        # Compute cone radius per block, including optional base radius
        with np.errstate(divide='ignore', invalid='ignore'):
            cone_radii = (heights * cone_radius / cone_height)

        # If min_width > 0, allow a base disk radius at z = z_i
        if min_radius > 0:
            cone_radii = np.where(heights == 0, min_width, cone_radii + min_radius)

        # Get blocks inside the cone + base
        inside_indices = np.where(dists <= cone_radii)[0]
        inside_blocks = block_coords[inside_indices]

        connected = 0

        for block in inside_blocks:
            block_key = (block[0], block[1], block[2])
            source_key = (x_i, y_i, z_i)

            if block_key not in block_to_value or source_key not in block_to_value:
                continue

            target = int(block_to_value[block_key])
            source = int(block_to_value[source_key])

            if source == target:
                continue  # skip self-loop

            edges.append((source, target))
            consts.append(99e99)
            mults.append(1)

            connected += 1
            internal_arc += 1

        if connected > 0:
            nodes_with_edge += 1
            arc_rate = np.around(internal_arc / (time.time() - start_UPL), 2)
            print(f"index = {i} node = {source} connected arcs = {connected} total arcs generated = {internal_arc} x = {x_i} y = {y_i} z = {z_i} angle = {angle_i} arc gen rate = {arc_rate}")

    print("Block precedence search complete")
    # Batch add edges and attributes
    print("Adding edges to the graph")
    g.add_edges(edges)
    start_index = len(g.es) - len(edges)
    g.es[start_index:]["const"] = consts
    g.es[start_index:]["mult"] = mults

    #print(g.es["const"])   # List of 'const' values for all edges
    #print(g.es["mult"])    # List of 'mult' values for all edges
    assert all(e["const"] is not None for e in g.es), "Some edges have missing 'const'"
    assert all(e["mult"] is not None for e in g.es), "Some edges have missing 'mult'"

    total_int_arc_rate = np.around(internal_arc / (time.time() - start_UPL), 2)
    total_node_rate = np.around(nodes_with_edge / (time.time() - start_UPL), 2)

    print("\nPerformance:")
    print(f"--- Total Nodes Processed: {i}")
    print(f"--- Total Nodes with Edges: {nodes_with_edge}")
    print(f"--- Node-Edge Generation Rate: {total_node_rate}/s")
    print(f"--- Total Precedence Arcs: {internal_arc}")
    print(f"--- Precedence Arc Generation Rate: {total_int_arc_rate}/s")
    print("Precedence Arcs done")
    print("--> Precedence Arc Generation time: --%s seconds" % (np.round(time.time() - start_UPL, 2)))

    return g

def Pseudoflow_UPL(BM,
                   sink,
                   source,
                   idx,
                   xsize,
                   ysize,
                   zsize,
                   xmin,
                   ymin,
                   zmin,
                   xmax,
                   ymax,
                   zmax,
                   xcol,
                   ycol,
                   zcol,
                   slopecol,
                   search_blocks_front,
                   search_blocks_back,
                   search_blocks_left,
                   search_blocks_right,
                   num_blocks_above,
                   BVal,
                   pitLimit,
                   Cashflow,
                   minWidth):

    print("Process Start...")
    start_UPL = time.time()
    source = source
    sink = sink

    x_coords = BM[:, 0]           # X is column 0
    y_coords = BM[:, 1]           # Y is column 1
    z_coords = BM[:, 2]           # Z is column 2

    # create graph with enough vertices
    num_vertices = sink + 1  # include sink itself
    g = Graph(directed=True)
    g.add_vertices(num_vertices)

    g.vs["x"] = x_coords.tolist()
    g.vs["y"] = y_coords.tolist()
    g.vs["z"] = z_coords.tolist()

    # Connecting nodes
    print("Sending Nodes")
    g = sendNodes(BM, sink, source, g, BVal)
    print("External Arcs done")

    print("Creating Precedence")
    g = createArcPrecedence(BM,
                            idx,
                            xsize,
                            ysize,
                            zsize,
                            xmin,
                            ymin,
                            zmin,
                            xmax,
                            ymax,
                            zmax,
                            xcol,
                            ycol,
                            zcol,
                            slopecol,
                            search_blocks_front,
                            search_blocks_back,
                            search_blocks_left,
                            search_blocks_right,
                            num_blocks_above,
                            g,
                            minWidth)

    # Optimisation using pseudoflow
    print("Solving Ultimate Pit Limit")
    solve_UPL = time.time()
    RangeLambda = [0]
    breakpoints, cuts, info = pf.hpf(g, source, sink, const_cap="const", mult_cap="mult", lambdaRange=RangeLambda, roundNegativeCapacity=False)

    #Finding the blocks inside the resulting UPL.
    B = {x:y for x, y in cuts.items() if y == [1] and x!=0}
    InsideList = list(B.keys())

    # initialise Pit Limit and Cashflow columns to 0
    BM[:,pitLimit] = 0
    BM[:,Cashflow] = 0

    for indUPL in range(len(InsideList)):
        # Set blocks inside UPL = 1
        BM[int(InsideList[indUPL] -1),pitLimit] = 1
        BM[int(InsideList[indUPL] -1),Cashflow] = BM[int(InsideList[indUPL] -1),BVal]

        # Calculate cashflow by getting the sum of Cash Flow column
        cashFlow = "{:,.2f}".format(np.sum(BM[:,Cashflow]))

    print("--> Pseudoflow Optimization time: --%s seconds " % (np.around((time.time() - solve_UPL), decimals=2)))
    print("--> Total process time: --%s seconds " % (np.around((time.time() - start_UPL), decimals=2)))
    print(f"Undiscounted Cashflow: ${cashFlow}")

    return BM

def main():
    print("Start")
    start_time = time.time()

########################################################################################

    # 1. Block model location
    filePath = 'marvin_copper_final.csv'

    # 2. Block model size
    xsize = 30
    ysize = 30
    zsize = 30

    # 3. Column number of xyz coordinates. note that column number starts at 0
    xcol = 1
    ycol = 2
    zcol = 3

    # 4. Block search boundary parameters
    num_blocks_front = 5
    num_blocks_back = 5
    num_blocks_left = 5
    num_blocks_right = 5
    num_blocks_above = 6

    # 5. Minimum mining width for pit bottom consideration (this will be added to the radius of the search cone)
    minWidth = 0.0

    # 6. Column number of Block Value (column number starts at 0)
    BVal = 7 

    # 7. Column numbler of Slope Angle
    slopecol = 8

    # 8. Column numbler of Index ID
    idx = 0

########################################################################################

    data = np.loadtxt(filePath, delimiter=',', skiprows=1) # Import Block Model

    x_col = data[:, xcol]
    y_col = data[:, ycol]
    z_col = data[:, zcol]

    xmin = x_col.min()
    xmax = x_col.max()

    ymin = y_col.min()
    ymax = y_col.max()

    zmin = z_col.min()
    zmax = z_col.max()

    nx = ((xmax - xmin) / xsize) + 1
    ny = ((ymax - ymin) / ysize) + 1
    nz = ((zmax - zmin) / zsize) + 1

    sink = int((data.shape[0]) + 1)
    source = 0

    orig_col = data.shape[1]
    print(f"Original Column = {orig_col}")

    # Add two new columns for pitLimit and CashFlow
    n_rows = data.shape[0]
    col1 = np.zeros((n_rows, 1))
    col2 = np.zeros((n_rows, 1))
    data = np.hstack((data, col1, col2))

    # Store column numbers (indices) of new columns
    pitLimit = orig_col      # first new column
    CashFlow = orig_col + 1  # second new column
    print(f"UPL Column = {pitLimit}")
    print(f"Cashflow Column = {CashFlow}")

    BlockModel = data

    # Call Pseudoflow function
    BlockModel = Pseudoflow_UPL(BlockModel,
                                sink,
                                source,
                                idx,
                                xsize,
                                ysize,
                                zsize,
                                xmin,
                                ymin,
                                zmin,
                                xmax,
                                ymax,
                                zmax,
                                xcol,
                                ycol,
                                zcol,
                                slopecol,
                                num_blocks_front,
                                num_blocks_back,
                                num_blocks_left,
                                num_blocks_right,
                                num_blocks_above,
                                BVal,
                                pitLimit,
                                CashFlow,
                                minWidth
                                )

    # Save Block Model
    base, ext = os.path.splitext(filePath)

    np.savetxt(
        fname=f"{base}_opt{ext}",
        X=BlockModel,
        fmt='%.3f',
        delimiter=',',
        header="id+1,X,Y,Z,tonne,au_ppm,cu_pct,block_val,slope,pit_limit,cash_flow",
        comments=''  # <- this removes the default '#' comment character
    )

if __name__ == "__main__":
    main()    
