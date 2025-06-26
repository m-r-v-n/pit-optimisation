# pit-optimisation

The block model needs to have the following data:

    • index – Unique identifier for each block. Must be sequential and start at 1 
    
    • x – X coordinate of the block centroid
    
    • y – Y coordinate of the block centroid
    
    • z – Z coordinate of the block centroid
    
    • block value – Net economic value of the blockTypically calculated as revenue – cost
    
    • slope_angle – Local geotechnical slope angle (in degrees)Used to construct the inverted cone or frustum for slope constraints

Running the script:

Install the libraries in the requirements.txt
    
Open pit_optimiser.py in python IDE / Jupyter Notebook
    
    • File location of the block model (line 312)
    • Specify block model size (lines 315-317)
    • Map the columns of the xyz coordinates (lines 320-322)
    • Set the search boundary parameters for the cone (lines 325-329)
    • Set the minimum mining width (line 332)
    • Map the column of the Block Value (line 335)
    • Map the column of the slope angle (line 338)
    • Map the index column (line 341)
