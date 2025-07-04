# pit-optimisation

## The block model csv file needs to have the following data:

    • index – Unique identifier for each block. Must be sequential and start at 1 
    • x – X coordinate of the block centroid
    • y – Y coordinate of the block centroid
    • z – Z coordinate of the block centroid
    • block value – Net economic value of the blockTypically calculated as revenue – cost
    • slope_angle – Local geotechnical slope angle (in degrees)Used to construct the inverted cone or frustum for slope constraints

## Running the script:

Install the libraries in the requirements.txt 

Open pit_optimiser.py in python IDE / Jupyter Notebook to configure the following:
    
    • File location of the block model (line 312)
    • Specify block model size (lines 315-317)
    • Map the columns of the xyz coordinates (lines 320-322)
    • Set the search boundary parameters for the cone (lines 325-329)
    • Set the minimum mining width (line 332)
    • Map the column of the Block Value (line 335)
    • Map the column of the slope angle (line 338)
    • Map the index column (line 341)

if you find any bugs, feel free to report

### Pseudoflow Licensing Notice

This project uses the **Pseudoflow** library for optimisation (e.g., max-flow/min-cut).

However:

- **Pseudoflow is not open source**
- It is **not included** in this repository
- It is licensed only for **educational, research, and not-for-profit purposes**
- **Commercial use of Pseudoflow requires a separate license** from UC Berkeley

Learn more:  
https://riot.ieor.berkeley.edu/Applications/Pseudoflow/maxflow.html

If your use case involves Pseudoflow, you are responsible for ensuring compliance with its license terms.
