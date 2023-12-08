ABM course project: Modified Schelling Model

stuff so far:
- model_basic: basic schelling model with grid size, number of agent types, similarity threshold, empty cell ratio
- model_upgraded_hanna: adds agent weights, multiple move strategies, prices per cell with agents preferring cheapest cells
- model_upgraded_jonas: increase prices near center
- model_upgraded_benja: adds prices per cell based on number of neighbors and agent wealth, with agents preferring most expensive affordable cells
- model_combined: combine above approaches, add points of interest that agents want to move closer to depending on preferences, maybe model real city

todo:
- ~write comments~
- ~add benjas price model~
- ~add weights for combined strategy~
- add random agent interest variations
- ~manually add points of interest~
- ~manually define cell types (ie lake)~
- ~output statistics, maybe animate grid~
- increase size -> improve performance, currently 1-2 min for size 50, 300 iterations (maybe port final version to c++?)
- presort cells by distance
- show parameters on output plot
