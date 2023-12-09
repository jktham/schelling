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
- ~show parameters on output plot~

results:

random:
- including desirability in satisfaction adds random noise and significantly affects threshold behavior.
- sat (1.0, 0.0), 3 types: stable threshold 0.66 (23-30)
- sat (1.0, 0.0), 2 types: stable threshold >0.8 (31-34)
- sat (0.5, 0.5), 3 types: stable threshold 0.5 (35-36)
- thresholds below 0.3 lead to only slight or no segregation, thresholds above stable never start to stabilize, despite lower threshold values eventually reaching higher averages -> could start threshold low and increase over time to reach higher average satisfaction?
- when agents are unstable and move mostly due to random noise, bias towards top of grid becomes apparent.

point distance:
- interesting pattern of agents shifting top left within radius (37,38)

dynamic price:
- loose clusters of agents form, with stable background pattern throughout

combined:
- weirdly the poorest 2 agents tend to get closer to the point and form more coherent groups, while the richest type (purple) are much more distributed yet generally remain more stable (39)
