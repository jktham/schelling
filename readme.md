ABM course project: Modified Schelling Model

The code has five .py files, four of them are deprecated as they were different tests of the model. The final version is model_combined.py

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
