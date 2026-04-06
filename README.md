# RLA2 — Antarctic robot charging station placement


## File structure

```
RLA2/
├── data/
│   ├── robot_locations.csv    # Robot lon/lat
│   ├── range_scenarios.csv    # Multi-scenario ranges (column s0 = deterministic) 
└── src/
    ├── data_loader.py         # CSV → ProblemData
    ├── utils.py               # Haversine, distance matrix, charging probability
    ├── solution.py            # Solution, evaluate, feasibility
    │
    ├── verify_phase0.py       # Data / pipeline sanity checks (run first)
    │
    ├── q1a_minlp.py           # Q1a: MINLP (Xpress)
    ├── q1b_construction.py    # Q1b: construction heuristic (+ k-scan in main)
    ├── q1c_alns.py            # Q1c: ALNS + LAHC (full 1072 uses Q1b k=120)
    │
    ├── q2_utils.py            # Expected cost, CVaR, etc. (library)
    ├── q2_benders.py          # Benders / L-shaped
    ├── q2_benders_full.py     # Q2: extended Benders workflow
    ├── q2_salns.py            # Q2 heuristic: stochastic ALNS (operators from q1c_alns)
    │
    ├── pilot_convergence.py   # ALNS convergence pilot (q1b + q1c_alns)
    └── visualization.py     # Maps / figures 
```

## Dependencies

- **Required:** `numpy`, `pandas`, `xpress` , `matplotlib`, `cartopy` 

## How to run


```bash
cd src
python verify_phase0.py
python q1b_construction.py
python q1c_alns.py
python q2_salns.py 
python pilot_convergence.py 
python visualization.py 

```
