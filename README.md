# WALL-E and EVE's Signal

A self-contained, classical belief-search mission simulator. WALL-E explores a grid, spends limited battery/time to move, scan, analyze evidence, and recharge, then produces a trace, mission report, and ASCII path map.

## Environment and agent

`agent.py` contains the simulator (`World`) and autonomous planner (`Agent`). The world keeps target, decoy, and clue placement private. `World.agent_config()` provides a public-only initialization; actions return local observations, and clue content appears only after analysis. Mission success requires the environment's direct-contact observation.

The agent maintains a normalized Bayesian belief over traversable cells. Initial bearing and strength, noisy scans, decoy outliers, uncertain bearings, probabilistic directional clues, and negative evidence from an empty analysis update the belief. It uses expected entropy reduction to rank a bounded set of scan sites, balances that value with target probability and route cost, tracks revisits, and checks A* return routes to chargers against battery, time, and action limits. A* expands the lowest `f(n) = g(n) + h(n)` first; `h(n)` is Manhattan distance `|x1-x2| + |y1-y2|`. Four-way equal-cost movement makes this heuristic admissible and yields shortest routes around obstacles.

Grid coordinates start at `[0,0]` in the upper left; east increases x and south increases y. `#` are impassable obstacles; `+` are recharge stations. Costs consume equal battery and mission-time units. `analyze` returns local evidence or no clue; `scan` costs two by default; `recharge` restores up to eight battery units by default and costs one time unit. The sample includes a decoy beacon, so the strongest signal return may be misleading.

## Configuration

The JSON config defines dimensions, seed, start, initial signal, resource budget and per-action costs, terrain, hidden signal target/status/noise/decoys, clue text/coordinates, and step cap. Optional settings include `signal.bearing_uncertainty`, `signal.decoy_outlier_probability`, and `resources.recharge_amount`. Confirmation clues follow configured EVE coordinates. Directional clues are emitted as uncertain observations consistent with the configured target. Authored hidden mission data is never passed to the agent.

## Run

The agent, tests, and benchmark require Python 3.10+ and use only the standard library. The playable game additionally requires Pygame (see `requirements.txt`).

```sh
python agent.py mission.json --out outputs/sample_run
```

The run writes `trace.json`, `report.json`, and `map.txt`. Every trace row records action, observation, strategy, and decision rationale. Sample artifacts in `outputs/sample_run/` are generated from the included mission.

Run validation and evaluation:

```powershell
py -m unittest discover -s tests -v
py benchmark.py --count 50 --seed 81024 --out outputs/benchmark
```

The benchmark generates deterministic valid and intentionally disconnected missions, varying target/start/charger locations, obstacles, signal noise, bearings, decoys, budgets, and seeds. A fixed-seed run generated 35 distinct EVE positions among 50 missions, including five intentionally unreachable targets. It compares the information-driven agent with a reconstructed MAP-chasing/periodic-scan baseline on identical worlds and seeds. Results include per-mission data plus success slices by noise and decoy presence in `outputs/benchmark/results.json`, with a readable summary in `outputs/benchmark/summary.md`.

Recorded run (seed `81024`, 50 missions; 45 solvable): the baseline contacted EVE in 35/50 missions (77.8% of solvable cases), while the information agent did so in 21/50 (46.7% of solvable cases). Mean actions for successful/failed/all runs were 22.11/34.20/25.74 for the baseline and 31.95/49.72/42.26 for the information agent. Energy per success was 27.97 vs. 31.43; mean explored cells 8.44 vs. 17.72; average final Manhattan prediction error 0.68 vs. 1.58. The information agent explored more broadly but was less successful and less efficient in this run, so the measurements do not support claiming an improvement. Noise and decoy slices are descriptive for this finite seeded sample, not proof of a causal trend.

## Limits

This is a grid-world demonstration, not a physics-accurate simulation. Exact EIG uses a discretized signal/bearing outcome model for the 16 highest-probability scan candidates plus the current cell to bound runtime. The baseline is a reconstruction of the earlier MAP-target/periodic-scan behavior that shares current evidence updates, not a preserved executable snapshot. The sensor likelihood approximates clipped max-of-beacons readings with an outlier mixture; calibration could change measured results. Multi-agent coordination and Gazebo/Webots are outside scope.

## Play the Pygame simulation

Install Python 3.10 or newer, then from the project folder run:

```powershell
py -m pip install -r requirements.txt
py game.py
```

If the Windows Python launcher is unavailable, use `python -m pip install -r requirements.txt` followed by `python game.py`. Pass another JSON mission as `py game.py path\to\mission.json`. Click **Start**, **Pause/Resume**, or **Restart** to control the simulation. Enter EVE X/Y and choose **Apply & Restart** to validate and save a new location and start a fresh search. Coordinates are zero-based; invalid and obstacle cells are rejected. Press **H** to toggle the belief heatmap; it visualizes only the agent's posterior. Space toggles pause and R restarts. The panel shows live strategy, estimated target, confidence, entropy, current action, and rationale. EVE is drawn only after direct contact. The game writes its trace and report to `outputs/pygame_run/`.





