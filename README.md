# WALL-E and EVE's Signal

A self-contained, classical belief-search mission simulator. WALL-E explores a grid, spends limited battery/time to move, scan, analyze evidence, and recharge, then produces a trace, mission report, and ASCII path map.

## Environment and agent

`agent.py` contains both the simulator (`World`) and autonomous planner (`Agent`). The world holds target coordinates and clue placements privately. Actions return local observations only; clues are disclosed only when WALL-E analyzes their tile. The simulated receiver produces noisy inverse-distance signal strength and a bearing. Signal readings update a categorical Bayesian belief over traversable map cells. The agent inspects visited tiles, follows the current maximum-posterior candidate, uses directional/status evidence to adapt its policy, scans periodically, and can recharge at configured chargers.

Grid coordinates start at `[0,0]` in the upper left; east increases x and south increases y. `#` are impassable obstacles; `+` are recharge stations. Costs consume equal battery and mission-time units. `analyze` returns a clue or no clue; `scan` costs two by default; `recharge` restores up to eight battery units and costs one time unit. The sample includes a decoy beacon; the strongest receiver return may be the decoy, while clue evidence can correct the search.

## Configuration

The JSON config defines dimensions, seed, start, initial signal, resource budget and per-action costs, terrain, hidden signal target/status/noise/decoys, clue text and hidden clue coordinates, and step cap. See `mission.json`. For a new mission, edit the `signal` and `clues` fields or change the seed and terrain. The current sample's clue placements are intentionally visible in the config file for mission authoring; they are not included in agent observations.

## Run

The agent CLI requires Python 3.10+ and uses only the standard library. The playable game additionally requires Pygame (see `requirements.txt`).

```sh
python agent.py mission.json --out outputs/sample_run
```

The run writes `trace.json`, `report.json`, and `map.txt`. Every trace row records action, observation, and decision rationale. Sample artifacts in `outputs/sample_run/` are generated from the included mission.

## Limits

This is a compact baseline, not a physics-accurate WALL-E simulation. The receiver bearing currently uses simulated direction, and the belief update uses a simple noise likelihood; a production version should make bearing noisy and include sensor calibration, route planning to recharge stations, and repeated randomized benchmark evaluation. Multi-agent coordination and Gazebo/Webots are outside this baseline.

## Play the Pygame simulation

Install Python 3.10 or newer, then from the project folder run:

```powershell
py -m pip install -r requirements.txt
py game.py
```

If the Windows Python launcher is unavailable, use `python -m pip install -r requirements.txt` followed by `python game.py`. Pass another JSON mission as `py game.py path\to\mission.json`. Click **Start** to begin, **Pause/Resume** to control the simulation, and **Restart** to reset the same mission. Space toggles pause and R restarts. WALL-E takes one real `Agent.choose`/`World.observe` step roughly every half second. The game writes the full trace and final report to `outputs/pygame_run/` without changing the existing sample run artifacts.

