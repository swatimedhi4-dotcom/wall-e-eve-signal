from __future__ import annotations
import argparse, heapq, json, math, random
from pathlib import Path

DIRS = {"north": (0,-1), "east": (1,0), "south": (0,1), "west": (-1,0)}

def dist(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

def a_star(start, goal, width, height, obstacles):
    """Return a shortest four-direction path, including endpoints, or []."""
    start, goal = tuple(start), tuple(goal)
    blocked = {tuple(p) for p in obstacles}
    inside = lambda p: 0 <= p[0] < width and 0 <= p[1] < height
    if not inside(start) or not inside(goal) or start in blocked or goal in blocked:
        return []
    frontier = [(dist(start, goal), 0, start)]
    came_from = {}
    cost_so_far = {start: 0}
    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if cost != cost_so_far.get(current):
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        for dx, dy in DIRS.values():
            neighbor = (current[0] + dx, current[1] + dy)
            if not inside(neighbor) or neighbor in blocked:
                continue
            new_cost = cost + 1
            if new_cost < cost_so_far.get(neighbor, float("inf")):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                estimate = new_cost + dist(neighbor, goal)
                heapq.heappush(frontier, (estimate, new_cost, neighbor))
    return []

class World:
    """Grid world. Hidden targets and clue positions are never exposed in observations."""
    def __init__(self, cfg):
        self.c=cfg; self.w=cfg["width"]; self.h=cfg["height"]
        if self.w <= 0 or self.h <= 0:
            raise ValueError("grid width and height must be positive")
        self.pos=tuple(cfg["start"]); self.battery=cfg["resources"]["battery"]; self.time=cfg["resources"]["time"]
        self.obstacles={tuple(p) for p in cfg["terrain"]["obstacles"]}; self.chargers={tuple(p) for p in cfg["terrain"]["chargers"]}
        self.eve=tuple(cfg["signal"]["eve_location"])
        if not self._inside(self.pos) or self.pos in self.obstacles:
            raise ValueError("start must be inside the grid and outside obstacles")
        if not self._inside(self.eve) or self.eve in self.obstacles:
            raise ValueError("signal.eve_location must be inside the grid and outside obstacles")
        for charger in self.chargers:
            if not self._inside(charger) or charger in self.obstacles:
                raise ValueError(f"charging station {charger} must be inside the grid and outside obstacles")
        if self.battery <= 0 or self.time <= 0:
            raise ValueError("battery and time budgets must be positive")
        for action in ("move", "scan", "analyze", "recharge"):
            if cfg["resources"].get(action + "_cost", 0) <= 0:
                raise ValueError(f"{action}_cost must be positive")
        self.recharge_amount=cfg["resources"].get("recharge_amount",8)
        if self.recharge_amount <= 0:
            raise ValueError("recharge_amount must be positive")
        self.signal_noise=max(0.0,float(cfg["signal"].get("noise",0.08)))
        self.bearing_uncertainty=min(1.0,max(0.0,float(cfg["signal"].get("bearing_uncertainty",0.12))))
        self.decoy_outlier_probability=min(0.45,max(0.0,float(cfg["signal"].get("decoy_outlier_probability",0.12))))
        # Confirmation is attached to EVE, so keep it colocated when her configured
        # location changes without requiring a second coordinate edit in the JSON.
        self.clues={}
        for clue in cfg["clues"]:
            placed=dict(clue)
            if placed.get("type")=="confirmation":
                placed["at"]=list(self.eve)
            point=tuple(placed["at"])
            if not self._inside(point) or point in self.obstacles:
                raise ValueError(f"clue location {point} must be inside the grid and outside obstacles")
            self.clues[point]=placed
        self.found=set()
        self.decoys=[(tuple(d["location"]),d) for d in cfg["signal"].get("decoys",[])]
        for location, _ in self.decoys:
            if not self._inside(location) or location in self.obstacles:
                raise ValueError(f"decoy location {location} must be inside the grid and outside obstacles")
        if cfg.get("max_steps", 60) <= 0:
            raise ValueError("max_steps must be positive")
        self.rng=random.Random(cfg["seed"]); self.tick=0
    def _inside(self,p): return 0<=p[0]<self.w and 0<=p[1]<self.h
    def agent_config(self):
        """A defensive public-only config; hidden signal and clue state stay in World."""
        return {
            "width":self.w,"height":self.h,"start":list(self.c["start"]),
            "initial_signal":dict(self.c["initial_signal"]),
            "resources":dict(self.c["resources"]),
            "terrain":{"obstacles":[list(p) for p in sorted(self.obstacles)],"chargers":[list(p) for p in sorted(self.chargers)]},
            "sensors":{"signal_noise":self.signal_noise,"bearing_uncertainty":self.bearing_uncertainty,"decoy_outlier_probability":self.decoy_outlier_probability},
            "max_steps":self.c.get("max_steps",60),
        }
    def observe(self, action):
        typ=action["type"]; r=self.c["resources"]; cost=r[typ+"_cost"]
        if self.battery < cost:
            return {"ok":False,"reason":"battery budget exhausted"}
        if self.time < cost:
            return {"ok":False,"reason":"mission time budget exhausted"}
        self.battery-=cost; self.time-=cost; self.tick+=1
        obs={"ok":True,"action":typ,"position":list(self.pos),"battery":self.battery,"time":self.time,"tick":self.tick}
        if typ=="move":
            dx,dy=DIRS[action["direction"]]; p=(self.pos[0]+dx,self.pos[1]+dy)
            if 0<=p[0]<self.w and 0<=p[1]<self.h and p not in self.obstacles: self.pos=p
            else: obs["bump"]=True
            obs["position"]=list(self.pos)
        elif typ=="scan":
            signal=self.c["signal"]; d=dist(self.pos,self.eve); true=1/(1+d)
            readings=[true+self.rng.uniform(-self.signal_noise,self.signal_noise)]
            for loc,decoy in self.decoys: readings.append(decoy.get("strength",0.8)/(1+dist(self.pos,loc))+self.rng.uniform(-self.signal_noise,self.signal_noise))
            obs["signal_strength"]=round(max(0,min(1,max(readings))),3)
            true_bearing="unknown" if self.pos==self.eve else max(DIRS,key=lambda z: (self.eve[0]-self.pos[0])*DIRS[z][0]+(self.eve[1]-self.pos[1])*DIRS[z][1])
            if self.rng.random()<self.bearing_uncertainty:
                obs["bearing"]=self.rng.choice(list(DIRS))
            else:
                obs["bearing"]=true_bearing
        elif typ=="analyze":
            clue=self.clues.get(self.pos)
            if clue and self.pos not in self.found:
                self.found.add(self.pos)
                disclosed={"type":clue["type"],"text":clue["text"]}
                if clue.get("type")=="directional":
                    direction=max(DIRS,key=lambda name:(self.eve[0]-self.pos[0])*DIRS[name][0]+(self.eve[1]-self.pos[1])*DIRS[name][1])
                    disclosed["direction"]=direction if self.pos!=self.eve else "unknown"
                    disclosed["reliability"]=0.65
                    disclosed["text"]=f"Weak tracks point {disclosed['direction']}; treat this as an uncertain lead."
                obs["clue"]=disclosed
            else: obs["clue"]=None
            # Analysis is local: the agent learns only clue content, never the hidden target coordinate.
            if self.pos==self.eve: obs["contact"]={"status":self.c["signal"]["eve_status"],"confirmation":True}
        elif typ=="recharge":
            if self.pos not in self.chargers: obs["recharged"]=False
            else: self.battery=min(self.c["resources"]["battery"],self.battery+self.recharge_amount); obs["recharged"]=True; obs["battery"]=self.battery
        return obs

class Agent:
    """Belief-state planner that receives only World.agent_config() and local observations."""
    def __init__(self, cfg):
        if any(key in cfg for key in ("signal", "clues", "decoys", "eve_location")):
            raise ValueError("Agent requires World.agent_config(); hidden mission data is not agent knowledge")
        self.cfg = cfg
        self.width, self.height = cfg["width"], cfg["height"]
        self.pos = tuple(cfg["start"])
        self.obstacles = {tuple(p) for p in cfg["terrain"]["obstacles"]}
        self.chargers = {tuple(p) for p in cfg["terrain"]["chargers"]}
        free = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in self.obstacles]
        if not free:
            raise ValueError("mission has no traversable grid cells")
        self.belief = {(x, y): (1 / len(free) if (x, y) not in self.obstacles else 0.0)
                       for y in range(self.height) for x in range(self.width)}
        direction = cfg["initial_signal"].get("direction")
        if direction in DIRS:
            error = min(1.0, max(0.0, cfg["sensors"].get("bearing_uncertainty", 0.12)))
            dx, dy = DIRS[direction]
            for point, probability in self.belief.items():
                if not probability:
                    continue
                dot = (point[0] - self.pos[0]) * dx + (point[1] - self.pos[1]) * dy
                predicted = "unknown" if dot == 0 else max(DIRS, key=lambda name:
                    (point[0] - self.pos[0]) * DIRS[name][0] + (point[1] - self.pos[1]) * DIRS[name][1])
                self.belief[point] *= (1 - error + error / 4) if predicted == direction else error / 4
        initial_strength = cfg["initial_signal"].get("strength")
        if initial_strength is not None:
            noise = max(0.015, cfg["sensors"].get("signal_noise", 0.08))
            outlier = min(0.45, max(0.0, cfg["sensors"].get("decoy_outlier_probability", 0.12)))
            for point, probability in self.belief.items():
                if not probability:
                    continue
                mean = 1 / (1 + dist(self.pos, point))
                compatible = max(0.0, min(1.0, mean + noise) - max(0.0, mean - noise))
                likelihood = (1 - outlier) * (1.0 / max(compatible, 1e-9)
                              if abs(float(initial_strength) - mean) <= noise else 1e-6) + outlier
                self.belief[point] *= likelihood
        self._normalise()
        self.signal_noise = max(0.0, cfg["sensors"].get("signal_noise", 0.08))
        self.bearing_uncertainty = min(1.0, max(0.0, cfg["sensors"].get("bearing_uncertainty", 0.12)))
        self.outlier_probability = min(0.45, max(0.0, cfg["sensors"].get("decoy_outlier_probability", 0.12)))
        self.resources = cfg["resources"]
        self.max_steps = cfg.get("max_steps", 60)
        self.found, self.inspected, self.eliminated = [], set(), set()
        self.scan_counts, self.visit_counts = {}, {self.pos: 1}
        self.position_history = [self.pos]
        self._eig_cache = {}
        self.stagnant_scans = 0
        self.strategy = "Bayesian information search"
        self.last_decision_reason = ""
        self.last_action = None
        self.current_target = self.estimated_target
        self.current_information_gain = 0.0
        self.failure_reason = None
        self.contact_location = None
        self.log = []

    def _normalise(self):
        total = sum(max(0.0, v) for v in self.belief.values())
        if total > 0:
            self.belief = {p: (max(0.0, v) / total if p not in self.obstacles else 0.0)
                           for p, v in self.belief.items()}

    @property
    def entropy(self):
        return -sum(p * math.log2(p) for p in self.belief.values() if p > 0)

    @property
    def estimated_target(self):
        return max(self.belief, key=self.belief.get)

    @property
    def confidence(self):
        return max(self.belief.values(), default=0.0)

    def _bearing_for(self, site, target):
        if site == target:
            return "unknown"
        return max(DIRS, key=lambda d: (target[0] - site[0]) * DIRS[d][0] + (target[1] - site[1]) * DIRS[d][1])

    def _scan_likelihoods(self, site, target):
        bins = 9
        mean = 1 / (1 + dist(site, target))
        noise = self.signal_noise
        signal = []
        for i in range(bins):
            center, half = i / (bins - 1), 0.5 / (bins - 1)
            if noise < 1e-9:
                mass = 1.0 if abs(center - mean) <= half else 0.0
            else:
                overlap = max(0.0, min(center + half, mean + noise) - max(center - half, mean - noise))
                mass = overlap / (2 * noise)
            signal.append((1 - self.outlier_probability) * mass + self.outlier_probability / bins)
        total = sum(signal)
        signal = [v / total for v in signal] if total else [1 / bins] * bins
        predicted, error = self._bearing_for(site, target), self.bearing_uncertainty
        bearings = list(DIRS) + ["unknown"]
        bearing_likelihoods = []
        for observed in bearings:
            if predicted == "unknown":
                value = 1 - error if observed == "unknown" else error / 4
            elif observed == "unknown":
                value = 1e-9
            else:
                value = 1 - error + error / 4 if observed == predicted else error / 4
            bearing_likelihoods.append(value)
        return [sp * bp for sp in signal for bp in bearing_likelihoods]

    def expected_information_gain(self, site):
        site = tuple(site)
        if site in self._eig_cache:
            return self._eig_cache[site]
        current = self.entropy
        states = [(p, prob) for p, prob in self.belief.items() if prob > 0]
        if current <= 1e-12 or not states:
            self._eig_cache[site] = 0.0
            return 0.0
        rows = [(prob, self._scan_likelihoods(site, p)) for p, prob in states]
        outcomes = len(rows[0][1])
        marginals = [sum(pr * likelihood[i] for pr, likelihood in rows) for i in range(outcomes)]
        expected_entropy = 0.0
        for i, marginal in enumerate(marginals):
            if marginal <= 0:
                continue
            posterior_entropy = 0.0
            for prior, likelihood in rows:
                posterior = prior * likelihood[i] / marginal
                if posterior > 0:
                    posterior_entropy -= posterior * math.log2(posterior)
            expected_entropy += marginal * posterior_entropy
        gain = max(0.0, current - expected_entropy)
        self._eig_cache[site] = gain
        return gain

    def update(self, obs):
        old_pos = self.pos
        self.pos = tuple(obs.get("position", self.pos))
        if obs.get("action") == "move" and self.pos != old_pos:
            self.visit_counts[self.pos] = self.visit_counts.get(self.pos, 0) + 1
            self.position_history.append(self.pos)
            self.position_history = self.position_history[-12:]
        if obs.get("clue"):
            clue = obs["clue"]
            self.found.append({"position": list(self.pos), **clue})
            self.strategy = f"Update search from discovered {clue['type']} clue"
            if clue.get("type") == "directional" and clue.get("direction") in DIRS:
                reliability = min(0.99, max(0.25, float(clue.get("reliability", 0.65))))
                direction = clue["direction"]
                dx, dy = DIRS[direction]
                for point, prior in list(self.belief.items()):
                    if prior <= 0 or point == self.pos:
                        continue
                    dot = (point[0]-self.pos[0])*dx + (point[1]-self.pos[1])*dy
                    likelihood = reliability if dot > 0 else (1-reliability)/3
                    self.belief[point] *= likelihood
                self._normalise()
                self._eig_cache.clear()
        if obs.get("action") == "analyze":
            self.inspected.add(self.pos)
            if obs.get("contact"):
                self.contact_location = self.pos
                self.belief = {p: (1.0 if p == self.pos else 0.0) for p in self.belief}
            elif self.belief.get(self.pos, 0.0) > 0:
                remaining = sum(v for p, v in self.belief.items() if p != self.pos)
                if remaining > 0:
                    self.belief[self.pos] = 0.0
                    self.eliminated.add(self.pos)
                    self._normalise()
                else:
                    self.failure_reason = "Contradictory observations eliminated every remaining location."
            self._eig_cache.clear()
        if "signal_strength" in obs:
            old_entropy = self.entropy
            observed, bearing = obs["signal_strength"], obs.get("bearing")
            logs = {}
            for p, prior in self.belief.items():
                if prior <= 0:
                    continue
                mean = 1 / (1 + dist(self.pos, p))
                noise = max(self.signal_noise, 0.015)
                strength = (1 - self.outlier_probability) * (1 / (2 * noise) if abs(observed - mean) <= noise else 1e-8) + self.outlier_probability
                predicted, error = self._bearing_for(self.pos, p), self.bearing_uncertainty
                if bearing == "unknown":
                    bearing_like = (1 - error if predicted == "unknown" else 1e-8)
                elif bearing in DIRS:
                    bearing_like = (1 - error + error / 4) if predicted == bearing else error / 4
                else:
                    bearing_like = 1e-8
                logs[p] = math.log(prior) + math.log(max(1e-12, strength)) + math.log(max(1e-12, bearing_like))
            if logs:
                peak = max(logs.values())
                self.belief = {p: (math.exp(logs[p] - peak) if p in logs else 0.0) for p in self.belief}
                self._normalise()
            self.scan_counts[self.pos] = self.scan_counts.get(self.pos, 0) + 1
            self.stagnant_scans = self.stagnant_scans + 1 if self.entropy >= old_entropy - 1e-3 else 0
            self._eig_cache.clear()
        self.last_action = obs.get("action", self.last_action)

    def _path_to_charger(self, start):
        routes = [(a_star(start, c, self.width, self.height, self.obstacles), c) for c in self.chargers]
        routes = [(p, c) for p, c in routes if p]
        return min(routes, key=lambda item: len(item[0])) if routes else ([], None)

    def choose(self, obs):
        self.failure_reason = None
        self.last_decision_reason = ""
        if obs.get("contact") or self.contact_location is not None:
            return None
        tick = obs.get("tick", 0)
        if tick >= self.max_steps:
            self.failure_reason = "Maximum action limit reached."
            return None
        battery, time_left = obs["battery"], obs["time"]
        move_cost = self.resources.get("move_cost", 1)
        scan_cost = self.resources.get("scan_cost", 2)
        analyze_cost = self.resources.get("analyze_cost", 1)
        recharge_cost = self.resources.get("recharge_cost", 1)
        capacity = self.resources["battery"]
        recharge_amount = self.resources.get("recharge_amount", 8)
        charger_path, charger = self._path_to_charger(self.pos)
        return_energy = (len(charger_path) - 1) * move_cost if charger_path else 0
        if charger_path and ((battery <= return_energy + move_cost) or
            (self.pos == charger and battery < capacity and battery <= capacity - recharge_amount)):
            if self.pos == charger:
                if time_left < recharge_cost:
                    self.failure_reason = "Not enough mission time remains to recharge."
                    return None
                self.strategy = "Recharge at the current station before continuing"
                self.last_decision_reason = f"Recharging; A* return reserve is {return_energy} battery."
                return {"type": "recharge"}
            if time_left < return_energy:
                self.failure_reason = "Not enough mission time remains to reach a charging station."
                return None
            self.strategy = "Return to the nearest reachable charger"
            self.last_decision_reason = f"Returning to a charger; A* route requires {return_energy} battery."
            return self._move_along(charger_path)
        if self.pos not in self.inspected and battery >= analyze_cost and time_left >= analyze_cost:
            self.strategy = "Analyze this unvisited location for local evidence"
            self.last_decision_reason = "Analyzing this location once for clues or direct contact."
            return {"type": "analyze"}
        if sum(self.belief.values()) <= 0:
            self.failure_reason = "No candidate locations remain after observations."
            return None

        entropy = self.entropy
        max_probability = max(self.confidence, 1e-12)
        adaptive_gain_weight = min(0.18, 0.06 * self.stagnant_scans)
        alternating = (len(self.position_history) >= 4 and
                       self.position_history[-1] == self.position_history[-3] and
                       self.position_history[-2] == self.position_history[-4] and
                       self.position_history[-1] != self.position_history[-2])
        candidates = []
        # Exact EIG is more expensive than route ranking. Evaluate the current
        # tile and the strongest 16 posterior candidates; refresh after evidence.
        ranked_sites = [p for p, value in sorted(self.belief.items(), key=lambda item: item[1], reverse=True)
                        if value > 0][:16]
        if self.pos not in ranked_sites:
            ranked_sites.append(self.pos)
        for site in ranked_sites:
            probability = self.belief[site]
            scans = self.scan_counts.get(site, 0)
            if site in self.obstacles or scans >= 3:
                continue
            path = a_star(self.pos, site, self.width, self.height, self.obstacles)
            if not path:
                continue
            return_path, _ = self._path_to_charger(site)
            return_steps = len(return_path) - 1 if return_path else 0
            uninspected = sum(1 for point in path[1:] if point not in self.inspected)
            travel_energy = (len(path) - 1) * move_cost + uninspected * analyze_cost
            required = travel_energy + scan_cost + return_steps * move_cost
            duration = travel_energy + scan_cost + return_steps * move_cost
            actions = len(path) - 1 + uninspected + 1 + return_steps
            if battery < required or time_left < duration or tick + actions > self.max_steps:
                continue
            eig = self.expected_information_gain(site)
            posterior_value = probability / max_probability
            novelty = 1 / (1 + scans)
            travel_penalty = 0.015 * (len(path) - 1)
            revisit_penalty = 0.035 * min(self.visit_counts.get(site, 0), 4) + 0.18 * scans
            if alternating and site in self.position_history[-4:]:
                revisit_penalty += 0.22
            gain_weight = 0.30 + adaptive_gain_weight
            score = gain_weight * eig / max(entropy, 1e-9) + (0.58 - adaptive_gain_weight / 2) * posterior_value + (0.12 - adaptive_gain_weight / 2) * novelty - travel_penalty - revisit_penalty
            candidates.append((score, eig, site, path, required))
        if not candidates:
            if self.pos in self.chargers and battery < capacity and time_left >= recharge_cost:
                self.strategy = "Recharge because no feasible search route remains"
                self.last_decision_reason = "Recharging because resources cannot support another feasible search route."
                return {"type": "recharge"}
            if charger_path and self.pos != charger and battery >= return_energy and time_left >= return_energy:
                self.strategy = "Return to charger; no exploration plan fits resources"
                self.last_decision_reason = f"No feasible scan plan; returning via a {len(charger_path)-1}-step A* route."
                return self._move_along(charger_path)
            self.failure_reason = "No feasible scan route remains within battery, time, and action limits."
            return None
        _, eig, site, path, _ = max(candidates, key=lambda item: item[0])
        self.current_target, self.current_information_gain = site, eig
        if site == self.pos:
            if eig > 0.015:
                self.strategy = "Scan here for expected information gain"
                self.last_decision_reason = f"Scanning here for {eig:.2f} bits expected information gain (entropy {entropy:.2f})."
                return {"type": "scan"}
            alternatives = [item for item in candidates if item[2] != self.pos]
            if alternatives:
                _, next_gain, next_site, next_path, _ = max(alternatives, key=lambda item: item[0])
                self.current_target, self.current_information_gain = next_site, next_gain
                self.strategy = "Investigate another candidate because this scan has little value"
                self.last_decision_reason = f"This site's expected gain is only {eig:.3f} bits; moving to candidate {next_site}."
                return self._move_along(next_path)
            self.failure_reason = "Expected information gain is negligible at feasible scan locations."
            return None
        self.strategy = "Navigate to a high-value scan location"
        loop_note = " Reconsidering recent alternating positions." if alternating else ""
        self.last_decision_reason = f"Moving toward {site}: expected information gain {eig:.2f} bits; A* route {len(path)-1} steps.{loop_note}"
        return self._move_along(path)

    @staticmethod
    def _move_along(path):
        dx, dy = path[1][0] - path[0][0], path[1][1] - path[0][1]
        direction = next(name for name, delta in DIRS.items() if delta == (dx, dy))
        return {"type": "move", "direction": direction}


def render(cfg, trace):
    w,h=cfg["width"],cfg["height"]; grid=[['.' for _ in range(w)] for _ in range(h)]
    for x,y in cfg["terrain"]["obstacles"]: grid[y][x]='#'
    for x,y in cfg["terrain"]["chargers"]: grid[y][x]='+' 
    for row in trace:
        p=row["observation"].get("position")
        if p: x,y=p; grid[y][x]='*'
    sx,sy=cfg["start"]; grid[sy][sx]='S'
    return "\n".join("".join(r) for r in grid)

def run(config_path,outdir):
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8-sig")); world=World(cfg); agent=Agent(world.agent_config()); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    initial={"position":list(world.pos),"battery":world.battery,"time":world.time,"tick":0}
    contact=False
    for step in range(cfg.get("max_steps",60)):
        action=agent.choose(initial)
        if action is None: break
        obs=world.observe(action)
        entry={"step":step+1,"action":action,"observation":obs,"decision":{"strategy":agent.strategy,"reasoning":agent.last_decision_reason or reason(action,agent,obs)}}
        agent.update(obs); agent.log.append(entry); initial=obs
        if obs.get("contact"): contact=True; break
        if not obs.get("ok"): break
    if not contact and not agent.failure_reason:
        agent.failure_reason = ("Mission time expired before direct contact." if world.time <= 0 else
                                "Maximum action limit reached before direct contact.")
    best=max(agent.belief,key=agent.belief.get); confidence=agent.belief[best]
    located=contact
    report={"mission":"WALL-E and EVE's Signal","outcome":"EVE located" if located else "signal search incomplete","failure_reason":None if located else (agent.failure_reason or "Mission ended before direct contact."),"evidence_found":agent.found,"inferred_location":{"coordinates":list(world.eve) if located else list(best),"confidence":round(confidence,3)},"inferred_status":world.c["signal"]["eve_status"] if located else ("powered and sheltered; likely stable" if any(c["type"]=="status" for c in agent.found) else "unknown"),"recommended_next_step":"Approach EVE carefully and initiate recovery protocol." if located else "Review resource budget or mission layout, then retry.","resources_remaining":{"battery":world.battery,"time":world.time},"actions":len(agent.log)}
    (outdir/"trace.json").write_text(json.dumps(agent.log,indent=2),encoding="utf-8"); (outdir/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); (outdir/"map.txt").write_text(render(cfg,agent.log),encoding="utf-8")
    print(json.dumps(report,indent=2)); print("\nPath map (S=start, *=visited, +=charger, #=obstacle):\n"+render(cfg,agent.log))

def reason(action,agent,obs):
    if action["type"]=="analyze": return "Inspect this location for discoverable evidence."
    if action["type"]=="scan": return "Refresh noisy signal evidence to update the spatial belief."
    if action["type"]=="recharge": return "Restore battery at a charger before continuing."
    if obs.get("clue"): return "New clue changes the search strategy; preserve energy and follow its evidence."
    return f"Move {action['direction']} toward the maximum-posterior candidate {max(agent.belief,key=agent.belief.get)}."

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("config",nargs="?",default="mission.json"); ap.add_argument("--out",default="outputs/sample_run"); args=ap.parse_args(); run(args.config,args.out)
