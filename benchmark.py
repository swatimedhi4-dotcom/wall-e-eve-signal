"""Reproducible, headless comparison of the information planner and a greedy baseline."""
import argparse
import copy
import json
import random
import statistics
import time
from pathlib import Path

from agent import Agent, World, a_star


class GreedyBaseline(Agent):
    """Reconstructed earlier policy: inspect, scan every third turn, chase MAP."""
    def choose(self, obs):
        if obs.get("contact") or self.contact_location is not None:
            return None
        if obs.get("tick", 0) >= self.max_steps:
            self.failure_reason = "Maximum action limit reached."
            return None
        move_cost = self.resources.get("move_cost", 1)
        battery, time_left = obs["battery"], obs["time"]
        route, charger = self._path_to_charger(self.pos)
        reserve = max(0, len(route)-1) * move_cost if route else 0
        if route and battery <= reserve + move_cost:
            self.last_decision_reason = "Returning to charger under the baseline reserve rule."
            if self.pos == charger:
                return {"type": "recharge"}
            return self._move_along(route)
        if self.pos not in self.inspected and battery >= self.resources.get("analyze_cost", 1):
            self.last_decision_reason = "Baseline inspects each visited location once."
            return {"type": "analyze"}
        if obs.get("tick", 0) % 3 == 0 and battery >= self.resources.get("scan_cost", 2):
            return {"type": "scan"}
        target = max(self.belief, key=self.belief.get)
        path = a_star(self.pos, target, self.width, self.height, self.obstacles)
        if len(path) > 1 and battery >= move_cost and time_left >= move_cost:
            self.last_decision_reason = f"Baseline follows maximum-posterior cell {target}."
            return self._move_along(path)
        if self.pos in self.chargers and battery < self.resources["battery"]:
            return {"type": "recharge"}
        self.failure_reason = "Baseline has no feasible greedy move."
        return None


def make_missions(count=50, seed=81024):
    rng = random.Random(seed)
    template = json.loads((Path(__file__).resolve().parent / "mission.json").read_text(encoding="utf-8-sig"))
    missions = []
    for i in range(count):
        cfg = copy.deepcopy(template)
        cfg.update(width=8, height=8, seed=seed + i, max_steps=70)
        impossible = i % 10 == 9
        start = rng.choice([(0, 0), (0, 7), (1, 0), (1, 7)])
        obstacles = set()
        # Scatter safe obstacles while preserving a walkable outer ring.
        for y in range(1, 7):
            for x in range(1, 7):
                if (x, y) != start and rng.random() < 0.11:
                    obstacles.add((x, y))
        if impossible:
            obstacles.update((3, y) for y in range(8))
            obstacles.discard(start)
        obstacles.discard((2, 2))
        obstacles.discard((6, 5))
        free = [(x, y) for y in range(8) for x in range(8)
                if (x, y) not in obstacles and (x, y) != start]
        if impossible:
            targets = [p for p in free if p[0] >= 4]
            target = rng.choice(targets)
        else:
            targets = [p for p in free if a_star(start, p, 8, 8, obstacles)]
            target = rng.choice(targets)
        cfg["start"] = list(start)
        # Do not place any public or hidden object on terrain.
        charger_candidates = [p for p in free if p != target and
                              (not impossible or p[0] < 3)]
        chargers = [list(rng.choice(charger_candidates))]
        cfg["terrain"] = {"obstacles": [list(p) for p in sorted(obstacles)],
                          "chargers": [list(p) for p in chargers]}
        cfg["signal"]["eve_location"] = list(target)
        cfg["signal"]["noise"] = rng.choice([0.03, 0.08, 0.14, 0.22])
        cfg["signal"]["bearing_uncertainty"] = rng.choice([0.05, 0.12, 0.25, 0.4])
        decoy_candidates = [p for p in free if p != target]
        decoy = rng.choice(decoy_candidates)
        cfg["signal"]["decoys"] = ([{"location": list(decoy),
                                         "strength": rng.choice([0.55, 0.8, 1.0]), "label": "benchmark echo"}]
                                     if i % 3 else [])
        cfg["initial_signal"] = {"direction": rng.choice(["north", "east", "south", "west"]),
                                 "strength": round(rng.uniform(0.15, 0.7), 2), "timestamp": 0}
        cfg["resources"].update(battery=rng.choice([35, 45, 55, 65]), time=rng.choice([90, 130, 180]),
                                recharge_amount=10)
        cfg["clues"] = [
            {"at": [2, 2], "type": "directional", "text": "Uncertain tracks detected."},
            {"at": [6, 5], "type": "status", "text": "A weak powered signal is nearby."},
            {"at": list(target), "type": "confirmation", "text": "EVE located."},
        ]
        path = a_star(start, target, 8, 8, obstacles)
        if impossible and path:
            raise RuntimeError("Generated mission intended to be impossible is connected")
        if not impossible and not path:
            # The safe perimeter should connect; fail loudly rather than mislabel.
            raise RuntimeError("Generated solvable mission is disconnected")
        missions.append({"id": f"mission-{i+1:03d}", "solvable": not impossible, "config": cfg})
    return missions


def evaluate(mission, agent_type):
    world = World(mission["config"])
    agent = agent_type(world.agent_config())
    obs = {"position": list(world.pos), "battery": world.battery,
           "time": world.time, "tick": 0}
    started = time.perf_counter()
    energy = 0
    recharges = 0
    for _ in range(mission["config"].get("max_steps", 60)):
        action = agent.choose(obs)
        if action is None:
            break
        before = world.battery
        obs = world.observe(action)
        if action["type"] == "recharge":
            recharges += int(obs.get("recharged", False))
        else:
            energy += max(0, before - world.battery)
        agent.update(obs)
        if obs.get("contact") or not obs.get("ok") or world.time <= 0:
            break
    success = bool(obs.get("contact"))
    estimate = max(agent.belief, key=agent.belief.get)
    distance = abs(estimate[0]-world.eve[0]) + abs(estimate[1]-world.eve[1])
    failure = None
    if not success:
        reason = (agent.failure_reason or obs.get("reason") or "").lower()
        if not mission["solvable"]:
            failure = "unreachable_goal"
        elif world.time <= 0 or "time" in reason:
            failure = "time_expiration"
        elif "action limit" in reason or world.tick >= mission["config"].get("max_steps", 60):
            failure = "action_limit"
        elif "unreachable" in reason or "no path" in reason:
            failure = "unreachable_goal"
        elif "information gain" in reason or "stagn" in reason or "no candidate" in reason:
            failure = "search_stagnation"
        elif "battery" in reason or "resource" in reason or "route" in reason or "feasible scan" in reason:
            failure = "resource_or_route_infeasible"
        else:
            failure = "planner_terminated"
    return {"mission_id": mission["id"], "solvable": mission["solvable"], "success": success,
            "actions": world.tick, "energy_consumed": energy, "unique_locations_investigated": len(agent.inspected),
            "prediction_distance": distance, "recharges": recharges, "failure_category": failure,
            "failure_reason": None if success else (agent.failure_reason or obs.get("reason") or "No terminal reason reported."),
            "noise": mission["config"]["signal"]["noise"],
            "has_decoy": bool(mission["config"]["signal"].get("decoys")),
            "elapsed_seconds": round(time.perf_counter()-started, 5)}


def summarize(rows):
    successes = [r for r in rows if r["success"]]
    failed = [r for r in rows if not r["success"]]
    by_noise = {}
    for noise in sorted({r["noise"] for r in rows}):
        group = [r for r in rows if r["noise"] == noise]
        by_noise[str(noise)] = {"missions": len(group), "successes": sum(r["success"] for r in group),
                                "success_rate": round(sum(r["success"] for r in group)/len(group), 4)}
    by_decoy = {}
    for has_decoy in (False, True):
        group = [r for r in rows if r["has_decoy"] is has_decoy]
        if group:
            by_decoy["with_decoy" if has_decoy else "without_decoy"] = {
                "missions": len(group), "successes": sum(r["success"] for r in group),
                "success_rate": round(sum(r["success"] for r in group)/len(group), 4)}
    return {"missions": len(rows), "successes": len(successes),
            "success_rate": round(len(successes)/len(rows), 4) if rows else 0,
            "success_rate_on_solvable": round(sum(r["success"] for r in rows if r["solvable"])/max(1, sum(r["solvable"] for r in rows)), 4),
            "mean_actions_successful": round(statistics.mean(r["actions"] for r in successes), 2) if successes else None,
            "mean_actions_failed": round(statistics.mean(r["actions"] for r in failed), 2) if failed else None,
            "mean_actions_all": round(statistics.mean(r["actions"] for r in rows), 2) if rows else None,
            "mean_energy_per_success": round(statistics.mean(r["energy_consumed"] for r in successes), 2) if successes else None,
            "mean_locations_investigated": round(statistics.mean(r["unique_locations_investigated"] for r in rows), 2) if rows else None,
            "mean_prediction_distance": round(statistics.mean(r["prediction_distance"] for r in rows), 2) if rows else None,
            "missions_with_recharge": sum(r["recharges"] > 0 for r in rows),
            "success_by_noise": by_noise, "success_by_decoy_presence": by_decoy,
            "failure_categories": {category: sum(r["failure_category"] == category for r in rows)
                                   for category in sorted({r["failure_category"] for r in rows if r["failure_category"]})}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=81024)
    parser.add_argument("--out", default="outputs/benchmark")
    args = parser.parse_args()
    scenarios = make_missions(args.count, args.seed)
    results = {"seed": args.seed, "count": args.count, "baseline": [], "information_agent": []}
    for mission in scenarios:
        results["baseline"].append(evaluate(mission, GreedyBaseline))
        results["information_agent"].append(evaluate(mission, Agent))
    results["summary"] = {key: summarize(results[key]) for key in ("baseline", "information_agent")}
    results["summary"]["solvable_missions"] = sum(m["solvable"] for m in scenarios)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Deterministic benchmark", "", f"Seed: {args.seed}; missions: {args.count}; solvable: {results['summary']['solvable_missions']}",
             "", "The baseline reconstructs the earlier MAP-chasing policy (inspect, periodic scan, A* route); both agents share the same world, observations, seeds, and contact-only success rule.", "",
             "| Policy | Success | Solvable success | Actions success / failed / all | Energy per success | Mean explored cells | Mean prediction distance | Recharged missions |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in ("baseline", "information_agent"):
        s = results["summary"][label]
        lines.append(f"| {label} | {s['successes']}/{s['missions']} ({s['success_rate']:.1%}) | {s['success_rate_on_solvable']:.1%} | {s['mean_actions_successful']} / {s['mean_actions_failed']} / {s['mean_actions_all']} | {s['mean_energy_per_success']} | {s['mean_locations_investigated']} | {s['mean_prediction_distance']} | {s['missions_with_recharge']} |")
    lines.extend(["", "## Success by signal noise", "", "| Policy | Noise | Successes / missions | Rate |", "|---|---:|---:|---:|"])
    for label in ("baseline", "information_agent"):
        for noise, group in results["summary"][label]["success_by_noise"].items():
            lines.append(f"| {label} | {noise} | {group['successes']} / {group['missions']} | {group['success_rate']:.1%} |")
    lines.extend(["", "## Success by decoy presence", "", "| Policy | Decoys | Successes / missions | Rate |", "|---|---|---:|---:|"])
    for label in ("baseline", "information_agent"):
        for category, group in results["summary"][label]["success_by_decoy_presence"].items():
            lines.append(f"| {label} | {category.replace('_', ' ')} | {group['successes']} / {group['missions']} | {group['success_rate']:.1%} |")
    (output / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
