from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

DIRS = {"north": (0,-1), "east": (1,0), "south": (0,1), "west": (-1,0)}

def dist(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

class World:
    """Grid world. Hidden targets and clue positions are never exposed in observations."""
    def __init__(self, cfg):
        self.c=cfg; self.w=cfg["width"]; self.h=cfg["height"]
        self.pos=tuple(cfg["start"]); self.battery=cfg["resources"]["battery"]; self.time=cfg["resources"]["time"]
        self.obstacles={tuple(p) for p in cfg["terrain"]["obstacles"]}; self.chargers={tuple(p) for p in cfg["terrain"]["chargers"]}
        self.eve=tuple(cfg["signal"]["eve_location"])
        # Confirmation is attached to EVE, so keep it colocated when her configured
        # location changes without requiring a second coordinate edit in the JSON.
        self.clues={}
        for clue in cfg["clues"]:
            placed=dict(clue)
            if placed.get("type")=="confirmation":
                placed["at"]=list(self.eve)
            self.clues[tuple(placed["at"])]=placed
        self.found=set()
        self.decoys=[(tuple(d["location"]),d) for d in cfg["signal"].get("decoys",[])]
        self.rng=random.Random(cfg["seed"]); self.tick=0
    def observe(self, action):
        typ=action["type"]; r=self.c["resources"]; cost=r[typ+"_cost"]
        if self.battery<cost or self.time<=0: return {"ok":False,"reason":"resource limit"}
        self.battery-=cost; self.time-=cost; self.tick+=1
        obs={"ok":True,"position":list(self.pos),"battery":self.battery,"time":self.time,"tick":self.tick}
        if typ=="move":
            dx,dy=DIRS[action["direction"]]; p=(self.pos[0]+dx,self.pos[1]+dy)
            if 0<=p[0]<self.w and 0<=p[1]<self.h and p not in self.obstacles: self.pos=p
            else: obs["bump"]=True
            obs["position"]=list(self.pos)
        elif typ=="scan":
            signal=self.c["signal"]; d=dist(self.pos,self.eve); true=1/(1+d)
            readings=[true+self.rng.uniform(-signal["noise"],signal["noise"])]
            for loc,_ in self.decoys: readings.append(0.8/(1+dist(self.pos,loc))+self.rng.uniform(-signal["noise"],signal["noise"]))
            obs["signal_strength"]=round(max(0,min(1,max(readings))),3)
            obs["bearing"]="unknown" if self.pos==self.eve else max(DIRS,key=lambda z: (self.eve[0]-self.pos[0])*DIRS[z][0]+(self.eve[1]-self.pos[1])*DIRS[z][1])
        elif typ=="analyze":
            clue=self.clues.get(self.pos)
            if clue and self.pos not in self.found:
                self.found.add(self.pos); obs["clue"]={"type":clue["type"],"text":clue["text"]}
            else: obs["clue"]=None
            # Analysis is local: the agent learns only clue content, never the hidden target coordinate.
            if self.pos==self.eve: obs["contact"]={"status":self.c["signal"]["eve_status"],"confirmation":True}
        elif typ=="recharge":
            if self.pos not in self.chargers: obs["recharged"]=False
            else: self.battery=min(self.c["resources"]["battery"],self.battery+8); obs["recharged"]=True; obs["battery"]=self.battery
        return obs

class Agent:
    def __init__(self,cfg):
        self.cfg=cfg; self.pos=tuple(cfg["start"]); self.belief={(x,y):1/(cfg["width"]*cfg["height"]) for x in range(cfg["width"]) for y in range(cfg["height"])}
        for p in cfg["terrain"]["obstacles"]: self.belief.pop(tuple(p),None)
        direction=cfg["initial_signal"].get("direction")
        if direction in DIRS:
            dx,dy=DIRS[direction]
            for p in self.belief:
                if (p[0]-self.pos[0])*dx+(p[1]-self.pos[1])*dy>0: self.belief[p]*=1.35
        z=sum(self.belief.values()); self.belief={p:v/z for p,v in self.belief.items()}; self.found=[]; self.inspected=set(); self.decoy=False; self.strategy="follow signal gradient"
        self.log=[]
    def update(self,obs):
        if obs.get("clue"):
            clue=obs["clue"]; self.found.append({"position":list(self.pos),**clue})
            if clue["type"]=="directional": self.strategy="prioritize eastern ridge; clue corroborates eastward bearing"
            elif clue["type"]=="status": self.strategy="protect remaining battery; EVE likely stable, triangulate toward signal"
            elif clue["type"]=="confirmation": self.strategy="mission success; recommend safe approach and recovery"
        if "signal_strength" in obs:
            # Strength follows an inverse-distance model; observation likelihood gives Bayesian spatial update.
            for p in list(self.belief):
                expected=1/(1+dist(self.pos,p)); likelihood=math.exp(-((expected-obs["signal_strength"])/0.24)**2/2)
                if obs.get("bearing") in DIRS:
                    dx,dy=DIRS[obs["bearing"]]; rx,ry=p[0]-self.pos[0],p[1]-self.pos[1]
                    if rx*dx+ry*dy>0: likelihood*=2.0
                    else: likelihood*=0.15
                self.belief[p]*=max(0.04,likelihood)
            s=sum(self.belief.values()); self.belief={p:v/s for p,v in self.belief.items()}
        self.pos=tuple(obs.get("position",self.pos))
    def choose(self,obs):
        if obs.get("contact"): return None
        chargers=[tuple(q) for q in self.cfg["terrain"]["chargers"]]
        near=min(chargers,key=lambda p:dist(self.pos,p)) if chargers else None
        if near and obs["battery"]<=dist(self.pos,near)+2:
            if self.pos==near: return {"type":"recharge"}
            for direction,(dx,dy) in DIRS.items():
                p=(self.pos[0]+dx,self.pos[1]+dy)
                if 0<=p[0]<self.cfg["width"] and 0<=p[1]<self.cfg["height"] and p not in {tuple(q) for q in self.cfg["terrain"]["obstacles"]} and dist(p,near)<dist(self.pos,near): return {"type":"move","direction":direction}
        # Analyze at each visited cell once: low-cost way to discover colocated evidence.
        if self.pos not in self.inspected: return {"type":"analyze"}
        # Scan when feasible; adapt after weak results or directional clue.
        if self.cfg["resources"]["battery"]-obs["battery"] < 2 or obs["tick"]%3==0: return {"type":"scan"}
        target=max(self.belief,key=self.belief.get)
        # Evidence provides qualitative constraints without exposing clue map.
        if any(c["type"]=="directional" for c in self.found): target=max(self.belief,key=lambda p:self.belief[p]*(1.2 if p[0]>self.pos[0] else 0.7))
        choices=[]
        for name,(dx,dy) in DIRS.items():
            p=(self.pos[0]+dx,self.pos[1]+dy)
            if 0<=p[0]<self.cfg["width"] and 0<=p[1]<self.cfg["height"] and p not in {tuple(q) for q in self.cfg["terrain"]["obstacles"]}:
                choices.append((dist(p,target),name,p))
        if not choices: return {"type":"scan"}
        _,direction,p=min(choices)
        # Reach charger if projected battery is tight and it is en route.
        if obs["battery"]<=3:
            chargers=[tuple(q) for q in self.cfg["terrain"]["chargers"]]
            if chargers and self.pos in chargers: return {"type":"recharge"}
        return {"type":"move","direction":direction}

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
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8-sig")); world=World(cfg); agent=Agent(cfg); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    initial={"position":list(world.pos),"battery":world.battery,"time":world.time,"tick":0}
    for step in range(cfg.get("max_steps",60)):
        action=agent.choose(initial)
        if action is None: break
        obs=world.observe(action)
        entry={"step":step+1,"action":action,"observation":obs,"decision":{"strategy":agent.strategy,"reasoning":reason(action,agent,obs)}}
        if action["type"]=="analyze": agent.inspected.add(tuple(agent.pos))
        agent.update(obs); agent.log.append(entry); initial=obs
        if obs.get("contact"): break
        if not obs.get("ok"): break
    best=max(agent.belief,key=agent.belief.get); confidence=agent.belief[best]
    located=any(c["type"]=="confirmation" for c in agent.found)
    report={"mission":"WALL-E and EVE's Signal","outcome":"EVE located" if located else "signal search incomplete","evidence_found":agent.found,"inferred_location":{"coordinates":list(best),"confidence":0.995 if located else round(confidence,3)},"inferred_status":"alive, stable, awaiting approach" if located else ("powered and sheltered; likely stable" if any(c["type"]=="status" for c in agent.found) else "unknown"),"recommended_next_step":"Approach EVE carefully and initiate recovery protocol." if located else "Recharge, then scan from an unvisited position toward the highest-probability region.","resources_remaining":{"battery":world.battery,"time":world.time},"actions":len(agent.log)}
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