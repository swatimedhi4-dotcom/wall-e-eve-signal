"""Pygame front end for the existing WALL-E signal search simulation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import pygame
except ImportError as exc:
    raise SystemExit(
        "Pygame is required. Install it with: python -m pip install pygame"
    ) from exc

from agent import Agent, World, reason, render

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "mission.json"
OUTPUT_DIR = ROOT / "outputs" / "pygame_run"
CELL = 70
MARGIN = 28
PANEL_W = 365
TOP_H = 100
WIDTH = MARGIN * 2 + 9 * CELL + PANEL_W
HEIGHT = TOP_H + MARGIN * 2 + 7 * CELL + 35
FPS = 60
STEP_INTERVAL = 0.55

BG = (18, 25, 35)
PANEL = (29, 39, 52)
GRID = (60, 75, 91)
TEXT = (237, 241, 245)
MUTED = (165, 178, 192)
WALL = (82, 96, 111)
YELLOW = (250, 196, 75)
TEAL = (67, 205, 190)
RED = (231, 93, 93)
GREEN = (110, 211, 130)
BLUE = (89, 162, 240)


class SignalGame:
    def __init__(self, config_path: Path = DEFAULT_CONFIG):
        self.config_path = Path(config_path)
        self.cfg = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        self.config_path = Path(config_path).resolve()
        self.reset()
        pygame.init()
        pygame.display.set_caption("WALL-E and EVE's Signal")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 19)
        self.small = pygame.font.SysFont("segoeui", 15)
        self.title_font = pygame.font.SysFont("segoeui", 31, bold=True)
        self.button_font = pygame.font.SysFont("segoeui", 17, bold=True)
        self.buttons = self._make_buttons()
        self.accumulator = 0.0
        self.running = True

    def reset(self):
        # Reload on every reset so edits to mission.json apply to a fresh mission.
        self.cfg = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        configured_eve = tuple(self.cfg["signal"]["eve_location"])
        print(f"[DEBUG] Loaded EVE coordinates from {self.config_path}: {configured_eve}")
        self.world = World(self.cfg)
        self.agent = Agent(self.cfg)
        self.initial_obs = {
            "position": list(self.world.pos), "battery": self.world.battery,
            "time": self.world.time, "tick": 0,
        }
        self.current_obs = self.initial_obs
        self.trace = []
        self.path = [tuple(self.world.pos)]
        self.started = False
        self.paused = False
        self.terminal = None
        self.latest_action = None
        self.latest_reason = "Press Start when you are ready to begin the autonomous search."
        self.latest_signal = None
        self.last_clue = None
        self.eve_found = False
        self.accumulator = 0.0

    def _make_buttons(self):
        y = HEIGHT - 73
        x = MARGIN
        return {
            "start": pygame.Rect(x, y, 145, 45),
            "pause": pygame.Rect(x + 160, y, 190, 45),
            "restart": pygame.Rect(x + 365, y, 150, 45),
        }

    def _button_text(self, key):
        if key == "start":
            return "Start" if not self.started else "Running"
        if key == "pause":
            return "Resume" if self.paused else "Pause"
        return "Restart"

    def handle_click(self, point):
        for key, rect in self.buttons.items():
            if rect.collidepoint(point):
                if key == "start" and not self.started and self.terminal is None:
                    self.started = True
                    self.paused = False
                elif key == "pause" and self.started and self.terminal is None:
                    self.paused = not self.paused
                elif key == "restart":
                    self.reset()
                return

    def do_action(self):
        if self.terminal is not None:
            return
        limit = self.cfg.get("max_steps", 60)
        if len(self.trace) >= limit:
            self.finish(False, "Action limit reached before EVE was found.")
            return
        action = self.agent.choose(self.current_obs)
        if action is None:
            self.finish(self.eve_found, "Agent ended its plan.")
            return
        obs = self.world.observe(action)
        decision_text = reason(action, self.agent, obs)
        if obs.get("clue"):
            decision_text = f"Clue discovered: {obs['clue']['text']}"
        if action["type"] == "analyze":
            # The legacy runner marks every analyzed location inspected to avoid repeat analysis.
            self.agent.inspected.add(tuple(self.agent.pos))
        entry = {
            "step": len(self.trace) + 1,
            "action": action,
            "observation": obs,
        }
        self.agent.update(obs)
        entry["decision"] = {"strategy": self.agent.strategy, "reasoning": decision_text}
        self.agent.log.append(entry)
        self.trace.append(entry)
        self.latest_action = action
        self.latest_reason = decision_text
        self.current_obs = obs
        self.path.append(tuple(obs.get("position", self.world.pos)))
        if "signal_strength" in obs:
            self.latest_signal = obs["signal_strength"]
        if obs.get("clue"):
            self.last_clue = obs["clue"]
        if obs.get("contact"):
            self.eve_found = True
            self.finish(True, "EVE's location confirmed by direct contact.")
        elif not obs.get("ok"):
            self.finish(False, obs.get("reason", "Resource limit reached."))
        elif self.world.time <= 0:
            self.finish(False, "Mission time expired before EVE was found.")

    def finish(self, success, message):
        if self.terminal is not None:
            return
        self.terminal = "success" if success else "failure"
        self.latest_reason = message
        best = max(self.agent.belief, key=self.agent.belief.get)
        report = {
            "mission": "WALL-E and EVE's Signal",
            "outcome": "EVE located" if success else "signal search incomplete",
            "evidence_found": self.agent.found,
            "inferred_location": {
                "coordinates": list(self.world.eve) if success else list(best),
                "confidence": 0.995 if success else round(self.agent.belief[best], 3),
            },
            "inferred_status": (
                self.cfg["signal"]["eve_status"] if success else
                ("powered and sheltered; likely stable" if any(c["type"] == "status" for c in self.agent.found) else "unknown")
            ),
            "recommended_next_step": (
                "Approach EVE carefully and initiate recovery protocol." if success else
                "Recharge, then scan from an unvisited position toward the highest-probability region."
            ),
            "resources_remaining": {"battery": self.world.battery, "time": self.world.time},
            "actions": len(self.trace),
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "trace.json").write_text(json.dumps(self.trace, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "map.txt").write_text(render(self.cfg, self.trace), encoding="utf-8")

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.started and self.terminal is None:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset()
            if self.started and not self.paused and self.terminal is None:
                self.accumulator += dt
                if self.accumulator >= STEP_INTERVAL:
                    self.accumulator %= STEP_INTERVAL
                    self.do_action()
            self.draw()
            pygame.display.flip()
        pygame.quit()

    def draw(self):
        self.screen.fill(BG)
        self._text("WALL-E and EVE's Signal", MARGIN, 24, self.title_font, TEXT)
        self._text("Autonomous search simulation", MARGIN, 62, self.small, MUTED)
        grid_x, grid_y = MARGIN, TOP_H
        for y in range(self.world.h):
            for x in range(self.world.w):
                rect = pygame.Rect(grid_x + x * CELL, grid_y + y * CELL, CELL, CELL)
                pygame.draw.rect(self.screen, GRID, rect, 1)
                p = (x, y)
                inner = rect.inflate(-8, -8)
                if p in self.world.obstacles:
                    pygame.draw.rect(self.screen, WALL, inner, border_radius=8)
                    pygame.draw.line(self.screen, (105, 118, 132), inner.topleft, inner.bottomright, 2)
                    pygame.draw.line(self.screen, (105, 118, 132), inner.topright, inner.bottomleft, 2)
                elif p in self.world.chargers:
                    pygame.draw.rect(self.screen, (38, 100, 87), inner, border_radius=9)
                    self._center_text("+", rect.centerx, rect.centery, self.title_font, TEAL)
                if p == tuple(self.cfg["start"]):
                    self._center_text("S", rect.centerx, rect.centery, self.button_font, MUTED)
                for clue in self.agent.found:
                    if p == tuple(clue["position"]):
                        pygame.draw.circle(self.screen, YELLOW, rect.center, 11)
                        self._center_text("!", rect.centerx, rect.centery, self.button_font, BG)
                if self.eve_found and p == self.world.eve:
                    pygame.draw.circle(self.screen, GREEN, rect.center, 20)
                    self._center_text("E", rect.centerx, rect.centery, self.button_font, BG)
        # Draw only the explored route and live robot; hidden target and clues remain invisible.
        points = [self._cell_center(p, grid_x, grid_y) for p in self.path]
        if len(points) > 1:
            pygame.draw.lines(self.screen, (119, 149, 177), False, points, 3)
        px, py = self._cell_center(self.world.pos, grid_x, grid_y)
        pygame.draw.circle(self.screen, BLUE, (px, py), 22)
        self._center_text("W", px, py, self.button_font, TEXT)
        self.draw_panel(grid_x + self.world.w * CELL + 26, grid_y)
        if self.terminal:
            self.draw_end_overlay()
        self.draw_buttons()

    def _cell_center(self, p, gx, gy):
        return gx + p[0] * CELL + CELL // 2, gy + p[1] * CELL + CELL // 2

    def draw_panel(self, x, y):
        rect = pygame.Rect(x, y, PANEL_W - 25, 7 * CELL)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=12)
        tx, ty = x + 20, y + 18
        self._text("MISSION STATUS", tx, ty, self.button_font, TEAL); ty += 40
        rows = [
            ("Battery", f"{self.world.battery} / {self.cfg['resources']['battery']}"),
            ("Time remaining", str(self.world.time)),
            ("Actions", str(len(self.trace))),
            ("Position", f"{self.world.pos[0]}, {self.world.pos[1]}"),
            ("Signal", "not scanned" if self.latest_signal is None else f"{self.latest_signal:.3f}"),
            ("Strategy", self.agent.strategy),
        ]
        for label, value in rows:
            self._text(label, tx, ty, self.small, MUTED); ty += 20
            ty = self._wrapped(value, tx, ty, PANEL_W - 70, self.font, TEXT) + 8
        self._text("LATEST DECISION", tx, ty + 2, self.button_font, YELLOW); ty += 30
        ty = self._wrapped(self.latest_reason, tx, ty, PANEL_W - 70, self.small, TEXT) + 12
        self._text("DISCOVERED CLUES", tx, ty, self.button_font, YELLOW); ty += 26
        if not self.agent.found:
            self._text("None yet", tx, ty, self.small, MUTED)
        else:
            for clue in self.agent.found[-3:]:
                text = f"({clue['position'][0]}, {clue['position'][1]}) {clue['type']}: {clue['text']}"
                ty = self._wrapped(text, tx, ty, PANEL_W - 70, self.small, TEXT) + 5

    def draw_buttons(self):
        for key, rect in self.buttons.items():
            enabled = not (key == "start" and self.started)
            color = (45, 117, 102) if key == "start" and enabled else (54, 86, 119) if key == "pause" else (78, 83, 94)
            if not enabled:
                color = (46, 61, 74)
            pygame.draw.rect(self.screen, color, rect, border_radius=9)
            self._center_text(self._button_text(key), rect.centerx, rect.centery, self.button_font, TEXT if enabled else MUTED)

    def draw_end_overlay(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 13, 20, 195))
        self.screen.blit(overlay, (0, 0))
        success = self.terminal == "success"
        heading = "EVE FOUND" if success else "MISSION ENDED"
        detail = "Direct contact confirmed. Recovery is the recommended next step." if success else self.latest_reason
        self._center_text(heading, WIDTH // 2, HEIGHT // 2 - 35, self.title_font, GREEN if success else RED)
        self._center_text(detail, WIDTH // 2, HEIGHT // 2 + 5, self.small, TEXT)
        self._center_text("Press Restart to play this mission again.", WIDTH // 2, HEIGHT // 2 + 35, self.small, MUTED)

    def _text(self, value, x, y, font, color):
        surface = font.render(str(value), True, color)
        self.screen.blit(surface, (x, y))

    def _center_text(self, value, x, y, font, color):
        surface = font.render(str(value), True, color)
        self.screen.blit(surface, surface.get_rect(center=(x, y)))

    def _wrapped(self, value, x, y, width, font, color):
        words = str(value).split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if font.size(candidate)[0] <= width:
                line = candidate
            else:
                self._text(line, x, y, font, color)
                y += font.get_linesize()
                line = word
        if line:
            self._text(line, x, y, font, color)
            y += font.get_linesize()
        return y


def main():
    config = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CONFIG
    if not config.is_file():
        raise SystemExit(f"Mission config not found: {config}")
    SignalGame(config).run()


if __name__ == "__main__":
    main()
