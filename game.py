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
HEIGHT = TOP_H + MARGIN * 2 + 7 * CELL + 110
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
        self.coordinate_fields = self._make_coordinate_fields()
        self.accumulator = 0.0
        self.running = True

    def reset(self):
        # Reload on every reset so edits to mission.json apply to a fresh mission.
        self.cfg = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        configured_eve = tuple(self.cfg["signal"]["eve_location"])
        print(f"[DEBUG] Loaded EVE coordinates from {self.config_path}: {configured_eve}")
        # Blank inputs keep the configured target hidden until the agent discovers it.
        self.eve_x_text = ""
        self.eve_y_text = ""
        self.config_error = ""
        self.active_coordinate = None
        self.world = World(self.cfg)
        self.agent = Agent(self.world.agent_config())
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
        self.show_heatmap = getattr(self, "show_heatmap", False)
        self.accumulator = 0.0

    def _make_buttons(self):
        y = HEIGHT - 145
        x = MARGIN
        return {
            "start": pygame.Rect(x, y, 145, 45),
            "pause": pygame.Rect(x + 160, y, 190, 45),
            "restart": pygame.Rect(x + 365, y, 150, 45),
            "apply": pygame.Rect(x + 730, y, 165, 45),
        }

    def _make_coordinate_fields(self):
        y = HEIGHT - 145
        return {
            "x": pygame.Rect(MARGIN + 558, y, 50, 45),
            "y": pygame.Rect(MARGIN + 657, y, 50, 45),
        }

    def _button_text(self, key):
        if key == "start":
            return "Start" if not self.started else "Running"
        if key == "pause":
            return "Resume" if self.paused else "Pause"
        return "Restart"

    def handle_click(self, point):
        for name, rect in self.coordinate_fields.items():
            if rect.collidepoint(point):
                self.active_coordinate = name
                if name == "x":
                    self.eve_x_text = ""
                else:
                    self.eve_y_text = ""
                return
        self.active_coordinate = None
        if self.buttons["apply"].collidepoint(point):
            self.apply_eve_coordinates()
            return
        for key, rect in self.buttons.items():
            if key == "apply":
                continue
            if rect.collidepoint(point):
                if key == "start" and not self.started and self.terminal is None:
                    self.started = True
                    self.paused = False
                elif key == "pause" and self.started and self.terminal is None:
                    self.paused = not self.paused
                elif key == "restart":
                    self.reset()
                return

    def apply_eve_coordinates(self):
        """Validate, persist, and immediately start a fresh mission at the chosen cell."""
        try:
            x, y = int(self.eve_x_text), int(self.eve_y_text)
        except ValueError:
            self.config_error = "Enter whole-number X and Y coordinates."
            return False
        if not (0 <= x < self.cfg["width"] and 0 <= y < self.cfg["height"]):
            self.config_error = f"Coordinates must be inside the {self.cfg['width']} by {self.cfg['height']} grid."
            return False
        if (x, y) in {tuple(p) for p in self.cfg["terrain"]["obstacles"]}:
            self.config_error = "EVE cannot be placed on an obstacle."
            return False

        updated = dict(self.cfg)
        updated_signal = dict(self.cfg["signal"])
        updated_signal["eve_location"] = [x, y]
        updated["signal"] = updated_signal
        updated["clues"] = [dict(clue) for clue in self.cfg["clues"]]
        for clue in updated["clues"]:
            if clue.get("type") == "confirmation":
                clue["at"] = [x, y]
        try:
            self.config_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            self.config_error = f"Could not save mission.json: {exc}"
            return False

        self.reset()
        self.started = True
        return True

    def do_action(self):
        if self.terminal is not None:
            return
        limit = self.cfg.get("max_steps", 60)
        if len(self.trace) >= limit:
            self.finish(False, "Action limit reached before EVE was found.")
            return
        action = self.agent.choose(self.current_obs)
        if action is None:
            self.finish(self.eve_found, self.agent.failure_reason or "Agent ended its plan.")
            return
        obs = self.world.observe(action)
        decision_text = self.agent.last_decision_reason or reason(action, self.agent, obs)
        if obs.get("clue"):
            decision_text += f" Clue discovered: {obs['clue']['text']}"
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
            "failure_reason": None if success else message,
            "evidence_found": self.agent.found,
            "inferred_location": {
                "coordinates": list(self.world.eve) if success else list(best),
                "confidence": round(self.agent.belief[best], 3),
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
                    if self.active_coordinate and event.key == pygame.K_BACKSPACE:
                        value = self.eve_x_text if self.active_coordinate == "x" else self.eve_y_text
                        value = value[:-1]
                        if self.active_coordinate == "x":
                            self.eve_x_text = value
                        else:
                            self.eve_y_text = value
                    elif self.active_coordinate and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.apply_eve_coordinates()
                    elif self.active_coordinate and event.unicode.isdigit():
                        if self.active_coordinate == "x":
                            self.eve_x_text += event.unicode
                        else:
                            self.eve_y_text += event.unicode
                    elif event.key == pygame.K_SPACE and self.started and self.terminal is None:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_h:
                        self.show_heatmap = not self.show_heatmap
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
        if self.show_heatmap:
            peak = max(self.agent.belief.values(), default=0.0)
            if peak > 0:
                for point, probability in self.agent.belief.items():
                    if probability <= 0 or point in self.world.obstacles:
                        continue
                    intensity = min(155, int(155 * probability / peak))
                    overlay = pygame.Surface((CELL - 8, CELL - 8), pygame.SRCALPHA)
                    overlay.fill((246, 75, 70, intensity))
                    self.screen.blit(overlay, (grid_x + point[0] * CELL + 4, grid_y + point[1] * CELL + 4))
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
        self.draw_coordinate_controls()

    def _cell_center(self, p, gx, gy):
        return gx + p[0] * CELL + CELL // 2, gy + p[1] * CELL + CELL // 2

    def draw_panel(self, x, y):
        rect = pygame.Rect(x, y, PANEL_W - 25, HEIGHT - y - 166)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=12)
        tx, ty = x + 20, y + 18
        self._text("MISSION STATUS", tx, ty, self.button_font, TEAL); ty += 40
        target = self.agent.current_target or self.agent.estimated_target
        rows = [
            ("Battery / time / actions", f"{self.world.battery} / {self.world.time} / {len(self.trace)}"),
            ("Position / signal", f"{self.world.pos} / {'—' if self.latest_signal is None else f'{self.latest_signal:.3f}'}"),
            ("Estimate / confidence", f"{target} / {self.agent.confidence:.1%}"),
            ("Belief entropy", f"{self.agent.entropy:.2f} bits"),
            ("Last action", "idle" if not self.latest_action else self.latest_action["type"] + (f" {self.latest_action['direction']}" if "direction" in self.latest_action else "")),
            ("Strategy", self.agent.strategy),
        ]
        for label, value in rows:
            self._text(label, tx, ty, self.small, MUTED)
            ty = self._wrapped(value, tx + 124, ty, PANEL_W - 194, self.small, TEXT) + 6
        self._text("LATEST DECISION", tx, ty + 2, self.button_font, YELLOW); ty += 30
        ty = self._wrapped(self.latest_reason, tx, ty, PANEL_W - 70, self.small, TEXT) + 12
        self._text("DISCOVERED CLUES", tx, ty, self.button_font, YELLOW); ty += 26
        if not self.agent.found:
            self._text("None yet", tx, ty, self.small, MUTED)
        else:
            for clue in self.agent.found[-2:]:
                detail = clue["text"]
                if len(detail) > 66:
                    detail = detail[:63].rstrip() + "..."
                text = f"({clue['position'][0]}, {clue['position'][1]}) {clue['type']}: {detail}"
                ty = self._wrapped(text, tx, ty, PANEL_W - 70, self.small, TEXT) + 5
        self._text(f"Heatmap: {'ON' if self.show_heatmap else 'OFF'} (H)", tx, rect.bottom - 27, self.small, TEAL)

    def draw_buttons(self):
        for key, rect in self.buttons.items():
            if key == "apply":
                continue
            enabled = not (key == "start" and self.started)
            color = (45, 117, 102) if key == "start" and enabled else (54, 86, 119) if key == "pause" else (78, 83, 94)
            if not enabled:
                color = (46, 61, 74)
            pygame.draw.rect(self.screen, color, rect, border_radius=9)
            self._center_text(self._button_text(key), rect.centerx, rect.centery, self.button_font, TEXT if enabled else MUTED)

    def draw_coordinate_controls(self):
        for name, rect in self.coordinate_fields.items():
            self._text(f"EVE {name.upper()}", rect.left - 40, rect.top + 14, self.small, MUTED)
            pygame.draw.rect(self.screen, (40, 52, 68), rect, border_radius=7)
            active = self.active_coordinate == name
            pygame.draw.rect(self.screen, TEAL if active else GRID, rect, 2, border_radius=7)
            value = self.eve_x_text if name == "x" else self.eve_y_text
            self._center_text(value or "?", rect.centerx, rect.centery, self.font, TEXT if value else MUTED)
        apply_rect = self.buttons["apply"]
        pygame.draw.rect(self.screen, (45, 117, 102), apply_rect, border_radius=9)
        self._center_text("Apply & Restart", apply_rect.centerx, apply_rect.centery, self.small, TEXT)
        if self.config_error:
            self._text(self.config_error, MARGIN + 545, apply_rect.bottom + 7, self.small, RED)

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
