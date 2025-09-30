from __future__ import annotations
from dataclasses import dataclass
import json
from typing import List, Literal, Optional

Level = Literal["none", "yellow", "red"]

@dataclass
class AlertEvent:
    t: float           # timestamp (s)
    level: Level       # "yellow" | "red"
    message: str       # human text

class AlertScheduler:
    def __init__(self, min_interval_s: float = 2.0):
        self.min_interval_s = float(min_interval_s)
        self.events: List[AlertEvent] = []
        self._last_emit_t: float = -1e9
        self._last_level: Level = "none"

    def _msg_for(self, level: Level) -> str:
        if level == "red":
            return "Danger"
        if level == "yellow":
            return "Be careful"
        return ""

    def update(self, t_s: float, level: Level, debug_print=print):
        if level not in ("red", "yellow", "none"):
            level = "none"

        if level == "none":
            self._last_level = "none"
            return

        if level == "red" and self._last_level != "red":
            ev = AlertEvent(t=float(t_s), level="red", message=self._msg_for("red"))
            self.events.append(ev)
            self._last_emit_t = float(t_s)
            self._last_level = "red"
            debug_print(f"[ALERT] t={t_s:.2f}s EMIT RED")
            return

        if level == self._last_level:
            if (t_s - self._last_emit_t) >= self.min_interval_s:
                ev = AlertEvent(t=float(t_s), level=level, message=self._msg_for(level))
                self.events.append(ev)
                self._last_emit_t = float(t_s)
                debug_print(f"[ALERT] t={t_s:.2f}s EMIT {level.upper()} (periodic)")
            return

        if level == "yellow" and self._last_level == "red":
            if (t_s - self._last_emit_t) >= self.min_interval_s:
                ev = AlertEvent(t=float(t_s), level="yellow", message=self._msg_for("yellow"))
                self.events.append(ev)
                self._last_emit_t = float(t_s)
                self._last_level = "yellow"
                debug_print(f"[ALERT] t={t_s:.2f}s EMIT YELLOW (downshift)")
            else:
                self._last_level = "yellow"
            return

        if level == "yellow" and self._last_level in ("none",):
            ev = AlertEvent(t=float(t_s), level="yellow", message=self._msg_for("yellow"))
            self.events.append(ev)
            self._last_emit_t = float(t_s)
            self._last_level = "yellow"
            debug_print(f"[ALERT] t={t_s:.2f}s EMIT YELLOW (start)")
            return

        self._last_level = level

    def save_json(self, out_json_path: str):
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump([e.__dict__ for e in self.events], f, ensure_ascii=False, indent=2)
