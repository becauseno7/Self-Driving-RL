"""High-fidelity procedural Pygame visuals for the Neon Highway RL lab."""

from __future__ import annotations

import math

import numpy as np
import pygame

from self_driving_rl.game_env import (
    ACTION_COUNT,
    NeonHighwayEnv,
    TrafficCar,
    decode_action,
    encode_action,
)
from self_driving_rl.metrics import format_duration


class NeonRenderer:
    WIDTH = 1440
    HEIGHT = 810
    ROAD_WIDTH = 720
    ROAD_LEFT = (WIDTH - ROAD_WIDTH) // 2
    EGO_Y = 620
    PIXELS_PER_METER = 6.4

    CYAN = (56, 223, 239)
    PINK = (249, 74, 127)
    GREEN = (75, 224, 157)
    AMBER = (255, 181, 71)
    RED = (255, 73, 99)
    TEXT = (234, 241, 250)
    MUTED = (124, 145, 172)
    PANEL = (8, 16, 30)

    TRAFFIC_COLORS = [
        (245, 81, 111),
        (255, 177, 66),
        (119, 221, 154),
        (123, 145, 255),
        (198, 125, 255),
        (238, 238, 243),
        (50, 188, 210),
        (242, 115, 70),
    ]

    def __init__(self, *, human: bool, fps: int, speed: float = 1.0) -> None:
        pygame.init()
        pygame.font.init()
        self.human = human
        self.fps = fps
        # World seconds per real second. 1.0 plays back as a simulation;
        # training wants fast-forward, which costs interpolated frames.
        self.speed = max(speed, 1e-6)
        self.clock = pygame.time.Clock()
        self.screen = (
            pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            if human
            else pygame.Surface((self.WIDTH, self.HEIGHT))
        )
        if human:
            pygame.display.set_caption("Neon Highway - RL Learning Lab")

        self.font_tiny = pygame.font.SysFont("Segoe UI", 13)
        self.font_small = pygame.font.SysFont("Segoe UI", 15)
        self.font_small_bold = pygame.font.SysFont("Segoe UI", 15, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI", 36, bold=True)
        self.font_speed = pygame.font.SysFont("Segoe UI", 58, bold=True)
        self.background = self._make_background()
        self.paused = False
        self.show_sensors = True
        self.last_crash_episode = -1
        # Fraction of the way through the current simulation step. The world
        # advances 0.1 s per step, so drawing one frame per step would run the
        # simulation at six times real time in 14-pixel jumps.
        self.alpha = 1.0

    def _make_background(self) -> pygame.Surface:
        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        top = np.array([4, 8, 22], dtype=float)
        bottom = np.array([9, 38, 44], dtype=float)
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = tuple((top * (1 - ratio) + bottom * ratio).astype(int))
            pygame.draw.line(surface, color, (0, y), (self.WIDTH, y))

        rng = np.random.default_rng(42)
        for _ in range(75):
            x = int(rng.integers(0, self.WIDTH))
            y = int(rng.integers(0, 330))
            radius = int(rng.integers(1, 3))
            pygame.draw.circle(surface, (51, 111, 135), (x, y), radius)

        side_ranges = [
            (0, self.ROAD_LEFT - 26),
            (self.ROAD_LEFT + self.ROAD_WIDTH + 26, self.WIDTH),
        ]
        for side_left, side_right in side_ranges:
            x = side_left + 8
            while x < side_right - 16:
                width = int(rng.integers(30, 78))
                height = int(rng.integers(110, 390))
                rect = pygame.Rect(x, self.HEIGHT - height, width, height)
                pygame.draw.rect(surface, (6, 14, 29), rect, border_radius=4)
                pygame.draw.line(surface, (15, 47, 65), rect.topleft, rect.topright, 2)
                for window_y in range(rect.top + 14, rect.bottom - 12, 23):
                    for window_x in range(rect.left + 8, rect.right - 6, 16):
                        if rng.random() > 0.52:
                            window_color = (31, 131, 143) if rng.random() > 0.18 else (180, 89, 125)
                            pygame.draw.rect(
                                surface,
                                window_color,
                                (window_x, window_y, 6, 8),
                                border_radius=1,
                            )
                x += width + int(rng.integers(7, 18))
        return surface

    def draw(self, env: NeonHighwayEnv) -> tuple[np.ndarray | None, bool]:
        if self.human and not self._handle_events():
            return None, False
        self._wait_while_paused(env)
        if env.quit_requested:
            return None, False

        new_crash = env.crashed and self.last_crash_episode != env.episode_index
        if self.human and new_crash:
            for frame_index in range(18):
                if not self._handle_events():
                    return None, False
                self._render_frame(env, crash_phase=(frame_index + 1) / 18)
                pygame.display.flip()
                self.clock.tick(60)
            self.last_crash_episode = env.episode_index
        elif self.human:
            # One simulation step covers env.DT of world time. Drawing it as a
            # single frame would play the world back at DT * fps times real
            # speed, so the step is split into interpolated frames instead.
            for frame_index in range(self._frames_per_step(env)):
                if frame_index and not self._handle_events():
                    return None, False
                alpha = (frame_index + 1) / self._frames_per_step(env)
                self._render_frame(
                    env, crash_phase=1.0 if env.crashed else 0.0, alpha=alpha
                )
                pygame.display.flip()
                if self.fps > 0:
                    self.clock.tick(self.fps)
        else:
            self._render_frame(env, crash_phase=1.0 if env.crashed else 0.0)

        frame = None if self.human else self._frame_array()
        return frame, True

    def _frames_per_step(self, env: NeonHighwayEnv) -> int:
        """Frames per simulation step needed to hit the requested playback speed."""
        if self.fps <= 0:
            return 1
        return max(1, round(self.fps * env.DT / self.speed))

    def _render_frame(
        self, env: NeonHighwayEnv, *, crash_phase: float, alpha: float = 1.0
    ) -> None:
        self.alpha = alpha
        self.screen.blit(self.background, (0, 0))
        self._draw_road(env)
        if self.show_sensors:
            self._draw_sensors(env)
        self._draw_traffic(env)
        self._draw_agent(env)
        if env.crashed:
            self._draw_impact_particles(env, crash_phase)
        self._draw_header(env)
        self._draw_left_dashboard(env)
        self._draw_right_dashboard(env)
        self._draw_action_strip(env)
        self._draw_footer(env)

        if env.crashed:
            self._draw_crash_overlay(env, crash_phase)
        elif env.completed:
            self._draw_completion_overlay(env)

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_h:
                    self.show_sensors = not self.show_sensors
        return True

    def _wait_while_paused(self, env: NeonHighwayEnv) -> None:
        while self.human and self.paused:
            if not self._handle_events():
                env.quit_requested = True
                self.paused = False
                return
            self._glass_panel(pygame.Rect(530, 310, 380, 150), alpha=238)
            self._center_text("SIMULATION PAUSED", self.font_large, self.CYAN, 350)
            self._center_text("Press SPACE to continue", self.font_small, self.MUTED, 410)
            pygame.display.flip()
            self.clock.tick(30)

    def _draw_road(self, env: NeonHighwayEnv) -> None:
        shadow = pygame.Rect(self.ROAD_LEFT - 22, 0, self.ROAD_WIDTH + 44, self.HEIGHT)
        pygame.draw.rect(self.screen, (2, 6, 12), shadow)
        pygame.draw.rect(
            self.screen,
            (25, 29, 37),
            (self.ROAD_LEFT, 0, self.ROAD_WIDTH, self.HEIGHT),
        )

        lane_width = self.ROAD_WIDTH / env.LANES
        target_x = int(self.ROAD_LEFT + env.target_lane * lane_width)
        target_overlay = pygame.Surface((int(lane_width), self.HEIGHT), pygame.SRCALPHA)
        target_overlay.fill((46, 211, 230, 10))
        self.screen.blit(target_overlay, (target_x, 0))

        shoulder = 14
        pygame.draw.rect(self.screen, (43, 51, 62), (self.ROAD_LEFT, 0, shoulder, self.HEIGHT))
        pygame.draw.rect(
            self.screen,
            (43, 51, 62),
            (self.ROAD_LEFT + self.ROAD_WIDTH - shoulder, 0, shoulder, self.HEIGHT),
        )
        pygame.draw.line(
            self.screen,
            self.CYAN,
            (self.ROAD_LEFT + shoulder, 0),
            (self.ROAD_LEFT + shoulder, self.HEIGHT),
            2,
        )
        pygame.draw.line(
            self.screen,
            self.PINK,
            (self.ROAD_LEFT + self.ROAD_WIDTH - shoulder - 1, 0),
            (self.ROAD_LEFT + self.ROAD_WIDTH - shoulder - 1, self.HEIGHT),
            2,
        )

        offset = int((self.ego_position(env) * self.PIXELS_PER_METER) % 62)
        for lane in range(1, env.LANES):
            x = int(self.ROAD_LEFT + lane * lane_width)
            for y in range(-62 + offset, self.HEIGHT, 62):
                pygame.draw.line(self.screen, (88, 97, 112), (x, y), (x, y + 30), 3)
                pygame.draw.line(self.screen, (39, 44, 53), (x + 3, y), (x + 3, y + 30), 1)

        for side in (-1, 1):
            x = self.ROAD_LEFT + (21 if side == -1 else self.ROAD_WIDTH - 21)
            color = self.CYAN if side == -1 else self.PINK
            for y in range(-40 + offset, self.HEIGHT, 62):
                pygame.draw.circle(self.screen, color, (x, y), 2)

        marker_offset = int((self.ego_position(env) * self.PIXELS_PER_METER) % 180)
        for y in range(-180 + marker_offset, self.HEIGHT, 180):
            ahead = (self.EGO_Y - y) / self.PIXELS_PER_METER
            marker = f"{int((self.ego_position(env) + ahead) // 100):02d}"
            self._text(marker, self.font_tiny, (67, 76, 90), self.ROAD_LEFT + 28, y)

    def _lane_center_x(self, env: NeonHighwayEnv, lane_position: float) -> float:
        lane_width = self.ROAD_WIDTH / env.LANES
        return self.ROAD_LEFT + lane_width * (lane_position + 0.5)

    def _lerp(self, previous: float, current: float) -> float:
        return previous + (current - previous) * self.alpha

    def ego_position(self, env: NeonHighwayEnv) -> float:
        return self._lerp(env.previous_ego_position, env.ego_position)

    def ego_lane(self, env: NeonHighwayEnv) -> float:
        return self._lerp(env.previous_lane_position, env.lane_position)

    def car_position(self, car: TrafficCar) -> float:
        return self._lerp(car.previous_position, car.position)

    def _screen_y(self, env: NeonHighwayEnv, world_position: float) -> float:
        return self.EGO_Y - (world_position - self.ego_position(env)) * self.PIXELS_PER_METER

    def _draw_sensors(self, env: NeonHighwayEnv) -> None:
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        ego_x = int(self._lane_center_x(env, self.ego_lane(env)))
        front_origin = (ego_x, self.EGO_Y - 26)
        rear_origin = (ego_x, self.EGO_Y + 26)
        for lane, (ahead_gap, relative_speed, behind_gap, behind_relative) in enumerate(
            env.lane_sensors()
        ):
            lane_x = int(self._lane_center_x(env, float(lane)))
            end_y = max(84, int(self.EGO_Y - ahead_gap * self.PIXELS_PER_METER))
            end = (lane_x, end_y)
            closing = max(-relative_speed, 0.0)
            ttc = ahead_gap / closing if closing > 0.1 else float("inf")
            if ttc < 2.0:
                color = (*self.RED, 150)
            elif ttc < 4.0:
                color = (*self.AMBER, 105)
            else:
                color = (*self.CYAN, 52)
            pygame.draw.aaline(overlay, color, front_origin, end)
            pygame.draw.circle(overlay, color, end, 8, 2)
            label = self.font_tiny.render(f"{ahead_gap:02.0f}m", True, color[:3])
            overlay.blit(label, (end[0] + 10, end[1] - 8))

            rear_end_y = min(
                self.HEIGHT - 30,
                int(self.EGO_Y + behind_gap * self.PIXELS_PER_METER),
            )
            rear_end = (lane_x, rear_end_y)
            rear_closing = max(behind_relative, 0.0)
            rear_ttc = behind_gap / rear_closing if rear_closing > 0.1 else float("inf")
            if rear_ttc < 2.0:
                rear_color = (*self.RED, 150)
            elif rear_ttc < 4.0:
                rear_color = (*self.AMBER, 105)
            else:
                rear_color = (*self.PINK, 52)
            pygame.draw.aaline(overlay, rear_color, rear_origin, rear_end)
            pygame.draw.circle(overlay, rear_color, rear_end, 8, 2)
            rear_label = self.font_tiny.render(
                f"{behind_gap:02.0f}m",
                True,
                rear_color[:3],
            )
            overlay.blit(rear_label, (rear_end[0] + 10, rear_end[1] - 8))
        self.screen.blit(overlay, (0, 0))

    def _draw_traffic(self, env: NeonHighwayEnv) -> None:
        visible = []
        for car in env.traffic:
            y = self._screen_y(env, self.car_position(car))
            if -100 < y < self.HEIGHT + 100:
                visible.append((y, car))
        for y, car in sorted(visible, key=lambda item: item[0]):
            x = self._lane_center_x(env, float(car.lane))
            self._draw_car(
                x,
                y,
                self.TRAFFIC_COLORS[car.color_index],
                car.style,
                agent=False,
                braking=car.braking,
                turn_direction=0,
            )

    def _draw_agent(self, env: NeonHighwayEnv) -> None:
        x = self._lane_center_x(env, self.ego_lane(env))
        threat = env.current_threat()["level"]
        pulse = int(22 + 12 * math.sin(env.step_count * 0.18))
        glow_color = self.RED if threat > 0.65 else self.CYAN
        glow = pygame.Surface((124, 145), pygame.SRCALPHA)
        pygame.draw.ellipse(glow, (*glow_color, pulse), (8, 10, 108, 122))
        self.screen.blit(glow, (x - 62, self.EGO_Y - 72))

        trail = pygame.Surface((70, 120), pygame.SRCALPHA)
        pygame.draw.polygon(trail, (*self.CYAN, 24), [(25, 0), (45, 0), (62, 115), (8, 115)])
        self.screen.blit(trail, (x - 35, self.EGO_Y + 35))

        turn_direction = int(np.sign(env.target_lane - self.ego_lane(env)))
        self._draw_car(
            x,
            self.EGO_Y,
            (39, 217, 235),
            2,
            agent=True,
            braking=env.brake > 0.05,
            turn_direction=turn_direction,
        )

    def _draw_car(
        self,
        center_x: float,
        center_y: float,
        color: tuple[int, int, int],
        style: int,
        *,
        agent: bool,
        braking: bool,
        turn_direction: int,
    ) -> None:
        width = 45 + style * 2
        height = 78 - style * 3
        x = int(center_x - width / 2)
        y = int(center_y - height / 2)
        shadow = pygame.Rect(x + 6, y + 8, width, height)
        pygame.draw.rect(self.screen, (3, 6, 10), shadow, border_radius=13)

        for wheel_y in (y + 13, y + height - 26):
            pygame.draw.rect(self.screen, (6, 8, 12), (x - 4, wheel_y, 7, 18), border_radius=3)
            pygame.draw.rect(
                self.screen,
                (6, 8, 12),
                (x + width - 3, wheel_y, 7, 18),
                border_radius=3,
            )

        body = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, color, body, border_radius=13)
        highlight = tuple(min(channel + 45, 255) for channel in color)
        pygame.draw.line(self.screen, highlight, (x + 10, y + 5), (x + width - 10, y + 5), 3)
        pygame.draw.rect(
            self.screen,
            (13, 27, 44),
            (x + 7, y + 16, width - 14, 23),
            border_radius=7,
        )
        pygame.draw.line(self.screen, (82, 125, 153), (x + 10, y + 19), (x + width - 11, y + 19), 2)
        pygame.draw.rect(
            self.screen,
            (18, 32, 46),
            (x + 9, y + 44, width - 18, 18),
            border_radius=5,
        )

        pygame.draw.rect(self.screen, (220, 249, 255), (x + 5, y + 4, 8, 5), border_radius=2)
        pygame.draw.rect(
            self.screen,
            (220, 249, 255),
            (x + width - 13, y + 4, 8, 5),
            border_radius=2,
        )
        tail_color = (255, 30, 58) if braking else (203, 36, 69)
        tail_size = 7 if braking else 5
        pygame.draw.rect(
            self.screen,
            tail_color,
            (x + 5, y + height - 10, 10, tail_size),
            border_radius=2,
        )
        pygame.draw.rect(
            self.screen,
            tail_color,
            (x + width - 15, y + height - 10, 10, tail_size),
            border_radius=2,
        )

        if turn_direction != 0 and (pygame.time.get_ticks() // 180) % 2 == 0:
            signal_x = x + 5 if turn_direction < 0 else x + width - 10
            pygame.draw.circle(self.screen, self.AMBER, (signal_x, y + height - 8), 4)

        if agent:
            pygame.draw.rect(self.screen, (174, 252, 255), body, 2, border_radius=13)
            badge = self.font_small_bold.render("RL", True, (225, 255, 255))
            self.screen.blit(badge, badge.get_rect(center=(int(center_x), y + 53)))

    def _draw_header(self, env: NeonHighwayEnv) -> None:
        self._glass_panel(pygame.Rect(18, 16, self.WIDTH - 36, 66), alpha=226, radius=18)
        self._text("NEON HIGHWAY", self.font_medium, self.CYAN, 40, 29)
        version = NeonHighwayEnv.VERSION.rsplit("-", 1)[-1].upper()
        subtitle = f"REINFORCEMENT LEARNING LAB / {version}"
        if env.endless:
            subtitle += " / ENDLESS"
        self._text(subtitle, self.font_tiny, self.MUTED, 40, 55)

        total = int(env.hud_data.get("training_total", 0))
        step = int(env.hud_data.get("training_step", 0))
        progress = min(step / total, 1.0) if total > 0 else env.difficulty
        bar_x, bar_y, bar_width = 416, 48, 610
        if total > 0:
            label, value = "TRAINING PROGRESS", f"{step:,} / {total:,}"
        elif env.endless:
            # No finish line to count down to, so the bar tracks this run
            # against the longest one of the session.
            best = float(env.hud_data.get("longest_survival", 0.0))
            label = (
                f"SURVIVED {format_duration(env.elapsed_seconds)}"
                f"   /   {env.ego_position / 1000.0:.2f} km"
            )
            value = f"LONGEST {format_duration(max(best, env.elapsed_seconds))}"
            progress = env.elapsed_seconds / max(best, env.elapsed_seconds, 1.0)
        else:
            label, value = "EPISODE PROGRESS", f"{progress * 100:4.1f}%"
        self._text(label, self.font_tiny, self.MUTED, bar_x, 27)
        self._text(value, self.font_tiny, self.TEXT, bar_x + bar_width - 92, 27)
        pygame.draw.rect(self.screen, (30, 41, 57), (bar_x, bar_y, bar_width, 10), border_radius=5)
        pygame.draw.rect(
            self.screen,
            self.CYAN,
            (bar_x, bar_y, int(bar_width * progress), 10),
            border_radius=5,
        )

        mode = str(env.hud_data.get("mode", "RUNNING"))
        mode_color = self.PINK if "EXPLOR" in mode or "RANDOM" in mode else self.GREEN
        self._pill(mode, 1170, 34, mode_color)

    def _draw_left_dashboard(self, env: NeonHighwayEnv) -> None:
        x, y, width, height = 18, 96, 318, 640
        self._glass_panel(pygame.Rect(x, y, width, height), alpha=218, radius=18)
        self._section_title("SESSION", x + 22, y + 18, self.CYAN)

        self._metric_pair("EPISODE", f"{env.episode_index:03d}", x + 22, y + 59)
        self._metric_pair("LIVE RETURN", f"{env.episode_return:+7.2f}", x + 22, y + 91)
        mean_return = float(env.hud_data.get("mean_return", 0.0))
        best_return = float(env.hud_data.get("best_return", 0.0))
        self._metric_pair("MEAN / 20", f"{mean_return:+7.2f}", x + 22, y + 123)
        self._metric_pair("BEST", f"{best_return:+7.2f}", x + 22, y + 155)

        self._divider(x + 20, y + 193, width - 40)
        self._section_title("LEARNING TREND", x + 22, y + 210, self.PINK)
        graph_rect = pygame.Rect(x + 22, y + 244, width - 44, 112)
        self._draw_sparkline(graph_rect, list(env.hud_data.get("recent_returns", [])))

        self._divider(x + 20, y + 374, width - 40)
        self._section_title("OUTCOMES", x + 22, y + 390, self.GREEN)
        collisions = int(env.hud_data.get("collisions", 0))
        completions = int(env.hud_data.get("completions", 0))
        self._stat_box("COMPLETED", str(completions), x + 22, y + 426, 126, self.GREEN)
        self._stat_box("CRASHED", str(collisions), x + 164, y + 426, 126, self.RED)

        collision_types = env.hud_data.get("collision_types", {})
        self._mini_metric(
            "FRONT IMPACT", int(collision_types.get("FRONT IMPACT", 0)), x + 22, y + 496
        )
        self._mini_metric(
            "SIDE IMPACT", int(collision_types.get("SIDE IMPACT", 0)), x + 22, y + 522
        )
        self._mini_metric(
            "REAR IMPACT", int(collision_types.get("REAR IMPACT", 0)), x + 22, y + 548
        )

        self._divider(x + 20, y + 578, width - 40)
        self._mini_metric("NEAR MISSES", env.near_misses, x + 22, y + 594)
        self._mini_metric("SAFE EVASIONS", env.safe_lane_changes, x + 154, y + 594)

    def _draw_right_dashboard(self, env: NeonHighwayEnv) -> None:
        x, y, width, height = self.WIDTH - 336, 96, 318, 640
        self._glass_panel(pygame.Rect(x, y, width, height), alpha=218, radius=18)
        self._section_title("DRIVER STATE", x + 22, y + 18, self.TEXT)
        self._text(f"{env.ego_speed * 3.6:03.0f}", self.font_speed, self.TEXT, x + 20, y + 48)
        self._text("km/h", self.font_small_bold, self.MUTED, x + 154, y + 90)
        self._pill(f"LANE {env.target_lane + 1}/{env.LANES}", x + 205, y + 58, self.CYAN)
        self._text("TARGET SPEED", self.font_tiny, self.MUTED, x + 205, y + 94)
        self._text(
            f"{env.target_speed * 3.6:03.0f} km/h",
            self.font_small_bold,
            self.CYAN,
            x + 205,
            y + 111,
        )
        self._text(
            f"ACCEL {env.longitudinal_acceleration:+.2f} m/s2",
            self.font_tiny,
            self.MUTED,
            x + 22,
            y + 114,
        )
        self._metric_pair("ACTION", env._info()["action"], x + 22, y + 139)
        self._pedal_bar("THROTTLE", env.throttle, x + 22, y + 174, 126, self.GREEN)
        self._pedal_bar("BRAKE", env.brake, x + 164, y + 174, 126, self.RED)

        self._divider(x + 20, y + 198, width - 40)
        self._section_title("360 SAFETY RADAR", x + 22, y + 215, self.AMBER)
        threat = env.current_threat()
        rear_threat = env.rear_threat()
        threat_color = (
            self.RED
            if threat["level"] > 0.65
            else self.AMBER
            if threat["level"] > 0.3
            else self.GREEN
        )
        self._gauge(x + 22, y + 250, width - 44, threat["level"], threat_color)
        front_ttc = f"{threat['ttc']:.1f}s" if np.isfinite(threat["ttc"]) else "CLEAR"
        rear_ttc = (
            f"{rear_threat['ttc']:.1f}s" if np.isfinite(rear_threat["ttc"]) else "CLEAR"
        )
        self._metric_pair("FRONT TTC", front_ttc, x + 22, y + 275)
        self._metric_pair("REAR TTC", rear_ttc, x + 22, y + 301)
        challenge_count = f"{env.challenges_resolved}/{len(env.challenge_steps)}"
        challenge_label = challenge_count if env.difficulty_mode == "hard" else "STANDARD"
        if env.endless:
            challenge_label = f"{env.challenges_resolved} CLEARED"
        self._metric_pair("CHALLENGES", challenge_label, x + 22, y + 327)

        self._divider(x + 20, y + 359, width - 40)
        self._section_title("REWARD SIGNAL", x + 22, y + 375, self.PINK)
        components = env.last_reward_components
        component_colors = [
            self.CYAN,
            self.AMBER,
            (120, 200, 255),
            (179, 125, 246),
            self.RED,
            self.GREEN,
        ]
        for index, name in enumerate(
            ["progress", "safety", "shaping", "comfort", "rules", "terminal"]
        ):
            self._reward_bar(
                name.upper(),
                float(components.get(name, 0.0)),
                x + 22,
                y + 402 + index * 19,
                width - 44,
                component_colors[index],
            )

        self._divider(x + 20, y + 512, width - 40)
        self._section_title("ACTION VALUES / Q", x + 22, y + 524, self.CYAN)
        q_values = [float(value) for value in env.hud_data.get("q_values", [0.0] * ACTION_COUNT)]
        self._draw_q_values(env, q_values, x + 22, y + 568, width - 44)

    def _draw_q_values(
        self,
        env: NeonHighwayEnv,
        q_values: list[float],
        x: int,
        y: int,
        width: int,
    ) -> None:
        """Draw the nine action values as a steer x pedal grid.

        A flat list of nine rows no longer fits the panel, and the grid shows
        the structure the action space actually has.
        """
        if len(q_values) != ACTION_COUNT:
            q_values = [0.0] * ACTION_COUNT
        minimum, maximum = min(q_values), max(q_values)
        spread = maximum - minimum
        best = q_values.index(maximum)
        selected_steer, _ = decode_action(env.last_action)

        cell_width = (width - 54) // 3
        for pedal, label in enumerate(["BRAKE", "COAST", "GAS"]):
            self._text(
                label, self.font_tiny, self.MUTED, x + 54 + pedal * cell_width, y - 14
            )
        for steer, label in enumerate(["LEFT", "KEEP", "RIGHT"]):
            row_y = y + steer * 22
            active_row = steer == selected_steer
            self._text(
                label,
                self.font_tiny,
                self.TEXT if active_row else self.MUTED,
                x,
                row_y + 1,
            )
            for pedal in range(3):
                action = encode_action(steer, pedal)
                value = q_values[action]
                normalized = (value - minimum) / spread if spread > 1e-6 else 0.5
                cell_x = x + 54 + pedal * cell_width
                bar_width = cell_width - 8
                is_best = action == best
                color = self.CYAN if is_best else (76, 91, 112)
                pygame.draw.rect(
                    self.screen, (25, 35, 49), (cell_x, row_y, bar_width, 7), border_radius=3
                )
                pygame.draw.rect(
                    self.screen,
                    color,
                    (cell_x, row_y, max(2, int(bar_width * normalized)), 7),
                    border_radius=3,
                )
                self._text(
                    f"{value:+.1f}",
                    self.font_tiny,
                    self.TEXT if is_best else self.MUTED,
                    cell_x,
                    row_y + 8,
                )

    def _draw_action_strip(self, env: NeonHighwayEnv) -> None:
        """Steering and pedal are chosen together, so each gets its own row."""
        x, y, width = self.ROAD_LEFT + 25, self.HEIGHT - 66, self.ROAD_WIDTH - 50
        self._glass_panel(pygame.Rect(x, y, width, 43), alpha=220, radius=15)
        steer, pedal = decode_action(env.last_action)
        groups = (
            (["LANE LEFT", "KEEP LANE", "LANE RIGHT"], steer, 0),
            (["BRAKE", "COAST", "GAS"], pedal, 1),
        )
        half = (width - 24) // 2
        for labels, selected, column in groups:
            button_width = half // 3
            for index, label in enumerate(labels):
                button_x = x + 8 + column * half + index * button_width
                active = index == selected
                color = self.CYAN if active else (57, 68, 85)
                pygame.draw.rect(
                    self.screen,
                    color,
                    (button_x, y + 8, button_width - 6, 27),
                    border_radius=13,
                )
                text_color = (5, 21, 28) if active else (193, 204, 220)
                surface = self.font_tiny.render(label, True, text_color)
                self.screen.blit(
                    surface,
                    surface.get_rect(center=(button_x + (button_width - 6) // 2, y + 21)),
                )

    def _draw_footer(self, env: NeonHighwayEnv) -> None:
        self._text(
            "SPACE  pause       H  toggle sensors       ESC  save and exit",
            self.font_tiny,
            self.MUTED,
            22,
            self.HEIGHT - 20,
        )
        observations = env.observation_space.shape[0]
        self._text(
            f"OBSERVATION: {observations} VALUES / ACTIONS: STEER x PEDAL",
            self.font_tiny,
            self.MUTED,
            1030,
            self.HEIGHT - 20,
        )

    def _draw_impact_particles(self, env: NeonHighwayEnv, phase: float) -> None:
        rng = np.random.default_rng(env.episode_index * 7919)
        center_x = self._lane_center_x(env, self.ego_lane(env))
        center_y = self.EGO_Y - 18
        for _ in range(34):
            angle = float(rng.uniform(0, math.tau))
            distance = float(rng.uniform(35, 155)) * phase
            x = int(center_x + math.cos(angle) * distance)
            y = int(center_y + math.sin(angle) * distance + 45 * phase * phase)
            color = self.AMBER if rng.random() > 0.35 else self.RED
            size = int(rng.integers(2, 6))
            pygame.draw.circle(self.screen, color, (x, y), size)
            tail_x = int(x - math.cos(angle) * 14)
            tail_y = int(y - math.sin(angle) * 14)
            pygame.draw.line(self.screen, color, (x, y), (tail_x, tail_y), 2)

        ring_radius = int(35 + 100 * phase)
        ring = pygame.Surface((ring_radius * 2 + 6, ring_radius * 2 + 6), pygame.SRCALPHA)
        pygame.draw.circle(
            ring,
            (*self.RED, int(170 * (1.0 - phase))),
            (ring_radius + 3, ring_radius + 3),
            ring_radius,
            4,
        )
        self.screen.blit(ring, (center_x - ring_radius - 3, center_y - ring_radius - 3))

    def _draw_crash_overlay(self, env: NeonHighwayEnv, phase: float) -> None:
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((185, 12, 42, int(40 + 48 * phase)))
        self.screen.blit(overlay, (0, 0))

        collision = env.last_collision
        if collision is None:
            return
        panel = pygame.Rect(470, 232, 500, 310)
        self._glass_panel(panel, alpha=int(150 + 85 * phase), radius=22)
        severity_color = {
            "LOW": self.AMBER,
            "MEDIUM": (255, 119, 65),
            "HIGH": self.RED,
        }[collision.severity]
        self._center_text("IMPACT DETECTED", self.font_large, self.RED, 268)
        self._pill(f"{collision.severity} SEVERITY", 621, 311, severity_color)
        self._center_text(collision.kind, self.font_medium, self.TEXT, 359)
        impact_summary = (
            f"Impact speed  {collision.impact_speed * 3.6:.1f} km/h   |   Lane {collision.lane + 1}"
        )
        self._center_text(
            impact_summary,
            self.font_small_bold,
            self.TEXT,
            405,
        )
        self._center_text(
            f"Episode return {env.episode_return:+.2f}   |   Near misses {env.near_misses}",
            self.font_small,
            self.MUTED,
            439,
        )
        self._center_text(
            "Transition stored in replay memory. Restarting with a new traffic seed.",
            self.font_small,
            self.CYAN,
            496,
        )

    def _draw_completion_overlay(self, env: NeonHighwayEnv) -> None:
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((15, 130, 104, 42))
        self.screen.blit(overlay, (0, 0))
        self._glass_panel(pygame.Rect(500, 290, 440, 170), alpha=232, radius=22)
        self._center_text("ROUTE COMPLETE", self.font_large, self.GREEN, 330)
        self._center_text(
            f"Cleared {env.challenges_resolved} waves  |  Return {env.episode_return:+.2f}",
            self.font_small_bold,
            self.TEXT,
            388,
        )
        self._center_text(
            "Completion bonus added to replay memory.", self.font_small, self.MUTED, 423
        )

    def _draw_sparkline(self, rect: pygame.Rect, values: list[float]) -> None:
        pygame.draw.rect(self.screen, (12, 23, 38), rect, border_radius=10)
        pygame.draw.line(
            self.screen,
            (37, 50, 67),
            (rect.left + 8, rect.centery),
            (rect.right - 8, rect.centery),
            1,
        )
        if len(values) < 2:
            self._center_text_in_rect(
                "Waiting for completed episodes", self.font_tiny, self.MUTED, rect
            )
            return

        minimum, maximum = min(values), max(values)
        spread = max(maximum - minimum, 1.0)
        points = []
        for index, value in enumerate(values):
            px = rect.left + 9 + index * (rect.width - 18) / (len(values) - 1)
            py = rect.bottom - 9 - (value - minimum) / spread * (rect.height - 18)
            points.append((int(px), int(py)))
        pygame.draw.aalines(self.screen, self.CYAN, False, points)
        for point in points[-3:]:
            pygame.draw.circle(self.screen, self.PINK, point, 3)
        self._text(f"{maximum:+.1f}", self.font_tiny, self.MUTED, rect.left + 8, rect.top + 4)
        self._text(f"{minimum:+.1f}", self.font_tiny, self.MUTED, rect.left + 8, rect.bottom - 18)

    def _reward_bar(
        self,
        label: str,
        value: float,
        x: int,
        y: int,
        width: int,
        color: tuple[int, int, int],
    ) -> None:
        self._text(label, self.font_tiny, self.MUTED, x, y - 4)
        bar_x = x + 72
        bar_width = width - 120
        pygame.draw.rect(self.screen, (25, 35, 49), (bar_x, y, bar_width, 7), border_radius=3)
        normalized = min(abs(value) / 0.22, 1.0)
        draw_color = color if value >= 0 else self.RED
        pygame.draw.rect(
            self.screen,
            draw_color,
            (bar_x, y, max(1, int(bar_width * normalized)), 7),
            border_radius=3,
        )
        self._text(f"{value:+.3f}", self.font_tiny, self.TEXT, x + width - 43, y - 5)

    def _pedal_bar(
        self,
        label: str,
        value: float,
        x: int,
        y: int,
        width: int,
        color: tuple[int, int, int],
    ) -> None:
        self._text(label, self.font_tiny, self.MUTED, x, y - 13)
        pygame.draw.rect(self.screen, (25, 35, 49), (x, y + 5, width, 8), border_radius=4)
        pygame.draw.rect(
            self.screen,
            color,
            (x, y + 5, max(2, int(width * value)), 8),
            border_radius=4,
        )

    def _gauge(
        self,
        x: int,
        y: int,
        width: int,
        value: float,
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(self.screen, (26, 36, 49), (x, y, width, 12), border_radius=6)
        pygame.draw.rect(
            self.screen,
            color,
            (x, y, max(3, int(width * value)), 12),
            border_radius=6,
        )
        for fraction in (0.33, 0.66):
            line_x = x + int(width * fraction)
            pygame.draw.line(self.screen, (8, 15, 25), (line_x, y), (line_x, y + 12), 2)

    def _stat_box(
        self,
        label: str,
        value: str,
        x: int,
        y: int,
        width: int,
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(self.screen, (14, 27, 42), (x, y, width, 54), border_radius=10)
        pygame.draw.line(self.screen, color, (x + 1, y + 8), (x + 1, y + 46), 3)
        self._text(label, self.font_tiny, self.MUTED, x + 12, y + 7)
        self._text(value, self.font_medium, color, x + 12, y + 25)

    def _mini_metric(self, label: str, value: object, x: int, y: int) -> None:
        self._text(label, self.font_tiny, self.MUTED, x, y)
        value_surface = self.font_small_bold.render(str(value), True, self.TEXT)
        self.screen.blit(value_surface, (x + 105, y - 2))

    def _metric_pair(self, label: str, value: str, x: int, y: int) -> None:
        self._text(label, self.font_small, self.MUTED, x, y)
        value_surface = self.font_small_bold.render(value, True, self.TEXT)
        self.screen.blit(value_surface, (x + 146, y - 1))

    def _section_title(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(self.screen, color, (x, y + 3, 4, 18), border_radius=2)
        self._text(text, self.font_small_bold, self.TEXT, x + 13, y)

    def _divider(self, x: int, y: int, width: int) -> None:
        pygame.draw.line(self.screen, (43, 57, 76), (x, y), (x + width, y), 1)

    def _glass_panel(
        self,
        rect: pygame.Rect,
        *,
        alpha: int = 215,
        radius: int = 16,
    ) -> None:
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (*self.PANEL, alpha), panel.get_rect(), border_radius=radius)
        pygame.draw.rect(panel, (76, 105, 139, 85), panel.get_rect(), 1, border_radius=radius)
        self.screen.blit(panel, rect.topleft)

    def _pill(self, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        surface = self.font_tiny.render(label, True, (5, 20, 27))
        rect = pygame.Rect(x, y, surface.get_width() + 22, 25)
        pygame.draw.rect(self.screen, color, rect, border_radius=12)
        self.screen.blit(surface, (x + 11, y + 4))

    def _text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        x: int,
        y: int,
    ) -> None:
        self.screen.blit(font.render(text, True, color), (x, y))

    def _center_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        y: int,
    ) -> None:
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=(self.WIDTH // 2, y)))

    def _center_text_in_rect(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        rect: pygame.Rect,
    ) -> None:
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=rect.center))

    def _frame_array(self) -> np.ndarray:
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self) -> None:
        if self.human:
            pygame.display.quit()
        pygame.quit()
