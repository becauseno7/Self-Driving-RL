"""Calm, information-dense Pygame visuals for the autonomous-driving lab."""

from __future__ import annotations

from collections import deque

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
    """Render a logical 16:9 driving canvas and scale it to the viewer window."""

    WIDTH = 1440
    HEIGHT = 810
    ROAD_WIDTH = 792  # Exactly 55% of the logical viewport.
    ROAD_LEFT = (WIDTH - ROAD_WIDTH) // 2
    EGO_Y = 620
    PIXELS_PER_METER = 22.5
    MIN_FUNCTIONAL_FONT_PX = 16
    DEFAULT_FUNCTIONAL_LABELS = (
        "Speed",
        "Target",
        "Lane",
        "Intent",
        "Inputs",
        "Throttle",
        "Brake",
        "Progress",
        "Safety",
        "Front TTC",
        "Rear TTC",
        "Net passes",
    )

    CANVAS = (10, 15, 20)
    PANEL = (17, 24, 32)
    PANEL_RAISED = (21, 30, 39)
    ROAD = (37, 43, 49)
    DIVIDER = (42, 53, 64)
    TEXT = (237, 242, 245)
    MUTED = (155, 168, 178)
    ACCENT = (112, 169, 190)
    SUCCESS = (112, 177, 139)
    WARNING = (211, 161, 86)
    DANGER = (212, 103, 112)

    # Compatibility aliases for small external renderer extensions.
    CYAN = ACCENT
    GREEN = SUCCESS
    AMBER = WARNING
    RED = DANGER

    TRAFFIC_COLORS = [
        (111, 124, 132),
        (125, 116, 100),
        (91, 111, 113),
        (101, 113, 130),
        (125, 124, 119),
        (87, 99, 108),
        (104, 119, 102),
        (126, 105, 99),
    ]

    LEFT_PANEL = pygame.Rect(16, 16, 292, 778)
    RIGHT_PANEL = pygame.Rect(1132, 16, 292, 778)

    def __init__(self, *, human: bool, fps: int, speed: float = 1.0) -> None:
        pygame.init()
        pygame.font.init()
        self.human = human
        self.fps = fps
        self.speed = max(speed, 1e-6)
        self.clock = pygame.time.Clock()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.display_surface: pygame.Surface | None = None
        if human:
            self.display_surface = pygame.display.set_mode(
                (self.WIDTH, self.HEIGHT), pygame.RESIZABLE
            )
            pygame.display.set_caption("Autonomy Lab — Night Drive")

        # Fourteen pixels is the floor for functional text in the logical view.
        self.font_tiny = pygame.font.SysFont("Segoe UI", self.MIN_FUNCTIONAL_FONT_PX)
        self.font_small = pygame.font.SysFont("Segoe UI", 17)
        self.font_small_bold = pygame.font.SysFont("Segoe UI", 17, bold=True)
        self.font_medium = pygame.font.SysFont("Segoe UI", 22, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI", 30, bold=True)
        self.font_speed = pygame.font.SysFont("Segoe UI", 54, bold=True)
        self.background = self._make_background()
        self.paused = False
        self.show_sensors = False
        self.analysis_open = False
        self.teacher_events: deque[str] = deque()
        self.last_crash_episode = -1
        self.alpha = 1.0

    def _make_background(self) -> pygame.Surface:
        """Build a quiet cockpit surround without decorative city noise."""
        surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        surface.fill(self.CANVAS)
        for y in range(self.HEIGHT):
            shade = int(5 * y / self.HEIGHT)
            pygame.draw.line(
                surface,
                (self.CANVAS[0] + shade, self.CANVAS[1] + shade, self.CANVAS[2] + shade),
                (0, y),
                (self.WIDTH, y),
            )
        pygame.draw.rect(surface, (8, 12, 16), (0, 0, self.WIDTH, 8))
        return surface

    def draw(self, env: NeonHighwayEnv) -> tuple[np.ndarray | None, bool]:
        if self.human and not self._handle_events():
            return None, False
        self._wait_while_paused(env)
        if env.quit_requested:
            return None, False

        new_crash = env.crashed and self.last_crash_episode != env.episode_index
        if self.human and new_crash:
            for frame_index in range(12):
                if not self._handle_events():
                    return None, False
                self._render_frame(env, crash_phase=(frame_index + 1) / 12)
                self._present()
                self.clock.tick(60)
            self.last_crash_episode = env.episode_index
        elif self.human:
            frames = self._frames_per_step(env)
            for frame_index in range(frames):
                if frame_index and not self._handle_events():
                    return None, False
                self._render_frame(
                    env,
                    crash_phase=1.0 if env.crashed else 0.0,
                    alpha=(frame_index + 1) / frames,
                )
                self._present()
                if self.fps > 0:
                    self.clock.tick(self.fps)
        else:
            self._render_frame(env, crash_phase=1.0 if env.crashed else 0.0)

        frame = None if self.human else self._frame_array()
        return frame, True

    def _frames_per_step(self, env: NeonHighwayEnv) -> int:
        """Frames per simulation step needed for the requested playback speed."""
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

        self._draw_panel_shells()
        if self.analysis_open:
            self._draw_analysis_view(env)
        else:
            self._draw_drive_view(env)

        if env.hud_data.get("dagger_collecting", False):
            self._draw_teacher_status(env)
        if env.crashed:
            self._draw_crash_overlay(env, crash_phase)
        elif env.completed:
            self._draw_completion_overlay(env)

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE and self.human:
                width = max(640, int(event.w))
                height = max(360, int(event.h))
                self.display_surface = pygame.display.set_mode(
                    (width, height), pygame.RESIZABLE
                )
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_h:
                    self.show_sensors = not self.show_sensors
                if event.key == pygame.K_TAB:
                    self.analysis_open = not self.analysis_open
                teacher_event = {
                    pygame.K_LEFT: "left",
                    pygame.K_a: "left",
                    pygame.K_k: "keep",
                    pygame.K_RIGHT: "right",
                    pygame.K_d: "right",
                    pygame.K_UP: "faster",
                    pygame.K_w: "faster",
                    pygame.K_DOWN: "slower",
                    pygame.K_s: "slower",
                    pygame.K_RETURN: "approve",
                    pygame.K_BACKSPACE: "undo",
                    pygame.K_u: "undo",
                }.get(event.key)
                if teacher_event is not None:
                    self.teacher_events.append(teacher_event)
        return True

    @classmethod
    def _letterbox_rect(cls, viewport: tuple[int, int]) -> pygame.Rect:
        """Return the largest centered 16:9 logical canvas inside a viewport."""
        viewport_width, viewport_height = viewport
        scale = min(viewport_width / cls.WIDTH, viewport_height / cls.HEIGHT)
        width = max(1, round(cls.WIDTH * scale))
        height = max(1, round(cls.HEIGHT * scale))
        return pygame.Rect(
            (viewport_width - width) // 2,
            (viewport_height - height) // 2,
            width,
            height,
        )

    def _present(self) -> None:
        if self.display_surface is None:
            return
        viewport = self.display_surface.get_size()
        target = self._letterbox_rect(viewport)
        self.display_surface.fill((4, 7, 9))
        if target.size == (self.WIDTH, self.HEIGHT):
            scaled = self.screen
        else:
            scaled = pygame.transform.smoothscale(self.screen, target.size)
        self.display_surface.blit(scaled, target.topleft)
        pygame.display.flip()

    def pop_teacher_event(self) -> str | None:
        """Return one deliberate DAgger label event, if supplied."""
        return self.teacher_events.popleft() if self.teacher_events else None

    def _wait_while_paused(self, env: NeonHighwayEnv) -> None:
        while self.human and self.paused:
            if not self._handle_events():
                env.quit_requested = True
                self.paused = False
                return
            self._render_frame(env, crash_phase=1.0 if env.crashed else 0.0)
            self._side_status(
                self.RIGHT_PANEL.inflate(-24, -500),
                "Simulation paused",
                "Space to continue",
                self.ACCENT,
            )
            self._present()
            self.clock.tick(30)

    def _draw_road(self, env: NeonHighwayEnv) -> None:
        road_rect = pygame.Rect(self.ROAD_LEFT, 0, self.ROAD_WIDTH, self.HEIGHT)
        pygame.draw.rect(self.screen, (5, 8, 11), road_rect.inflate(14, 0))
        pygame.draw.rect(self.screen, self.ROAD, road_rect)

        lane_width = self.ROAD_WIDTH / env.LANES
        target_left = int(self.ROAD_LEFT + env.target_lane * lane_width)
        target_layer = pygame.Surface((int(lane_width), self.HEIGHT), pygame.SRCALPHA)
        target_layer.fill((*self.ACCENT, 10))
        self.screen.blit(target_layer, (target_left, 0))

        shoulder = 17
        pygame.draw.rect(
            self.screen,
            (30, 37, 43),
            (self.ROAD_LEFT, 0, shoulder, self.HEIGHT),
        )
        pygame.draw.rect(
            self.screen,
            (30, 37, 43),
            (self.ROAD_LEFT + self.ROAD_WIDTH - shoulder, 0, shoulder, self.HEIGHT),
        )
        edge = (104, 116, 124)
        pygame.draw.line(
            self.screen,
            edge,
            (self.ROAD_LEFT + shoulder, 0),
            (self.ROAD_LEFT + shoulder, self.HEIGHT),
            2,
        )
        pygame.draw.line(
            self.screen,
            edge,
            (self.ROAD_LEFT + self.ROAD_WIDTH - shoulder, 0),
            (self.ROAD_LEFT + self.ROAD_WIDTH - shoulder, self.HEIGHT),
            2,
        )

        offset = int((self.ego_position(env) * self.PIXELS_PER_METER) % 72)
        for lane in range(1, env.LANES):
            x = round(self.ROAD_LEFT + lane * lane_width)
            for y in range(-72 + offset, self.HEIGHT, 72):
                pygame.draw.line(self.screen, (108, 119, 126), (x, y), (x, y + 34), 3)

        for y in range(-90 + offset, self.HEIGHT, 90):
            pygame.draw.line(
                self.screen,
                (45, 53, 59),
                (self.ROAD_LEFT + 5, y),
                (self.ROAD_LEFT + 12, y + 24),
                2,
            )
            pygame.draw.line(
                self.screen,
                (45, 53, 59),
                (self.ROAD_LEFT + self.ROAD_WIDTH - 6, y),
                (self.ROAD_LEFT + self.ROAD_WIDTH - 13, y + 24),
                2,
            )

        self._draw_merge_path(env)

    def _draw_merge_path(self, env: NeonHighwayEnv) -> None:
        """Draw a restrained route guide from the ego car to the target lane."""
        layer = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        start_x = self._lane_center_x(env, self.ego_lane(env))
        end_x = self._lane_center_x(env, float(env.target_lane))
        start_y, end_y = self.EGO_Y + 56, 118
        points: list[tuple[int, int]] = []
        for index in range(43):
            t = index / 42
            eased = t * t * (3.0 - 2.0 * t)
            x = start_x + (end_x - start_x) * eased
            y = start_y + (end_y - start_y) * t
            points.append((round(x), round(y)))
        for index in range(0, len(points) - 2, 4):
            pygame.draw.aaline(layer, (*self.ACCENT, 112), points[index], points[index + 2])
        pygame.draw.line(
            layer,
            (*self.ACCENT, 135),
            (round(end_x - 26), end_y),
            (round(end_x + 26), end_y),
            3,
        )
        self.screen.blit(layer, (0, 0))

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
        """Show low-contrast range geometry without placing labels over traffic."""
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        ego_x = round(self._lane_center_x(env, self.ego_lane(env)))
        half_length = self.car_footprint(env)[1] / 2.0
        front_origin = (ego_x, round(self.EGO_Y - half_length))
        rear_origin = (ego_x, round(self.EGO_Y + half_length))
        for lane, (ahead_gap, relative_speed, behind_gap, behind_relative) in enumerate(
            env.lane_sensors()
        ):
            lane_x = round(self._lane_center_x(env, float(lane)))
            front_y = max(20, round(self.EGO_Y - ahead_gap * self.PIXELS_PER_METER))
            closing = max(-relative_speed, 0.0)
            front_ttc = ahead_gap / closing if closing > 0.1 else float("inf")
            front_color = self._risk_color(front_ttc)
            pygame.draw.aaline(overlay, (*front_color, 66), front_origin, (lane_x, front_y))
            pygame.draw.circle(overlay, (*front_color, 115), (lane_x, front_y), 6, 2)

            rear_y = min(
                self.HEIGHT - 20,
                round(self.EGO_Y + behind_gap * self.PIXELS_PER_METER),
            )
            rear_closing = max(behind_relative, 0.0)
            rear_ttc = behind_gap / rear_closing if rear_closing > 0.1 else float("inf")
            rear_color = self._risk_color(rear_ttc)
            pygame.draw.aaline(overlay, (*rear_color, 54), rear_origin, (lane_x, rear_y))
            pygame.draw.circle(overlay, (*rear_color, 100), (lane_x, rear_y), 6, 2)
        self.screen.blit(overlay, (0, 0))

    def _draw_traffic(self, env: NeonHighwayEnv) -> None:
        visible: list[tuple[float, TrafficCar]] = []
        for car in env.traffic:
            y = self._screen_y(env, self.car_position(car))
            if -140 < y < self.HEIGHT + 140:
                visible.append((y, car))
        for y, car in sorted(visible, key=lambda item: item[0]):
            x = self._lane_center_x(env, env.traffic_lateral_position(car))
            self._draw_car(
                env,
                x,
                y,
                self.TRAFFIC_COLORS[car.color_index % len(self.TRAFFIC_COLORS)],
                car.style,
                agent=False,
                braking=car.braking,
                turn_direction=(
                    0
                    if car.target_lane is None
                    else (-1 if car.target_lane < car.lane else 1)
                ),
            )

    def _draw_agent(self, env: NeonHighwayEnv) -> None:
        x = self._lane_center_x(env, self.ego_lane(env))
        self._draw_car(
            env,
            x,
            self.EGO_Y,
            (65, 80, 90),
            2,
            agent=True,
            braking=env.brake > 0.05,
            turn_direction=int(np.sign(env.target_lane - self.ego_lane(env))),
        )

    @classmethod
    def car_footprint(cls, env: NeonHighwayEnv) -> tuple[float, float]:
        """Return the exact rendered footprint implied by the physics model."""
        lane_pixels = cls.ROAD_WIDTH / env.LANES
        width = env.CAR_WIDTH / env.LANE_WIDTH * lane_pixels
        length = env.CAR_LENGTH * cls.PIXELS_PER_METER
        return width, length

    def _draw_car(
        self,
        env: NeonHighwayEnv,
        center_x: float,
        center_y: float,
        color: tuple[int, int, int],
        style: int,
        *,
        agent: bool,
        braking: bool,
        turn_direction: int,
    ) -> None:
        width, height = self.car_footprint(env)
        sprite_width = max(24, round(width))
        sprite_height = max(38, round(height))
        sprite = pygame.Surface((sprite_width, sprite_height), pygame.SRCALPHA)

        def point(x_fraction: float, y_fraction: float) -> tuple[int, int]:
            return round(x_fraction * (sprite_width - 1)), round(
                y_fraction * (sprite_height - 1)
            )

        silhouettes = (
            [
                (0.28, 0.01),
                (0.72, 0.01),
                (0.88, 0.10),
                (0.96, 0.31),
                (0.93, 0.76),
                (0.82, 0.96),
                (0.18, 0.96),
                (0.07, 0.76),
                (0.04, 0.31),
                (0.12, 0.10),
            ],
            [
                (0.22, 0.01),
                (0.78, 0.01),
                (0.91, 0.13),
                (0.96, 0.36),
                (0.94, 0.82),
                (0.82, 0.98),
                (0.18, 0.98),
                (0.06, 0.82),
                (0.04, 0.36),
                (0.09, 0.13),
            ],
            [
                (0.17, 0.01),
                (0.83, 0.01),
                (0.94, 0.12),
                (0.97, 0.88),
                (0.86, 0.99),
                (0.14, 0.99),
                (0.03, 0.88),
                (0.06, 0.12),
            ],
        )
        outline = [point(*coordinates) for coordinates in silhouettes[style % 3]]
        shadow = [(x + 3, min(y + 4, sprite_height - 1)) for x, y in outline]
        pygame.draw.polygon(sprite, (2, 5, 7, 150), shadow)

        wheel_height = max(8, round(sprite_height * 0.22))
        wheel_width = max(3, round(sprite_width * 0.07))
        for wheel_y in (round(sprite_height * 0.18), round(sprite_height * 0.66)):
            pygame.draw.rect(
                sprite,
                (4, 6, 8),
                (1, wheel_y, wheel_width, wheel_height),
                border_radius=2,
            )
            pygame.draw.rect(
                sprite,
                (4, 6, 8),
                (sprite_width - wheel_width - 1, wheel_y, wheel_width, wheel_height),
                border_radius=2,
            )

        pygame.draw.polygon(sprite, color, outline)
        body_edge = tuple(min(channel + 28, 255) for channel in color)
        pygame.draw.aalines(sprite, body_edge, True, outline)

        glass = (15, 25, 32)
        glass_edge = (78, 100, 112)
        windows = (
            [point(0.28, 0.28), point(0.72, 0.28), point(0.79, 0.43), point(0.21, 0.43)],
            [point(0.22, 0.46), point(0.78, 0.46), point(0.75, 0.61), point(0.25, 0.61)],
            [point(0.25, 0.64), point(0.75, 0.64), point(0.69, 0.77), point(0.31, 0.77)],
        )
        for window in windows:
            pygame.draw.polygon(sprite, glass, window)
            pygame.draw.aalines(sprite, glass_edge, True, window)

        headlight = (220, 228, 223)
        pygame.draw.line(sprite, headlight, point(0.17, 0.08), point(0.35, 0.05), 3)
        pygame.draw.line(sprite, headlight, point(0.65, 0.05), point(0.83, 0.08), 3)
        tail_color = self.DANGER if braking else (133, 68, 72)
        tail_width = 4 if braking else 2
        pygame.draw.line(sprite, tail_color, point(0.17, 0.91), point(0.38, 0.94), tail_width)
        pygame.draw.line(sprite, tail_color, point(0.62, 0.94), point(0.83, 0.91), tail_width)

        if turn_direction and (pygame.time.get_ticks() // 220) % 2 == 0:
            signal_x = 0.13 if turn_direction < 0 else 0.87
            pygame.draw.circle(sprite, self.WARNING, point(signal_x, 0.91), 3)

        if agent:
            # The roof stripe and crisp outline identify ego without an aura.
            pygame.draw.aalines(sprite, self.ACCENT, True, outline)
            pygame.draw.line(sprite, self.ACCENT, point(0.49, 0.30), point(0.49, 0.76), 3)
            pygame.draw.line(sprite, self.TEXT, point(0.53, 0.30), point(0.53, 0.76), 1)

        self.screen.blit(
            sprite,
            (round(center_x - sprite_width / 2), round(center_y - sprite_height / 2)),
        )

    def _draw_panel_shells(self) -> None:
        self._panel(self.LEFT_PANEL)
        self._panel(self.RIGHT_PANEL)

    def _draw_drive_view(self, env: NeonHighwayEnv) -> None:
        self._draw_driver_panel(env)
        self._draw_safety_panel(env)

    def _draw_driver_panel(self, env: NeonHighwayEnv) -> None:
        x = self.LEFT_PANEL.x + 18
        self._section_title("Speed", x, 35, self.ACCENT)
        self._text(
            f"{env.ego_speed * 3.6:03.0f} km/h",
            self.font_speed,
            self.TEXT,
            x,
            78,
            max_width=256,
        )

        self._small_card(
            pygame.Rect(x, 168, 121, 74),
            "Target",
            f"{env.target_speed * 3.6:.0f} km/h",
        )
        self._small_card(
            pygame.Rect(x + 135, 168, 121, 74),
            "Lane",
            f"{env.target_lane + 1} of {env.LANES}",
        )

        intent = str(env.hud_data.get("driving_intent", env._info()["action"]))
        desired_speed = float(env.hud_data.get("desired_speed", env.target_speed)) * 3.6
        _, pedal = decode_action(env.last_action)
        pedal_name = ["Brake", "Coast", "Accelerate"][pedal]
        braking_mode = str(env.hud_data.get("braking_mode", pedal_name))
        self._section_title("Intent", x, 277, self.ACCENT)
        self._text(intent, self.font_medium, self.TEXT, x, 319, max_width=256)
        self._text(
            f"{desired_speed:.0f} km/h · {braking_mode}",
            self.font_tiny,
            self.MUTED,
            x,
            354,
            max_width=256,
        )

        self._divider(x, 405, 256)
        self._section_title("Inputs", x, 426, self.MUTED)
        self._value_bar("Throttle", env.throttle, x, 474, 256, self.SUCCESS)
        self._value_bar("Brake", env.brake, x, 526, 256, self.DANGER)

        self._text("Space  Pause · H  Sensors", self.font_tiny, self.MUTED, x, 730)
        self._text("Tab  Analysis · Esc  Exit", self.font_tiny, self.ACCENT, x, 762)

    def _draw_safety_panel(self, env: NeonHighwayEnv) -> None:
        x = self.RIGHT_PANEL.x + 18
        mode = str(env.hud_data.get("mode", "Running"))
        mode_color = (
            self.WARNING
            if any(word in mode.upper() for word in ("EXPLOR", "RANDOM"))
            else self.SUCCESS
        )
        self._pill(mode.title(), x, 32, mode_color)

        _, value, progress = self._progress_state(env)
        self._section_title("Progress", x, 85, self.ACCENT)
        self._text(value, self.font_small_bold, self.TEXT, x, 132, max_width=256)
        self._progress_bar(x, 174, 256, progress, self.ACCENT)

        self._divider(x, 216, 256)
        self._section_title("Safety", x, 239, self.WARNING)
        front = env.current_threat()
        rear = env.rear_threat()
        front_value = self._format_ttc(float(front["ttc"]))
        rear_value = self._format_ttc(float(rear["ttc"]))
        self._ttc_card(
            pygame.Rect(x, 282, 121, 84),
            "Front TTC",
            front_value,
            self._risk_color(float(front["ttc"])),
        )
        self._ttc_card(
            pygame.Rect(x + 135, 282, 121, 84),
            "Rear TTC",
            rear_value,
            self._risk_color(float(rear["ttc"])),
        )

        net_passes = env.overtakes - env.passed_by_traffic
        self._divider(x, 411, 256)
        self._small_card(
            pygame.Rect(x, 439, 256, 82),
            "Net passes",
            f"{net_passes:+d}",
            self.SUCCESS if net_passes >= 0 else self.WARNING,
        )
        self._text("Tab  Open full analysis", self.font_tiny, self.ACCENT, x, 762)

    def _draw_analysis_view(self, env: NeonHighwayEnv) -> None:
        self._draw_analysis_left(env)
        self._draw_analysis_right(env)

    def _draw_analysis_left(self, env: NeonHighwayEnv) -> None:
        x = self.LEFT_PANEL.x + 18
        self._text("Analysis", self.font_medium, self.TEXT, x, 31)
        self._text("Tab returns to drive", self.font_tiny, self.ACCENT, x, 59)
        self._divider(x, 88, 256)

        self._section_title("Reward signal", x, 104, self.ACCENT)
        components = env.last_reward_components
        colors = {
            "progress": self.SUCCESS,
            "traffic": self.ACCENT,
            "safety": self.WARNING,
            "shaping": self.ACCENT,
            "comfort": self.SUCCESS,
            "rules": self.WARNING,
            "terminal": self.DANGER,
        }
        for index, name in enumerate(
            ("progress", "traffic", "safety", "shaping", "comfort", "rules", "terminal")
        ):
            self._reward_bar(
                name.title(),
                float(components.get(name, 0.0)),
                x,
                139 + index * 28,
                256,
                colors[name],
            )

        self._divider(x, 350, 256)
        self._section_title("Observation", x, 367, self.ACCENT)
        observation_count = int(env.observation_space.shape[0])
        self._metric_row("State values", str(observation_count), x, 405, 256)
        self._metric_row("Action space", f"{ACTION_COUNT} steer × pedal", x, 435, 256)
        self._metric_row("Range sensors", "Visible" if self.show_sensors else "Hidden", x, 465, 256)
        self._metric_row(
            "Traffic model",
            "Dynamic" if env.dynamic_traffic else "Fixed",
            x,
            495,
            256,
        )

        self._divider(x, 535, 256)
        self._section_title("Collision history", x, 552, self.DANGER)
        collisions = int(env.hud_data.get("collisions", 0))
        completions = int(env.hud_data.get("completions", 0))
        collision_types = env.hud_data.get("collision_types", {})
        self._metric_row("All collisions", str(collisions), x, 590, 256)
        self._metric_row("Front impact", str(collision_types.get("FRONT IMPACT", 0)), x, 620, 256)
        self._metric_row("Side impact", str(collision_types.get("SIDE IMPACT", 0)), x, 650, 256)
        self._metric_row("Rear impact", str(collision_types.get("REAR IMPACT", 0)), x, 680, 256)
        self._metric_row("Completed", str(completions), x, 710, 256)
        self._text("H toggles range geometry", self.font_tiny, self.MUTED, x, 758)

    def _draw_analysis_right(self, env: NeonHighwayEnv) -> None:
        x = self.RIGHT_PANEL.x + 18
        self._text("Policy trace", self.font_medium, self.TEXT, x, 31)
        self._text("Live diagnostics", self.font_tiny, self.MUTED, x, 59)
        self._divider(x, 88, 256)

        q_values = [float(value) for value in env.hud_data.get("q_values", [0.0] * ACTION_COUNT)]
        self._section_title("Q values", x, 104, self.ACCENT)
        self._draw_q_values(env, q_values, x, 146, 256)

        self._divider(x, 248, 256)
        self._section_title("Training trend", x, 261, self.SUCCESS)
        recent_returns = list(env.hud_data.get("recent_returns", []))
        self._draw_sparkline(pygame.Rect(x, 296, 256, 96), recent_returns)

        self._divider(x, 410, 256)
        self._section_title("Decision trace", x, 425, self.ACCENT)
        learned = str(env.hud_data.get("raw_action", env._info()["action"]))
        residual = str(
            env.hud_data.get(
                "dagger_decision",
                env.hud_data.get("preference_decision", "Base policy"),
            )
        )
        shield = str(env.hud_data.get("lane_veto_reason", "")) or "Clear"
        self._metric_row("Learned", learned, x, 463, 256)
        self._metric_row("Residual", residual, x, 491, 256)
        self._metric_row("Safety shield", shield, x, 519, 256)

        self._divider(x, 551, 256)
        self._section_title("Session and performance", x, 563, self.SUCCESS)
        avoidable_rate = env.avoidable_following_steps / max(env.step_count, 1)
        mean_return = float(env.hud_data.get("mean_return", 0.0))
        best_return = float(env.hud_data.get("best_return", 0.0))
        drive_rows = (
            ("Episode", str(env.episode_index)),
            ("Live return", f"{env.episode_return:+.2f}"),
            ("Mean / 20", f"{mean_return:+.2f}"),
            ("Best", f"{best_return:+.2f}"),
            ("Overtakes / passed", f"{env.overtakes} / {env.passed_by_traffic}"),
            ("Lane changes", str(env.lane_changes)),
            ("Traffic lane changes", str(env.traffic_lane_changes)),
            ("Longitudinal accel", f"{env.longitudinal_acceleration:+.2f} m/s²"),
            ("Near misses", str(env.near_misses)),
            ("Avoidable follow", f"{avoidable_rate:.0%}"),
        )
        for index, (label, value) in enumerate(drive_rows):
            self._metric_row(label, value, x, 596 + index * 19, 256)

    def _draw_q_values(
        self,
        env: NeonHighwayEnv,
        q_values: list[float],
        x: int,
        y: int,
        width: int,
    ) -> None:
        if len(q_values) != ACTION_COUNT:
            q_values = [0.0] * ACTION_COUNT
        minimum, maximum = min(q_values), max(q_values)
        spread = maximum - minimum
        best = q_values.index(maximum)
        selected_steer, _ = decode_action(env.last_action)
        label_width = 48
        cell_width = (width - label_width) // 3
        for pedal, label in enumerate(("Brake", "Coast", "Gas")):
            self._text(
                label,
                self.font_tiny,
                self.MUTED,
                x + label_width + pedal * cell_width,
                y - 23,
            )
        for steer, label in enumerate(("Left", "Keep", "Right")):
            row_y = y + steer * 30
            self._text(
                label,
                self.font_tiny,
                self.TEXT if steer == selected_steer else self.MUTED,
                x,
                row_y,
            )
            for pedal in range(3):
                action = encode_action(steer, pedal)
                value = q_values[action]
                normalized = (value - minimum) / spread if spread > 1e-6 else 0.5
                cell_x = x + label_width + pedal * cell_width
                cell_rect = pygame.Rect(cell_x, row_y - 1, cell_width - 7, 24)
                pygame.draw.rect(self.screen, self.PANEL_RAISED, cell_rect, border_radius=5)
                pygame.draw.rect(
                    self.screen,
                    self.ACCENT if action == best else self.DIVIDER,
                    (cell_x, row_y + 20, max(2, round((cell_width - 7) * normalized)), 3),
                    border_radius=2,
                )
                self._center_text_in_rect(
                    f"{value:+.1f}",
                    self.font_tiny,
                    self.TEXT if action == best else self.MUTED,
                    cell_rect,
                )

    def _draw_teacher_status(self, env: NeonHighwayEnv) -> None:
        """Keep teaching state in the left cockpit rail, never over the road."""
        rect = pygame.Rect(self.LEFT_PANEL.x + 10, 558, self.LEFT_PANEL.width - 20, 226)
        self._panel(rect, raised=True)
        x = rect.x + 14
        self._section_title("Human teacher", x, rect.y + 16, self.SUCCESS)
        proposal = str(env.hud_data.get("dagger_proposal", "Hold"))
        last_label = str(env.hud_data.get("dagger_last_label", "Waiting for a label"))
        labels = int(env.hud_data.get("dagger_labels", 0))
        lane = int(env.hud_data.get("dagger_lane_corrections", 0))
        speed = int(env.hud_data.get("dagger_speed_corrections", 0))
        self._text(
            f"Proposal  {proposal}",
            self.font_small_bold,
            self.TEXT,
            x,
            rect.y + 57,
            max_width=244,
        )
        self._text(last_label, self.font_tiny, self.MUTED, x, rect.y + 91, max_width=244)
        self._text(
            f"{labels} labels · {lane} lane · {speed} speed",
            self.font_tiny,
            self.TEXT,
            x,
            rect.y + 126,
            max_width=244,
        )
        self._text(
            "Arrows/WASD · Enter · U undo",
            self.font_tiny,
            self.ACCENT,
            x,
            rect.y + 169,
            max_width=244,
        )

    def _draw_crash_overlay(self, env: NeonHighwayEnv, phase: float) -> None:
        collision = env.last_collision
        if collision is None:
            title, detail = "Drive ended", "Collision detected"
        else:
            title = "Impact detected"
            detail = f"{collision.kind.title()} · {collision.severity.title()}"
        card = pygame.Rect(self.RIGHT_PANEL.x + 10, 352, self.RIGHT_PANEL.width - 20, 250)
        self._panel(card, raised=True)
        x = card.x + 16
        pygame.draw.rect(self.screen, self.DANGER, (x, card.y + 17, 42, 4), border_radius=2)
        self._text(title, self.font_large, self.TEXT, x, card.y + 40, max_width=240)
        self._text(detail, self.font_small_bold, self.DANGER, x, card.y + 90, max_width=240)
        if collision is not None:
            self._text(
                f"Impact speed {collision.impact_speed * 3.6:.1f} km/h",
                self.font_tiny,
                self.TEXT,
                x,
                card.y + 124,
            )
            self._text(f"Lane {collision.lane + 1}", self.font_tiny, self.MUTED, x, card.y + 151)
        self._text(
            f"Episode return {env.episode_return:+.2f}",
            self.font_tiny,
            self.MUTED,
            x,
            card.y + 181,
            max_width=240,
        )
        self._text("Replay memory updated", self.font_tiny, self.ACCENT, x, card.y + 211)

    def _draw_completion_overlay(self, env: NeonHighwayEnv) -> None:
        card = pygame.Rect(self.RIGHT_PANEL.x + 10, 352, self.RIGHT_PANEL.width - 20, 220)
        self._panel(card, raised=True)
        x = card.x + 16
        pygame.draw.rect(self.screen, self.SUCCESS, (x, card.y + 17, 42, 4), border_radius=2)
        self._text("Route complete", self.font_large, self.TEXT, x, card.y + 40, max_width=240)
        self._text(
            f"{env.challenges_resolved} waves cleared",
            self.font_small_bold,
            self.SUCCESS,
            x,
            card.y + 96,
        )
        self._text(
            f"Episode return {env.episode_return:+.2f}",
            self.font_tiny,
            self.TEXT,
            x,
            card.y + 132,
        )
        self._text(
            "Completion stored in replay memory",
            self.font_tiny,
            self.MUTED,
            x,
            card.y + 172,
            max_width=240,
        )

    def _progress_state(self, env: NeonHighwayEnv) -> tuple[str, str, float]:
        total = int(env.hud_data.get("training_total", 0))
        step = int(env.hud_data.get("training_step", 0))
        if total > 0:
            return "Training progress", f"{step:,} / {total:,}", min(step / total, 1.0)
        if env.endless:
            best = float(env.hud_data.get("longest_survival", 0.0))
            value = f"{format_duration(env.elapsed_seconds)} · {env.ego_position / 1000.0:.2f} km"
            progress = env.elapsed_seconds / max(best, env.elapsed_seconds, 1.0)
            return "Current drive", value, progress
        progress = min(env.step_count / max(env.max_episode_steps, 1), 1.0)
        return "Episode progress", f"{progress * 100:.1f}%", progress

    @staticmethod
    def _format_ttc(value: float) -> str:
        return f"{value:.1f} s" if np.isfinite(value) else "Clear"

    def _risk_color(self, ttc: float) -> tuple[int, int, int]:
        if ttc < 2.0:
            return self.DANGER
        if ttc < 4.0:
            return self.WARNING
        return self.SUCCESS

    def _draw_sparkline(self, rect: pygame.Rect, values: list[float]) -> None:
        pygame.draw.rect(self.screen, self.PANEL_RAISED, rect, border_radius=9)
        pygame.draw.line(
            self.screen,
            self.DIVIDER,
            (rect.left + 9, rect.centery),
            (rect.right - 9, rect.centery),
        )
        if len(values) < 2:
            self._center_text_in_rect(
                "Waiting for completed episodes",
                self.font_tiny,
                self.MUTED,
                rect,
            )
            return
        minimum, maximum = min(values), max(values)
        spread = max(maximum - minimum, 1.0)
        points = [
            (
                round(rect.left + 9 + index * (rect.width - 18) / (len(values) - 1)),
                round(rect.bottom - 9 - (value - minimum) / spread * (rect.height - 18)),
            )
            for index, value in enumerate(values)
        ]
        pygame.draw.aalines(self.screen, self.ACCENT, False, points)
        pygame.draw.circle(self.screen, self.SUCCESS, points[-1], 4)
        self._text(f"{maximum:+.1f}", self.font_tiny, self.MUTED, rect.left + 8, rect.top + 4)
        self._text(f"{minimum:+.1f}", self.font_tiny, self.MUTED, rect.left + 8, rect.bottom - 22)

    def _reward_bar(
        self,
        label: str,
        value: float,
        x: int,
        y: int,
        width: int,
        color: tuple[int, int, int],
    ) -> None:
        self._text(label, self.font_tiny, self.MUTED, x, y - 5)
        bar_x = x + 78
        bar_width = width - 132
        pygame.draw.rect(self.screen, self.PANEL_RAISED, (bar_x, y, bar_width, 7), border_radius=3)
        normalized = min(abs(value) / 0.22, 1.0)
        draw_color = color if value >= 0 else self.DANGER
        pygame.draw.rect(
            self.screen,
            draw_color,
            (bar_x, y, max(1, round(bar_width * normalized)), 7),
            border_radius=3,
        )
        self._text(f"{value:+.3f}", self.font_tiny, self.TEXT, x + width - 48, y - 6)

    def _value_bar(
        self,
        label: str,
        value: float,
        x: int,
        y: int,
        width: int,
        color: tuple[int, int, int],
    ) -> None:
        self._text(label, self.font_tiny, self.MUTED, x, y - 20)
        pygame.draw.rect(self.screen, self.PANEL_RAISED, (x, y + 4, width, 9), border_radius=4)
        pygame.draw.rect(
            self.screen,
            color,
            (x, y + 4, max(2, round(width * value)), 9),
            border_radius=4,
        )

    def _progress_bar(
        self,
        x: int,
        y: int,
        width: int,
        value: float,
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(self.screen, self.PANEL_RAISED, (x, y, width, 9), border_radius=4)
        pygame.draw.rect(
            self.screen,
            color,
            (x, y, max(2, round(width * min(max(value, 0.0), 1.0))), 9),
            border_radius=4,
        )

    def _small_card(
        self,
        rect: pygame.Rect,
        label: str,
        value: str,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        pygame.draw.rect(self.screen, self.PANEL_RAISED, rect, border_radius=9)
        self._text(label, self.font_tiny, self.MUTED, rect.x + 11, rect.y + 9)
        self._text(
            value,
            self.font_small_bold,
            color or self.TEXT,
            rect.x + 11,
            rect.y + 37,
            max_width=rect.width - 22,
        )

    def _ttc_card(
        self,
        rect: pygame.Rect,
        label: str,
        value: str,
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(self.screen, self.PANEL_RAISED, rect, border_radius=9)
        pygame.draw.rect(
            self.screen,
            color,
            (rect.x, rect.y + 9, 3, rect.height - 18),
            border_radius=2,
        )
        self._text(label, self.font_tiny, self.MUTED, rect.x + 12, rect.y + 10)
        self._text(
            value,
            self.font_medium,
            color,
            rect.x + 12,
            rect.y + 39,
            max_width=rect.width - 20,
        )

    def _metric_row(self, label: str, value: str, x: int, y: int, width: int) -> None:
        self._text(label, self.font_tiny, self.MUTED, x, y)
        rendered = self.font_small_bold.render(str(value), True, self.TEXT)
        if rendered.get_width() > width - 112:
            value_text = self._ellipsize(str(value), self.font_small_bold, width - 112)
            rendered = self.font_small_bold.render(value_text, True, self.TEXT)
        self.screen.blit(rendered, (x + width - rendered.get_width(), y - 2))

    def _section_title(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(self.screen, color, (x, y + 2, 3, 20), border_radius=2)
        self._text(text, self.font_small_bold, self.TEXT, x + 12, y)

    def _divider(self, x: int, y: int, width: int) -> None:
        pygame.draw.line(self.screen, self.DIVIDER, (x, y), (x + width, y))

    def _panel(self, rect: pygame.Rect, *, raised: bool = False) -> None:
        pygame.draw.rect(
            self.screen,
            self.PANEL_RAISED if raised else self.PANEL,
            rect,
            border_radius=14,
        )
        pygame.draw.rect(self.screen, self.DIVIDER, rect, 1, border_radius=14)

    def _side_status(
        self,
        rect: pygame.Rect,
        title: str,
        detail: str,
        color: tuple[int, int, int],
    ) -> None:
        self._panel(rect, raised=True)
        self._center_text_in_rect(title, self.font_medium, color, rect.move(0, -18))
        self._center_text_in_rect(detail, self.font_tiny, self.MUTED, rect.move(0, 22))

    def _pill(
        self,
        label: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        *,
        align_right: bool = False,
    ) -> None:
        surface = self.font_tiny.render(label, True, self.CANVAS)
        rect = pygame.Rect(x, y, surface.get_width() + 18, 25)
        if align_right:
            rect.right = x
        pygame.draw.rect(self.screen, color, rect, border_radius=12)
        self.screen.blit(surface, (rect.x + 9, rect.y + 3))

    @staticmethod
    def _ellipsize(text: str, font: pygame.font.Font, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text
        candidate = text
        while candidate and font.size(candidate + "…")[0] > max_width:
            candidate = candidate[:-1]
        return candidate.rstrip() + "…"

    def _text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        x: int,
        y: int,
        *,
        max_width: int | None = None,
    ) -> None:
        display_text = str(text)
        if max_width is not None:
            display_text = self._ellipsize(display_text, font, max_width)
        self.screen.blit(font.render(display_text, True, color), (x, y))

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
