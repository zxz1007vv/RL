import threading
import time


class CmdGenerator:
    """Generate [vx, vy, yaw_target] commands from a gamepad.

    Logitech F710 recommended setup:
    - put the side switch in X mode (XInput)
    - keep the front Mode light off, otherwise the D-pad and left stick are swapped

    In XInput mode this class first uses SDL's standard Controller mapping.
    If SDL does not recognize the device as a Controller, it falls back to the
    raw Joystick axes from config.yaml.

    Default command mapping:
    - left stick up/down: forward/backward velocity
    - left stick left/right: lateral velocity
    - right stick left/right: yaw rate or accumulated yaw target
    - A button: reset all commands
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "Gamepad control needs pygame. Install it with: python -m pip install pygame"
            ) from exc

        self.pygame = pygame
        self.device_index = int(cfg.get("device_index", 0))
        self.deadzone = float(cfg.get("deadzone", 0.08))
        self.max_vx = float(cfg.get("max_vx", 1.0))
        self.max_vy = float(cfg.get("max_vy", 0.5))
        self.max_yaw_rate = float(cfg.get("max_yaw_rate", 1.0))
        self.yaw_mode = cfg.get("yaw_mode", "heading")
        self.prefer_sdl_controller = bool(cfg.get("prefer_sdl_controller", True))
        self.axis_left_x = int(cfg.get("axis_left_x", 0))
        self.axis_left_y = int(cfg.get("axis_left_y", 1))
        self.axis_right_x = int(cfg.get("axis_right_x", 3))
        self.reset_button = int(cfg.get("reset_button", 0))
        self.update_dt = float(cfg.get("update_dt", 0.02))

        self.vx = 0.0
        self.vy = 0.0
        self.yaw_target = 0.0
        self._lock = threading.Lock()
        self._running = True
        self.controller = None
        self.use_controller = False

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= self.device_index:
            raise RuntimeError(
                "No gamepad found. Check connection with: python -m pygame.examples.joystick"
            )

        self.joystick = pygame.joystick.Joystick(self.device_index)
        self.joystick.init()
        self._try_open_sdl_controller()

        if self.use_controller:
            print(f"Gamepad connected through SDL Controller mapping: {self.controller.name}")
        else:
            print(
                "Gamepad connected through raw Joystick mapping: "
                f"{self.joystick.get_name()} "
                f"axes={self.joystick.get_numaxes()} buttons={self.joystick.get_numbuttons()}"
            )
            print("Tip for Logitech F710: move the side switch to X mode and restart this script.")

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _pump_events(self):
        try:
            self.pygame.event.pump()
        except self.pygame.error as exc:
            if "video system not initialized" not in str(exc):
                raise
            self.pygame.display.init()
            self.pygame.event.pump()

    def _try_open_sdl_controller(self):
        if not self.prefer_sdl_controller:
            return

        try:
            from pygame._sdl2 import controller as sdl_controller
        except Exception:
            return

        try:
            sdl_controller.init()
            if not sdl_controller.is_controller(self.device_index):
                return
            self.controller = sdl_controller.Controller(self.device_index)
            self.controller.init()
            self.use_controller = True
        except Exception as exc:
            print(f"SDL Controller mapping unavailable, falling back to raw Joystick: {exc}")
            self.controller = None
            self.use_controller = False

    def _apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def _get_joystick_axis(self, axis_id):
        if axis_id < self.joystick.get_numaxes():
            return self._apply_deadzone(self.joystick.get_axis(axis_id))
        return 0.0

    def _get_controller_axis(self, axis_id):
        # SDL_GameControllerAxis enum:
        # 0 LEFTX, 1 LEFTY, 2 RIGHTX, 3 RIGHTY, 4 TRIGGERLEFT, 5 TRIGGERRIGHT
        value = self.controller.get_axis(axis_id)
        if abs(value) > 1.0:
            value = value / 32767.0
        return self._apply_deadzone(float(value))

    def _get_controller_button(self, button_id):
        # SDL_GameControllerButton enum:
        # 0 A, 1 B, 2 X, 3 Y, 4 BACK, 5 GUIDE, 6 START, ...
        return bool(self.controller.get_button(button_id))

    def _loop(self):
        while self._running:
            self._pump_events()

            if self.use_controller:
                # SDL Controller mapping is standardized, especially useful for F710 in XInput mode.
                # Common convention: pushing stick up gives negative Y.
                left_x = self._get_controller_axis(0)
                left_y = self._get_controller_axis(1)
                right_x = self._get_controller_axis(2)
                reset_pressed = self._get_controller_button(0)
            else:
                # Raw Joystick fallback. Axis IDs are configured in config.yaml.
                left_x = self._get_joystick_axis(self.axis_left_x)
                left_y = self._get_joystick_axis(self.axis_left_y)
                right_x = self._get_joystick_axis(self.axis_right_x)
                reset_pressed = (
                    self.reset_button < self.joystick.get_numbuttons()
                    and self.joystick.get_button(self.reset_button)
                )

            with self._lock:
                self.vx = -left_y * self.max_vx
                self.vy = -left_x * self.max_vy
                if self.yaw_mode == "yaw_rate":
                    self.yaw_target = -right_x * self.max_yaw_rate
                else:
                    self.yaw_target += -right_x * self.max_yaw_rate * self.update_dt

                if reset_pressed:
                    self.vx = 0.0
                    self.vy = 0.0
                    self.yaw_target = 0.0

            time.sleep(self.update_dt)

    def get_cmd(self):
        with self._lock:
            return [self.vx, self.vy, self.yaw_target]
