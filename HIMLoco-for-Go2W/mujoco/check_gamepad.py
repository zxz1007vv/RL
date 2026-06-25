import time

import pygame


def main():
    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()
    print(f"Detected joystick count: {count}")
    if count == 0:
        print("No gamepad detected. Check F710 receiver, batteries, and X/D switch.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick 0: {joystick.get_name()}")
    print(f"axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}")

    try:
        from pygame._sdl2 import controller

        controller.init()
        print(f"SDL says joystick 0 is controller: {controller.is_controller(0)}")
        if controller.is_controller(0):
            pad = controller.Controller(0)
            pad.init()
            print(f"SDL Controller name: {pad.name}")
    except Exception as exc:
        print(f"SDL Controller API unavailable: {exc}")

    print("Move sticks / press buttons. Ctrl+C to exit.")
    while True:
        pygame.event.pump()
        axes = [round(joystick.get_axis(i), 3) for i in range(joystick.get_numaxes())]
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
        print(f"\raxes={axes} buttons={buttons} hats={hats}", end="")
        time.sleep(0.05)


if __name__ == "__main__":
    main()
