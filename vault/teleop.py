"""teleop -- keyboard/gamepad nominal-task test rig for the contact/slam safety filter.

Drives the REAL deployed balance controller (vault-controller's CCoupledController, same as
evaluate.py) with a live human reference command [v_ref, psi_dot_ref] from the keyboard or a
gamepad, passes its output through a safety filter built from safety_filter.py's modular
fallback/monitor/intervention pieces (OptimizationIntervention -- "the QCBF" -- by default; the
simpler fixed-eps SwitchIntervention is available via --filter-mode switch), and steps the MuJoCo
plant (contact_geometry=True). Prints a live HUD: the full-ODD margin (f_cert.odd_margin), the
contact/slam margin (contact_margin.margin), and whether the filter overrode the human's command
this step -- adapted from examples/manual_control_debug.py's live margin_g printout, generalized
from keyboard-only to keyboard+gamepad via pygame.joystick.

REQUIRES the vault-controller repo (no fallback, same as evaluate.py):
    make -C ../vault-controller/balance_controller/c lib
    PYTHONPATH=$PWD:../vault-controller python -m vault.teleop [--no-filter] [--filter-mode dcbf|switch]

Controls: W/S = forward/back, A/D = turn left/right, R = reset, F = toggle filter, ESC = quit.
Gamepad (if connected): left stick vertical = forward/back, left stick horizontal = turn.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import config as C
from . import contact_margin as CM
from . import f_cert as F
from .checkpoints import load_safety_sac
from .safety_filter import SACFallback, ValueMonitor, SwitchIntervention, OptimizationIntervention
from .mujoco_plant import MujocoPlant

try:
    from balance_controller.c_coupled_controller import CCoupledController
except ImportError as e:  # fail loud -- teleop is meaningless without the real controller
    raise SystemExit(
        "vault.teleop requires the vault-controller repo (the deployed C iLQR), not on PYTHONPATH.\n"
        "  1) clone vault-controller beside this repo\n"
        "  2) make -C vault-controller/balance_controller/c lib\n"
        "  3) export PYTHONPATH=/path/to/vault-controller:$PYTHONPATH\n"
        f"(import error: {e})"
    )

JOY_DEADZONE = 0.15


def ref_from_keys(keys, pygame) -> np.ndarray:
    v = C.V_ODD[1] * (float(keys[pygame.K_w]) - float(keys[pygame.K_s]))
    psi = C.PSI_ODD * (float(keys[pygame.K_a]) - float(keys[pygame.K_d]))
    return np.array([v, psi])


def ref_from_joystick(js) -> np.ndarray:
    def dz(v):
        return 0.0 if abs(v) < JOY_DEADZONE else v
    v = -C.V_ODD[1] * dz(js.get_axis(1))      # forward = stick up = negative raw axis
    psi = -C.PSI_ODD * dz(js.get_axis(0))
    return np.array([v, psi])


def teleop_step(plant, ctrl, cf, x, t, ref, filter_on):
    """One control step: human ref -> deployed controller -> filter -> plant. Returns
    (x_next, u_task, u_safe, overridden, g_odd, g_contact). g_odd is f_cert.odd_margin --
    the full-ODD margin (pitch/roll AND v/psi_dot-in-envelope) that ContactSafetyEnv actually
    trains/terminates against -- not the narrower pitch/roll-only margin()."""
    u_task = np.clip(np.asarray(ctrl.step(x, ref, t), float), -C.TAU_MAX, C.TAU_MAX)
    if filter_on:
        u_safe, overridden = cf(x, u_task, plant.mu)
    else:
        u_safe, overridden = u_task, False
    x_next = plant.step(u_safe)
    g_odd = float(F.odd_margin(x_next))
    g_ct = CM.margin(plant)
    return x_next, u_task, u_safe, overridden, g_odd, g_ct


def _make_filter(mode: str, model, gamma: float, eps: float):
    fallback = SACFallback(model)
    monitor = ValueMonitor(model)
    if mode == "dcbf":
        return OptimizationIntervention(monitor, fallback, gamma=gamma)
    if mode == "switch":
        return SwitchIntervention(monitor, fallback, eps=eps)
    raise ValueError(f"filter-mode must be 'dcbf' or 'switch', got {mode!r}")


def _dry_run(steps: int, mu: float, filter_on: bool, filter_mode: str, gamma: float, eps: float,
             model_path: str):
    """Headless self-test: scripted ref sequence, no pygame/display. Used to validate the
    control loop wiring without requiring a keyboard/screen (see verification in the plan)."""
    plant = MujocoPlant(wheel="cylinder", mu=mu, dt=0.0005, substeps=20, contact_geometry=True)
    ctrl = CCoupledController()
    model = load_safety_sac(model_path)
    cf = _make_filter(filter_mode, model, gamma, eps)

    plant.reset(np.array([0.0, 0.0, 0.0, 0.0]))
    ctrl.reset()
    x = plant.get_state()
    n_overridden = 0
    worst = (1.0, 1.0)
    for i in range(steps):
        ref = np.array([1.2, 2.0 if i > steps // 2 else -2.0])   # aggressive scripted command
        x, u_task, u_safe, overridden, g_odd, g_ct = teleop_step(
            plant, ctrl, cf, x, i * plant.dt, ref, filter_on)
        n_overridden += int(overridden)
        worst = (min(worst[0], g_odd), min(worst[1], g_ct))
        if i % max(1, steps // 10) == 0:
            print(f"t={i:4d} x={np.round(x, 2)} u_task={np.round(u_task, 2)} "
                  f"u_safe={np.round(u_safe, 2)} override={overridden} g_odd={g_odd:+.3f} g_ct={g_ct:+.3f}")
    print(f"\ndry-run done: {n_overridden}/{steps} steps overridden "
          f"({100 * n_overridden / steps:.0f}%), worst (g_odd, g_contact) = {worst}")


def _live(mu: float, filter_on: bool, filter_mode: str, gamma: float, eps: float, model_path: str):
    import pygame
    plant = MujocoPlant(wheel="cylinder", mu=mu, dt=0.0005, substeps=20, contact_geometry=True)
    ctrl = CCoupledController()
    model = load_safety_sac(model_path)
    cf = _make_filter(filter_mode, model, gamma, eps)

    try:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(plant.model, plant.data)
    except Exception as e:  # no display, headless box, etc. -- keep going, HUD-only
        print(f"(no 3D viewer: {e} -- continuing with console HUD only)")
        viewer = None

    pygame.init()
    pygame.joystick.init()
    screen = pygame.display.set_mode((420, 200))
    pygame.display.set_caption("vault teleop (focus for keyboard) -- WASD, R reset, F filter, ESC quit")
    clock = pygame.time.Clock()
    joy = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
    if joy is not None:
        joy.init()
        print(f"gamepad connected: {joy.get_name()}")
    else:
        print("no gamepad detected -- keyboard only")

    plant.reset(np.array([0.0, 0.0, 0.0, 0.0]))
    ctrl.reset()
    x = plant.get_state()
    t0 = time.time()
    step = 0
    running = True
    print("\ncontrols: W/S forward/back, A/D turn, R reset, F toggle filter, ESC quit\n")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
            continue
        if keys[pygame.K_r]:
            plant.reset(np.array([0.0, 0.0, 0.0, 0.0]))
            ctrl.reset()
            x = plant.get_state()
            step = 0
            print("reset")
        if keys[pygame.K_f]:
            filter_on = not filter_on
            print(f"filter {'ON' if filter_on else 'OFF'}")
            time.sleep(0.2)   # debounce the toggle key

        ref = ref_from_joystick(joy) if joy is not None else ref_from_keys(keys, pygame)
        x, u_task, u_safe, overridden, g_odd, g_ct = teleop_step(
            plant, ctrl, cf, x, step * plant.dt, ref, filter_on)
        step += 1

        if viewer is not None:
            viewer.sync()
        status = "OVERRIDE" if overridden else "task"
        print(f"\rt={step * plant.dt:6.2f}s ref={np.round(ref, 2)} u={np.round(u_safe, 2)} "
              f"[{status:8s}] g_odd={g_odd:+.3f} g_contact={g_ct:+.3f}   ", end="")
        if g_odd < 0 or g_ct < 0:
            print(f"\n  *** SAFETY BREACH  g_odd={g_odd:+.3f} g_contact={g_ct:+.3f} ***")

        clock.tick(int(round(1.0 / plant.dt)))

    print(f"\n\nsession ended after {step} steps ({time.time() - t0:.1f}s wall-clock)")
    pygame.quit()
    if viewer is not None:
        viewer.close()


def main():
    ap = argparse.ArgumentParser(description="Keyboard/gamepad teleop through the contact QCBF filter")
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--filter-mode", choices=["dcbf", "switch"], default="dcbf",
                     help="dcbf = QCBFFilter (rate-limited, the actual QCBF); "
                          "switch = ContactValueFilter (fixed-eps least-restrictive baseline)")
    ap.add_argument("--gamma", type=float, default=0.2,
                     help="QCBFFilter: max fractional one-step drop in the safety value")
    ap.add_argument("--eps", type=float, default=0.0, help="ContactValueFilter: fixed safety floor")
    ap.add_argument("--no-filter", action="store_true", help="start with the filter disabled")
    ap.add_argument("--dry-run", type=int, default=0, metavar="STEPS",
                     help="headless self-test with a scripted command, no pygame/display")
    ap.add_argument("--model", type=str, default=str(C.MODELS / "contact_safety_sac_v3"),
                     help="SafetySAC checkpoint (.zip) to load for the filter's critic")
    args = ap.parse_args()

    if args.dry_run:
        _dry_run(args.dry_run, args.mu, not args.no_filter, args.filter_mode, args.gamma, args.eps,
                  args.model)
        return 0
    _live(args.mu, not args.no_filter, args.filter_mode, args.gamma, args.eps, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
