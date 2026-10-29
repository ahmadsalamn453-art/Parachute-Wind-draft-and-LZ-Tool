"""
Parachute Wind Drift and Landing Zone Radius Calculator
Mission 05 (SEDS-TASK-RCK-SRILANKA-020) - Ahmad Salman

Pure Python standard library (no third-party dependencies). Python 3.8+.

Purpose
-------
Model wind drift of a dual-deploy rocket (drogue at apogee, main chute at
500 ft AGL) under 5 m/s and 10 m/s winds and verify the landing point stays
inside the approved 400 m safe recovery radius around the launch pad.

Conventions
-----------
* Wind direction is METEOROLOGICAL: the bearing the wind blows FROM,
  degrees clockwise from true north (135 = wind from the south-east).
* Drift (landing) bearing is the opposite: (wind_dir + 180) % 360.
* Cartesian frame: origin = launch pad, +x = East, +y = North.
* Altitudes passed to calculate_descent() are metres above mean sea level
  (MSL); descent is computed on height above ground level (AGL) =
  deployment altitude - ground elevation.
"""

import math
from typing import Optional

COMPASS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

FT_TO_M = 0.3048
MAIN_DEPLOY_500FT_AGL_M = 500.0 * FT_TO_M  # 152.4 m, per altimeter spec (Task 02)


def compass_point(bearing_deg: float) -> str:
    """Nearest 8-point compass label for a bearing in degrees."""
    return COMPASS_8[int(round((bearing_deg % 360.0) / 45.0)) % 8]


class ParachuteDriftCalculator:
    def __init__(
        self,
        safe_radius_m: float = 400.0,
        baseline_descent_ms: float = 5.2,
        drogue_descent_ms: Optional[float] = None,
        main_deploy_agl_m: Optional[float] = None,
        shear_exponent: float = 0.0,
        wind_ref_height_m: float = 10.0,
        safety_factor: float = 1.0,
        n_steps: int = 1000,
    ):
        if safe_radius_m <= 0:
            raise ValueError("safe_radius_m must be positive.")
        if baseline_descent_ms <= 0:
            raise ValueError("baseline_descent_ms must be positive.")
        if (drogue_descent_ms is None) != (main_deploy_agl_m is None):
            raise ValueError(
                "drogue_descent_ms and main_deploy_agl_m must be given together."
            )
        if drogue_descent_ms is not None:
            if drogue_descent_ms <= 0:
                raise ValueError("drogue_descent_ms must be positive.")
            if drogue_descent_ms < baseline_descent_ms:
                raise ValueError(
                    "drogue_descent_ms must not be slower than the main descent speed."
                )
            if main_deploy_agl_m < 0:
                raise ValueError("main_deploy_agl_m must be non-negative.")
        if shear_exponent < 0:
            raise ValueError("shear_exponent must be non-negative.")
        if wind_ref_height_m <= 0:
            raise ValueError("wind_ref_height_m must be positive.")
        if safety_factor < 1.0:
            raise ValueError("safety_factor must be >= 1.0.")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")

        self.safe_radius_m = float(safe_radius_m)
        self.baseline_descent_ms = float(baseline_descent_ms)
        self.drogue_descent_ms = None if drogue_descent_ms is None else float(drogue_descent_ms)
        self.main_deploy_agl_m = None if main_deploy_agl_m is None else float(main_deploy_agl_m)
        self.shear_exponent = float(shear_exponent)
        self.wind_ref_height_m = float(wind_ref_height_m)
        self.safety_factor = float(safety_factor)
        self.n_steps = int(n_steps)

    def _wind_factor(self, h_agl: float) -> float:
        if self.shear_exponent == 0.0:
            return 1.0
        return (max(h_agl, self.wind_ref_height_m) / self.wind_ref_height_m) ** self.shear_exponent

    def _phases(self, effective_alt_m: float):
        if self.drogue_descent_ms is None or effective_alt_m <= self.main_deploy_agl_m:
            return [(0.0, effective_alt_m, self.baseline_descent_ms)]
        return [
            (0.0, self.main_deploy_agl_m, self.baseline_descent_ms),
            (self.main_deploy_agl_m, effective_alt_m, self.drogue_descent_ms),
        ]

    def _integrate(self, effective_alt_m: float, wind_speed_ms: float):
        fall_time = 0.0
        drift = 0.0
        for h_low, h_high, speed in self._phases(effective_alt_m):
            span = h_high - h_low
            if span <= 0:
                continue
            dh = span / self.n_steps
            dt = dh / speed
            for i in range(self.n_steps):
                h_mid = h_low + (i + 0.5) * dh
                fall_time += dt
                drift += wind_speed_ms * self._wind_factor(h_mid) * dt
        return fall_time, drift

    def calculate_descent(
        self,
        deployment_alt_m: float,
        ground_elevation_m: float,
        wind_speed_ms: float,
        wind_dir_deg: float,
    ) -> dict:
        if wind_speed_ms < 0:
            raise ValueError("wind_speed_ms must be non-negative.")
        if not 0.0 <= wind_dir_deg <= 360.0:
            raise ValueError("wind_dir_deg must be between 0 and 360.")

        effective_alt_m = deployment_alt_m - ground_elevation_m
        if effective_alt_m <= 0:
            raise ValueError("deployment_alt_m must be above ground_elevation_m.")

        fall_duration_s, total_drift_m = self._integrate(effective_alt_m, wind_speed_ms)

        drift_bearing_deg = (wind_dir_deg + 180.0) % 360.0
        drift_rad = math.radians(drift_bearing_deg)
        target_x = total_drift_m * math.sin(drift_rad)
        target_y = total_drift_m * math.cos(drift_rad)

        design_drift_m = total_drift_m * self.safety_factor

        return {
            "effective_alt_m": effective_alt_m,
            "fall_duration_s": fall_duration_s,
            "total_drift_m": total_drift_m,
            "design_drift_m": design_drift_m,
            "target_x": target_x,
            "target_y": target_y,
            "wind_speed_ms": float(wind_speed_ms),
            "wind_from_bearing_deg": wind_dir_deg % 360.0,
            "wind_from_compass": compass_point(wind_dir_deg),
            "drift_bearing_deg": drift_bearing_deg,
            "drift_compass": compass_point(drift_bearing_deg),
            "landing_bearing": f"{drift_bearing_deg:.0f}° {compass_point(drift_bearing_deg)}",
            "margin_m": self.safe_radius_m - design_drift_m,
            "is_compliant": design_drift_m <= self.safe_radius_m,
            "safe_radius_m": self.safe_radius_m,
        }

    def max_compliant_altitude_agl(self, wind_speed_ms: float) -> Optional[float]:
        if wind_speed_ms < 0:
            raise ValueError("wind_speed_ms must be non-negative.")

        def design_drift(alt: float) -> float:
            return self._integrate(alt, wind_speed_ms)[1] * self.safety_factor

        floor_alt = self.main_deploy_agl_m if self.main_deploy_agl_m else 1.0
        if design_drift(floor_alt) > self.safe_radius_m:
            return None
        ceiling = 50000.0
        if design_drift(ceiling) <= self.safe_radius_m:
            return float("inf")
        lo, hi = floor_alt, ceiling
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if design_drift(mid) <= self.safe_radius_m:
                lo = mid
            else:
                hi = mid
        return lo


def run_self_tests() -> None:
    """Sanity checks; raises AssertionError on failure."""
    close = lambda a, b, tol=1e-6: abs(a - b) <= tol

    calc = ParachuteDriftCalculator()
    r = calc.calculate_descent(350, 0, 5.0, 135)
    assert close(r["fall_duration_s"], 350 / 5.2)
    assert close(r["total_drift_m"], 5.0 * 350 / 5.2)
    assert r["target_x"] < 0 < r["target_y"]
    assert r["drift_compass"] == "NW" and r["wind_from_compass"] == "SE"
    assert close(math.hypot(r["target_x"], r["target_y"]), r["total_drift_m"])
    assert r["is_compliant"]
    assert not calc.calculate_descent(350, 0, 10.0, 135)["is_compliant"]

    r = calc.calculate_descent(100, 0, 4.0, 270)
    assert r["target_x"] > 0 and abs(r["target_y"]) < 1e-9
    for d in (0, 360):
        r = calc.calculate_descent(100, 0, 4.0, d)
        assert r["target_y"] < 0 and abs(r["target_x"]) < 1e-9

    assert close(calc.calculate_descent(450, 100, 5.0, 90)["effective_alt_m"], 350)

    m = MAIN_DEPLOY_500FT_AGL_M
    dual = ParachuteDriftCalculator(drogue_descent_ms=25.0, main_deploy_agl_m=m)
    rd = dual.calculate_descent(300, 0, 5.0, 135)
    expected_t = (300 - m) / 25.0 + m / 5.2
    assert close(rd["fall_duration_s"], expected_t, 1e-6)
    assert close(rd["total_drift_m"], 5.0 * expected_t, 1e-6)
    low = dual.calculate_descent(100, 0, 5.0, 135)
    assert close(low["fall_duration_s"], 100 / 5.2)

    for w in (5.0, 10.0):
        t_total = 400.0 / w
        expected_alt = m + (t_total - m / 5.2) * 25.0
        assert close(dual.max_compliant_altitude_agl(w), expected_alt, 0.01)
    assert ParachuteDriftCalculator(
        drogue_descent_ms=25.0, main_deploy_agl_m=m
    ).max_compliant_altitude_agl(0.0) == float("inf")

    strict = ParachuteDriftCalculator(safety_factor=1.3)
    assert not strict.calculate_descent(350, 0, 5.0, 135)["is_compliant"]
    sheared = ParachuteDriftCalculator(shear_exponent=0.14)
    assert sheared.calculate_descent(350, 0, 5.0, 135)["total_drift_m"] > 5.0 * 350 / 5.2

    bad_calls = [
        lambda: calc.calculate_descent(350, 0, -1, 90),
        lambda: calc.calculate_descent(350, 0, 5, 400),
        lambda: calc.calculate_descent(100, 200, 5, 90),
        lambda: ParachuteDriftCalculator(drogue_descent_ms=25.0),
        lambda: ParachuteDriftCalculator(drogue_descent_ms=3.0, main_deploy_agl_m=100.0),
        lambda: ParachuteDriftCalculator(safe_radius_m=0),
    ]
    for call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("Expected ValueError")


def _fmt_time(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m {secs:02d}s"


if __name__ == "__main__":
    run_self_tests()
    print("Self-tests passed successfully!\n")

    # ------------------------------------------------------------------ #
    # MISSION SCENARIO (SEDS Sri Lanka Grand Prix - Task 020)
    # ------------------------------------------------------------------ #
    GROUND_ELEVATION_M = 0.0
    APOGEE_AGL_M = 300.0        # Flight-sim target apogee AGL
    DROGUE_DESCENT_MS = 25.0    # Drogue descent rate (Fruity Chutes heritage)
    MAIN_DESCENT_MS = 5.2       # Interceptor Design Report, Section 3
    WIND_DIR_FROM_DEG = 135.0   # Wind from South-East (SE)
    WIND_CASES_MS = (5.0, 10.0) # Nominal wind and gust cases

    calc = ParachuteDriftCalculator(
        safe_radius_m=400.0,
        baseline_descent_ms=MAIN_DESCENT_MS,
        drogue_descent_ms=DROGUE_DESCENT_MS,
        main_deploy_agl_m=MAIN_DEPLOY_500FT_AGL_M,
    )

    print("=" * 66)
    print("SEDS MISSION 05: PARACHUTE DRIFT & LZ RADIUS SUMMARY")
    print("Candidate: Ahmad Salman (SEDS-TASK-RCK-SRILANKA-020)")
    print("=" * 66)
    print("Assumptions (dual-deploy profile)")
    print(f"  Apogee / drogue deploy : {APOGEE_AGL_M:.1f} m AGL")
    print(f"  Drogue descent speed   : {DROGUE_DESCENT_MS:.1f} m/s")
    print(f"  Main deploy altitude   : {MAIN_DEPLOY_500FT_AGL_M:.1f} m AGL (500 ft)")
    print(f"  Main descent speed     : {MAIN_DESCENT_MS:.1f} m/s")
    print(f"  Wind from              : {WIND_DIR_FROM_DEG:.0f}° {compass_point(WIND_DIR_FROM_DEG)}")
    print("-" * 66)

    all_pass = True
    for wind in WIND_CASES_MS:
        res = calc.calculate_descent(
            deployment_alt_m=GROUND_ELEVATION_M + APOGEE_AGL_M,
            ground_elevation_m=GROUND_ELEVATION_M,
            wind_speed_ms=wind,
            wind_dir_deg=WIND_DIR_FROM_DEG,
        )
        limit = calc.max_compliant_altitude_agl(wind)
        limit_txt = "none (main-only descent too long)" if limit is None else f"{limit:.0f} m AGL"
        all_pass &= res["is_compliant"]

        print(f"Wind {wind:.0f} m/s")
        print(f"  Fall Duration        : {_fmt_time(res['fall_duration_s'])}")
        print(f"  Total Drift          : {res['total_drift_m']:.2f} m")
        print(f"  Landing Bearing      : {res['landing_bearing']}")
        print(f"  Target Offset (E,N)  : ({res['target_x']:.2f} m, {res['target_y']:.2f} m)")
        print(f"  Margin to Limit      : {res['margin_m']:.2f} m")
        print(f"  Max apogee for 400 m : {limit_txt}")
        print(f"  400 m Compliance     : {'PASSED' if res['is_compliant'] else 'FAILED'}")
        print("-" * 66)

    print(f"OVERALL: {'PASSED' if all_pass else 'FAILED'} (all wind cases within 400 m)")
    print("=" * 66)