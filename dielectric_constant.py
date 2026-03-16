"""
Dielectric Constant Calculator for Parallel Plate Capacitors

Given voltage (V) and charge (Q) measurements from a parallel plate capacitor,
this program fits the data to calculate capacitance, then derives the dielectric
constant (relative permittivity) of the material between the plates.

Physics:
    C = Q / V           (capacitance from charge and voltage)
    C = ε₀ * εᵣ * A / d (capacitance of parallel plate capacitor)
    => εᵣ = C * d / (ε₀ * A)
"""

import numpy as np

# Physical constants
EPSILON_0 = 8.854187817e-12  # Vacuum permittivity (F/m)


def calculate_capacitance(voltages, charges):
    """
    Fit a line to Q vs V data to extract capacitance.
    C is the slope of Q = C * V.

    Returns:
        C       - capacitance in Farads
        C_err   - standard error of C
        r2      - R² goodness of fit
    """
    voltages = np.array(voltages)
    charges = np.array(charges)

    # Linear least-squares fit: Q = C * V (forced through origin)
    C = np.dot(voltages, charges) / np.dot(voltages, voltages)

    # Residuals and R²
    residuals = charges - C * voltages
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((charges - np.mean(charges))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # Standard error of slope
    n = len(voltages)
    s2 = ss_res / max(n - 1, 1)
    C_err = np.sqrt(s2 / np.dot(voltages, voltages))

    return C, C_err, r2


def calculate_dielectric_constant(C, C_err, plate_area_m2, separation_m):
    """
    Derive dielectric constant εᵣ from measured capacitance.

    Returns:
        epsilon_r       - dielectric constant (dimensionless)
        epsilon_r_err   - propagated uncertainty
    """
    epsilon_r = (C * separation_m) / (EPSILON_0 * plate_area_m2)
    epsilon_r_err = (C_err * separation_m) / (EPSILON_0 * plate_area_m2)
    return epsilon_r, epsilon_r_err


def get_measurements():
    """Prompt user to enter voltage/charge measurement pairs."""
    print("\nEnter your (Voltage, Charge) measurement pairs.")
    print("Type 'done' when finished (minimum 2 points required).\n")

    voltages = []
    charges = []
    i = 1

    while True:
        raw = input(f"  Measurement {i} — Voltage (V): ").strip()
        if raw.lower() == "done":
            if len(voltages) < 2:
                print("  Need at least 2 measurements. Keep going.")
                continue
            break
        try:
            v = float(raw)
            q = float(input(f"  Measurement {i} — Charge (C): ").strip())
            voltages.append(v)
            charges.append(q)
            i += 1
        except ValueError:
            print("  Invalid input — enter a number or 'done'.")

    return voltages, charges


def main():
    print("=" * 55)
    print("   Parallel Plate Capacitor — Dielectric Constant")
    print("=" * 55)

    # --- Plate geometry ---
    print("\nPlate Geometry")
    print("-" * 30)
    try:
        area = float(input("  Plate area (m²): "))
        separation = float(input("  Plate separation (m): "))
    except ValueError:
        print("Invalid geometry input. Exiting.")
        return

    # --- Measurements ---
    voltages, charges = get_measurements()

    # --- Calculate ---
    C, C_err, r2 = calculate_capacitance(voltages, charges)
    epsilon_r, epsilon_r_err = calculate_dielectric_constant(C, C_err, area, separation)

    # --- Results ---
    print("\n" + "=" * 55)
    print("Results")
    print("-" * 55)
    print(f"  Number of measurements : {len(voltages)}")
    print(f"  Capacitance (C)        : {C:.4e} ± {C_err:.2e} F")
    print(f"  R² (goodness of fit)   : {r2:.6f}")
    print(f"  Dielectric constant εᵣ : {epsilon_r:.4f} ± {epsilon_r_err:.4f}")
    print("=" * 55)

    if r2 < 0.99:
        print(f"\n  Warning: R² = {r2:.4f} — check your measurements for outliers.")


if __name__ == "__main__":
    main()
