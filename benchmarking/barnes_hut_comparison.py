
import time
import numpy as np

from main import create_universe
from core.gravity import GravityEngine

def run_comparison():

    bodies = create_universe()

    print("=" * 65)
    print("DIRECT VS BARNES-HUT PERFORMANCE COMPARISON")
    print("=" * 65)

    print(f"\nNumber of bodies: {len(bodies)}")

    # Direct calculation
    direct_solver = GravityEngine(
        method="direct"
    )

    start = time.perf_counter()

    direct_forces = direct_solver.calculate_forces(
        bodies
    )

    direct_time = time.perf_counter() - start

    # Barnes-Hut calculation
    barnes_solver = GravityEngine(
        method="barnes_hut",
        theta=0.5
    )

    start = time.perf_counter()

    barnes_forces = barnes_solver.calculate_forces(
        bodies
    )

    barnes_time = time.perf_counter() - start

    # Accuracy comparison
    difference = (
        barnes_forces - direct_forces
    )

    error = np.linalg.norm(difference)

    reference = np.linalg.norm(direct_forces)

    relative_error = (
        error / max(reference, 1e-30)
    )

    speedup = (
        direct_time / barnes_time
        if barnes_time > 0 else 0
    )

    print("\nDIRECT SOLVER")
    print("-" * 40)

    print(f"Execution time: {direct_time:.6f} seconds")

    print("\nBARNES-HUT SOLVER")
    print("-" * 40)

    print(f"Execution time: {barnes_time:.6f} seconds")

    print(
        f"Octree nodes: "
        f"{barnes_solver.barnes_hut.tree.count_nodes()}"
    )

    print("\nACCURACY ANALYSIS")
    print("-" * 40)

    print(f"Absolute force error: {error:.6e}")

    print(f"Relative force error: {relative_error:.8e}")

    print("\nPERFORMANCE ANALYSIS")
    print("-" * 40)

    print(f"Speedup: {speedup:.4f}x")

    if barnes_time < direct_time:
        print("Barnes-Hut was faster for this test.")
    else:
        print("Direct Solver was faster for this test.")

    print("\nCOMPARISON COMPLETED")

if __name__ == "__main__":
    run_comparison()