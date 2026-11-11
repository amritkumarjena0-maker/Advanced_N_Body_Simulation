
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.body import Body
from core.simulation import NBodySimulation


class IntegrationComparison:

    def __init__(self, body_count=50, steps=500, dt=1000.0):
        self.body_count = body_count
        self.steps = steps
        self.dt = dt

        self.methods = [
            "euler",
            "semi_implicit",
            "verlet"
        ]

    def create_bodies(self):

        import numpy as np

        rng = np.random.default_rng(42)
        bodies = []

        for i in range(self.body_count):

            body = Body(
                mass=rng.uniform(1e20, 5e20),
                position=rng.uniform(-1e9, 1e9, 3),
                velocity=rng.uniform(-100, 100, 3),
                radius=1e5,
                name=f"Body_{i + 1}"
            )

            bodies.append(body)

        return bodies

    def run_test(self, method):

        simulation = NBodySimulation(
            bodies=self.create_bodies(),
            dt=self.dt,
            method=method
        )

        initial_energy = simulation.total_energy()

        start = time.perf_counter()

        simulation.run(self.steps)

        elapsed = time.perf_counter() - start

        final_energy = simulation.total_energy()

        energy_error = (
            abs(final_energy - initial_energy)
            / max(abs(initial_energy), 1e-30)
        ) * 100

        return {
            "method": method,
            "execution_time": elapsed,
            "energy_error": energy_error
        }

    def compare(self):

        results = []

        print("\n" + "=" * 55)
        print("       N-BODY INTEGRATION COMPARISON")
        print("=" * 55)

        for method in self.methods:

            try:

                result = self.run_test(method)
                results.append(result)

                print(f"\nAlgorithm: {method}")
                print(
                    f"Execution Time: "
                    f"{result['execution_time']:.6f} seconds"
                )
                print(
                    f"Energy Error: "
                    f"{result['energy_error']:.8f}%"
                )

            except Exception as error:

                print(f"\nError in {method}: {error}")

        return results

    def plot_results(self, results):

        if not results:
            print("No successful results to plot.")
            return

        output = PROJECT_ROOT / "output"
        output.mkdir(exist_ok=True)

        methods = [r["method"] for r in results]

        times = [r["execution_time"] for r in results]
        errors = [r["energy_error"] for r in results]

        plt.figure(figsize=(9, 5))
        plt.bar(methods, times)
        plt.title("Integration Execution Time")
        plt.xlabel("Algorithm")
        plt.ylabel("Seconds")
        plt.tight_layout()
        plt.savefig(output / "integration_execution_time.png")
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.bar(methods, errors)
        plt.title("Integration Energy Error")
        plt.xlabel("Algorithm")
        plt.ylabel("Energy Error (%)")
        plt.tight_layout()
        plt.savefig(output / "integration_energy_error.png")
        plt.close()

        print("\nBenchmark graphs saved in output/.")


if __name__ == "__main__":

    benchmark = IntegrationComparison()

    results = benchmark.compare()

    benchmark.plot_results(results)
