import numpy as np

from core.body import Body
from core.simulation import NBodySimulation

from config import (
    NUMBER_OF_BODIES,
    TOTAL_STEPS,
    TIME_STEP
)
simulation = NBodySimulation(
    bodies,
    dt=TIME_STEP,
    method="verlet",
    gravity_method="barnes_hut",
    theta=0.5
)

def create_universe():

    bodies = []

    for i in range(NUMBER_OF_BODIES):

        position = np.random.uniform(
            -1e9, 1e9, size=3
        )

        velocity = np.random.uniform(
            -100, 100, size=3
        )

        mass = np.random.uniform(
            1e18, 1e20
        )

        radius = np.random.uniform(
            1e5, 1e6
        )

        bodies.append(
            Body(
                mass=mass,
                position=position,
                velocity=velocity,
                radius=radius,
                name=f"Body-{i}"
            )
        )

    return bodies


def main():

    print("=" * 60)
    print("ADVANCED N-BODY GRAVITATIONAL SIMULATION")
    print("=" * 60)

    bodies = create_universe()

    simulation = NBodySimulation(
        bodies,
        dt=TIME_STEP,
        method="verlet"
    )

    initial_energy = simulation.total_energy()

    print(f"Initial bodies: {len(bodies)}")
    print(f"Initial energy: {initial_energy:.6e}")

    for step in range(TOTAL_STEPS):

        simulation.step()

        if step % 100 == 0:

            energy = simulation.total_energy()

            print(
                f"Step: {step:5d} | "
                f"Bodies: {len(simulation.bodies):5d} | "
                f"Energy: {energy:.6e}"
            )

    print("\nSIMULATION COMPLETED")

    print(f"Final bodies: {len(simulation.bodies)}")
    print(f"Simulation time: {simulation.time:.2f}")

    final_energy = simulation.total_energy()

    print(f"Final energy: {final_energy:.6e}")


if __name__ == "__main__":

    main()