
import numpy as np

from config import (
    GRAVITATIONAL_CONSTANT,
    SOFTENING,
    CHUNK_SIZE
)

from algorithms.barnes_hut import BarnesHutSolver

class GravityEngine:

    def __init__(
        self,
        method="direct",
        theta=0.5,
        gravitational_constant=GRAVITATIONAL_CONSTANT
    ):

        self.G = gravitational_constant
        self.method = None

        self.barnes_hut = BarnesHutSolver(
            theta=theta,
            gravitational_constant=self.G,
            softening=SOFTENING
        )

        self.set_method(method)

    def set_method(self, method):

        if method not in ("direct", "barnes_hut"):

            raise ValueError(
                f"Unknown gravity method: {method}"
            )

        self.method = method

    def calculate_forces(self, bodies):

        if self.method == "barnes_hut":

            return self.barnes_hut.calculate_forces(
                bodies
            )

        count = len(bodies)

        positions = np.array(
            [body.position for body in bodies]
        )

        masses = np.array(
            [body.mass for body in bodies]
        )

        forces = np.zeros(
            (count, 3),
            dtype=np.float64
        )

        for start in range(0, count, CHUNK_SIZE):

            end = min(start + CHUNK_SIZE, count)

            displacement = (
                positions[None, :, :]
                - positions[start:end, None, :]
            )

            distance_squared = (
                np.sum(displacement ** 2, axis=2)
                + SOFTENING ** 2
            )

            distance_squared[
                np.arange(end - start),
                np.arange(start, end)
            ] = np.inf

            inverse_cube = distance_squared ** -1.5

            force = (
                self.G
                * masses[start:end, None]
                * masses[None, :]
                * inverse_cube[:, :, None]
                * displacement
            )

            forces[start:end] = np.sum(
                force, axis=1
            )

        return forces

    def calculate_accelerations(self, bodies):

        if not bodies:
            return np.empty((0, 3))

        forces = self.calculate_forces(bodies)

        masses = np.array(
            [body.mass for body in bodies]
        )

        return forces / masses[:, None]