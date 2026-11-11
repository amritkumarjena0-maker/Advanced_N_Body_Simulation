
import numpy as np

from algorithms.octree import Octree
from config import GRAVITATIONAL_CONSTANT, SOFTENING


class BarnesHutSolver:

    def __init__(
        self,
        theta=0.5,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
        softening=SOFTENING
    ):

        if theta <= 0:
            raise ValueError("Theta must be positive.")

        self.theta = theta
        self.G = gravitational_constant
        self.softening = softening

        self.tree = None

    def build_tree(self, bodies):

        positions = np.array(
            [body.position for body in bodies],
            dtype=np.float64
        )

        masses = np.array(
            [body.mass for body in bodies],
            dtype=np.float64
        )

        self.tree = Octree(
            positions,
            masses,
            capacity=1,
            max_depth=40
        )

        return self.tree

    def _calculate_force(
        self,
        node,
        target_index,
        target_position,
        target_mass
    ):

        if node.mass == 0:
            return np.zeros(3)

        displacement = (
            node.center_of_mass - target_position
        )

        distance_squared = (
            np.dot(displacement, displacement)
            + self.softening ** 2
        )

        distance = np.sqrt(distance_squared)

        if node.is_leaf:

            force = np.zeros(3)

            for index in node.indices:

                if index == target_index:
                    continue

                delta = (
                    self.tree.positions[index]
                    - target_position
                )

                r_squared = (
                    np.dot(delta, delta)
                    + self.softening ** 2
                )

                force += (
                    self.G
                    * target_mass
                    * self.tree.masses[index]
                    * delta
                    / (r_squared ** 1.5)
                )

            return force

        contains_target = np.all(
            np.abs(
                target_position - node.center
            ) <= node.half_size
        )

        node_width = 2.0 * node.half_size

        if (
            not contains_target
            and node_width / distance < self.theta
        ):

            return (
                self.G
                * target_mass
                * node.mass
                * displacement
                / (distance_squared ** 1.5)
            )

        total_force = np.zeros(3)

        for child in node.children:

            if child is not None:

                total_force += self._calculate_force(
                    child,
                    target_index,
                    target_position,
                    target_mass
                )

        return total_force

    def calculate_forces(self, bodies):

        self.build_tree(bodies)

        count = len(bodies)

        forces = np.zeros(
            (count, 3),
            dtype=np.float64
        )

        for index, body in enumerate(bodies):

            forces[index] = self._calculate_force(
                self.tree.root,
                index,
                body.position,
                body.mass
            )

        return forces

    def calculate_accelerations(self, bodies):

        forces = self.calculate_forces(bodies)

        masses = np.array(
            [body.mass for body in bodies]
        )

        return forces / masses[:, None]
