
from core.gravity import GravityEngine
from core.collision import CollisionEngine

from integrators.integration_engine import (
    IntegrationEngine
)

class NBodySimulation:

    def __init__(
        self,
        bodies,
        dt=1000.0,
        method="verlet",
        gravity_method="direct",
        theta=0.5
    ):

        self.bodies = bodies
        self.dt = dt

        self.gravity = GravityEngine(
            method=gravity_method,
            theta=theta
        )

        self.collision = CollisionEngine()

        self.integrator = IntegrationEngine(
            self.gravity,
            method
        )

        self.time = 0.0
        self.iteration = 0

    def set_integrator(self, method):

        self.integrator.set_method(method)

    def set_gravity_method(self, method):

        self.gravity.set_method(method)

    def step(self):

        self.integrator.step(
            self.bodies,
            self.dt
        )

        self.bodies = self.collision.merge_bodies(
            self.bodies
        )

        self.time += self.dt
        self.iteration += 1

        return self.bodies

    def run(self, steps):

        for _ in range(steps):
            self.step()

        return self.bodies

    def total_energy(self):

        kinetic = sum(
            body.kinetic_energy
            for body in self.bodies
        )

        potential = 0.0

        for i in range(len(self.bodies)):

            for j in range(i + 1, len(self.bodies)):

                displacement = (
                    self.bodies[i].position
                    - self.bodies[j].position
                )

                distance = (
                    displacement @ displacement
                    + self.gravity.barnes_hut.softening ** 2
                ) ** 0.5

                potential -= (
                    self.gravity.G
                    * self.bodies[i].mass
                    * self.bodies[j].mass
                    / distance
                )

        return kinetic + potential