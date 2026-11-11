from dataclasses import dataclass
import numpy as np


@dataclass
class Body:

    mass: float
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    name: str = "Body"

    def __post_init__(self):

        self.position = np.asarray(
            self.position, dtype=np.float64
        )

        self.velocity = np.asarray(
            self.velocity, dtype=np.float64
        )

        if self.mass <= 0:
            raise ValueError("Mass must be positive.")

        if self.radius <= 0:
            raise ValueError("Radius must be positive.")

        if self.position.shape != (3,):
            raise ValueError("Position must have 3 coordinates.")

        if self.velocity.shape != (3,):
            raise ValueError("Velocity must have 3 coordinates.")

    @property
    def momentum(self):

        return self.mass * self.velocity

    @property
    def kinetic_energy(self):

        return 0.5 * self.mass * np.dot(
            self.velocity, self.velocity
        )