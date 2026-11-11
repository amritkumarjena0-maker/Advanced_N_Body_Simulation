class VelocityVerlet:

    def __init__(self, gravity_engine):
        self.gravity = gravity_engine

    def step(self, bodies, dt):

        old_accelerations = (
            self.gravity.calculate_accelerations(bodies)
        )

        for body, acceleration in zip(
            bodies, old_accelerations
        ):

            body.position += (
                body.velocity * dt
                + 0.5 * acceleration * dt ** 2
            )

        new_accelerations = (
            self.gravity.calculate_accelerations(bodies)
        )

        for body, old_a, new_a in zip(
            bodies,
            old_accelerations,
            new_accelerations
        ):

            body.velocity += (
                0.5 * (old_a + new_a) * dt
            )

        return bodies