class EulerIntegrator:

    def __init__(self, gravity_engine):
        self.gravity = gravity_engine

    def step(self, bodies, dt):

        accelerations = (
            self.gravity.calculate_accelerations(bodies)
        )

        for body, acceleration in zip(
            bodies, accelerations
        ):

            old_velocity = body.velocity.copy()

            body.position += old_velocity * dt

            body.velocity += acceleration * dt

        return bodies