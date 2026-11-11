from integrators.euler import EulerIntegrator
from integrators.semi_implicit_euler import SemiImplicitEuler
from integrators.velocity_verlet import VelocityVerlet


class IntegrationEngine:

    def __init__(self, gravity_engine, method="verlet"):

        self.gravity = gravity_engine
        self.method = None
        self.integrator = None

        self.set_method(method)

    def set_method(self, method):

        methods = {
            "euler": EulerIntegrator,
            "semi_implicit": SemiImplicitEuler,
            "verlet": VelocityVerlet
        }

        if method not in methods:
            raise ValueError(
                f"Unknown integration method: {method}"
            )

        self.method = method

        self.integrator = methods[method](
            self.gravity
        )

    def step(self, bodies, dt):

        return self.integrator.step(bodies, dt)

    def get_method(self):

        return self.method