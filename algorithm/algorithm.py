
class Algorithm:
    def run_single_timestep(engine, batch):
        raise NotImplementedError

    def validation_step(engine, batch):
        raise NotImplementedError