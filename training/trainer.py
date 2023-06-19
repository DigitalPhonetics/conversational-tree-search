
import torch

# TODO observations from results have to be converted to tensors


class Trainer:
    def __init__(self, device, cache, 
                    training, validation, testing, 
                    seed, cudnn_deterministic, 
                    optimizer, model, algorithm,
                    logging, environment,
                    actions, state) -> None:

        # TODO store all initialized modules in a global store
        # TODO setup tree
        # TODO config cache
        # TODO load shared embeddings once
        # TODO setup wandb
        # TODO setup metrics
        # TODO setup saving
        # TODO move model to device
        # TODO train, eval, test loops

        


        print("DEVICE", device)
        print("HELLO")
        print("CACHE", cache)
        print("TRAINING", training)
        