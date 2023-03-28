from typing import Any
from config import INSTANCES, InstanceType
import redisai as rai

def requires_caching(obj):
    return hasattr(obj, '__dict__') and "caching" in obj and obj["caching"] == True

class Cache:
    def __init__(self, active: bool, host: str, port: int) -> None:
        global INSTANCES
        INSTANCES[InstanceType.CACHE] = self

        candidate_keys = filter(lambda x: requires_caching(INSTANCES[InstanceType.CONFIG].experiment.state[x]), INSTANCES[InstanceType.CONFIG].experiment.state)
        self.connections = {
            input_key: rai.Client(host=host, port=port, db=INSTANCES[InstanceType.CONFIG].experiment.state[input_key].cache_db_index)
            for input_key in candidate_keys
        }
        
        print("CANDS", self.connections)
        # db_indices = list(filter(lambda state_input: "cache_db_index" in state_input, .values()))
        # print("Setup Cache for db indices ", db_indices)


        # print("DB INDEX", .node_text)

    def get(self, state_input_key: str, key: str, value: Any):
        pass

    def set(self, state_input_key: str, key: str):
        pass