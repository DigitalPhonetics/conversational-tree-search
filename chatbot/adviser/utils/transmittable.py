from typing import Iterable, Union, Mapping, Any

class Transmittable:
    @staticmethod
    def deserialize(obj: Union[Iterable, Mapping, Any]) -> 'Transmittable':
        """
        Provide a means to deserialize a json object parsed from your `to_json()` method
        data structure to an instance of your class.
        Inverse Method to `to_json()`.
        """
        raise NotImplementedError

    def serialize(self):
        """
        Provide a means to serialize an object instance of your class into a json-serializable
        object (e.g. dict, list).
        Inverse method to `from_json`
        """
        raise NotImplementedError
