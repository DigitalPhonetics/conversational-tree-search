from datetime import datetime, timedelta
from typing import Any, Iterable, Dict
from threading import Thread, RLock
import time

GC_INTERVAL = 300 # garbabe collection interval (in seconds): 300 ~ run GC every 5 minutes
KEEP_DATA_FOR_N_SECONDS = timedelta(seconds=300) # duration to store user data in memory

class _Memory:
    def __init__(self, lock: RLock):
        self.mem = {}
        self.timestamps = {}
        self.lock = lock

    def add_value(self, user_id: str, attribute_name: str, attribute_value: Any):
        """
        Adds / udpates the given value for the provided user and attribute name.
        Will also update the timestamp for this user's data to prevent GC.
        """
        self.lock.acquire()
        if not user_id in self.mem:
            # create storage for new user
            self.mem[user_id] = {}
        # update memory for given user and attribute name
        # also, update last access time
        self.mem[user_id][attribute_name] = attribute_value
        self.timestamps[user_id] = datetime.now()   
        self.lock.release()
    
    def get_value(self, user_id: str, attribute_name: str):
        """
        Returns the stored value for the provided user and attribute.
        If the user or attribute are unknown, will return 'None'.
        Will also update the timestamp for this user's data to prevent GC.
        """
        self.lock.acquire()
        if user_id in self.mem and attribute_name in self.mem[user_id]:
            self.timestamps[user_id] = datetime.now()
            self.lock.release()
            return self.mem[user_id][attribute_name]
        self.lock.release()
        return None

    def delete_values(self, user_id: str):
        """
        Will delete all values for the provided user.
        """
        self.lock.acquire()
        if user_id in self.mem:
            del self.mem[user_id]
        if user_id in self.timestamps:
            del self.timestamps[user_id]
        self.lock.release()

    def gc(self):
        self.lock.acquire()
        # collect expired user data
        now = datetime.now()
        expired_users = [user_id for user_id in self.timestamps if now - self.timestamps[user_id] > KEEP_DATA_FOR_N_SECONDS]
        # delete expired user data
        for user_id in expired_users:
            del self.timestamps[user_id]
            del self.mem[user_id]
        self.lock.release()


class _MemoryPool:
    __instances = {}
    _gc_thread = None
    _locks = {}

    @staticmethod
    def get_instance(cls):
        """
        Singleton: Return a (new) instance of Memory for the given _Service class.
        """
        if _MemoryPool._gc_thread == None:
            # init GC (make GC daemon so it ends on program exit)
            _MemoryPool._gc_thread = Thread(target=_MemoryPool.gc, daemon=True)
            _MemoryPool._gc_thread.start()

        if not type(cls).__name__ in _MemoryPool.__instances:
            # create new shared memory with associated lock
            lock = RLock()
            _MemoryPool._locks[type(cls).__name__] = lock
            _MemoryPool.__instances[type(cls).__name__] = _Memory(lock)
        return _MemoryPool.__instances[type(cls).__name__]
    
    @staticmethod
    def gc():
        while True:
            time.sleep(GC_INTERVAL)    # wait for next GC event
            # sweep & clean memory
            now = datetime.now()
            free_user_ids = []
            for service_type in _MemoryPool.__instances:
                _MemoryPool.__instances[service_type].gc()


class _Service:
    def __init__(self):
        # associate memory
        self._memory = _MemoryPool.get_instance(self)

    def get_state(self, user_id: str, attribute_name: str):
        return self._memory.get_value(user_id=user_id, attribute_name=attribute_name)

    def set_state(self, user_id: str, attribute_name: str, attribute_value: Any):
        self._memory.add_value(user_id=user_id, attribute_name=attribute_name, attribute_value=attribute_value)


