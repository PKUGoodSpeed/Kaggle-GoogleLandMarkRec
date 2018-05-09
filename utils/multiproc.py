import multiprocessing
import multiprocessing.pool

class NoDeamonProcess(multiprocessing.Process):
    def _get_deamon(self):
        return False
    def _set_daemon(self):
        pass
    daemon = property(_get_deamon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    process = NoDeamonProcess
