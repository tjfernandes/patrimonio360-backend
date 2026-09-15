"""Opt-in bounded GPU execution and per-conversation serialization.

Cancellation cannot stop a CUDA kernel. The executor retains ownership of a
running job until its thread actually finishes, even if its HTTP caller leaves.
"""
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache,wraps
import threading
from weakref import WeakValueDictionary

class InferenceBusyError(RuntimeError):
    pass

class GPUExecutor:
    def __init__(self,capacity: int=8):
        if capacity<1:raise ValueError('capacity must be positive')
        self._pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='p360-gpu')
        self._capacity=threading.BoundedSemaphore(capacity)

    async def run(self,fn,*args,**kwargs):
        if not self._capacity.acquire(blocking=False):
            raise InferenceBusyError('GPU queue is full; retry later')
        try:future=self._pool.submit(fn,*args,**kwargs)
        except BaseException:
            self._capacity.release();raise
        future.add_done_callback(lambda _:self._capacity.release())
        wrapped=asyncio.wrap_future(future)
        # Retrieve eventual errors even when the awaiting request is cancelled.
        wrapped.add_done_callback(lambda done:done.exception() if not done.cancelled() else None)
        try:return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            future.cancel() # cancels queued jobs; running jobs keep their slot
            raise

    def close(self):self._pool.shutdown(wait=True,cancel_futures=True)

@lru_cache(maxsize=1)
def get_gpu_executor():return GPUExecutor()

async def run_gpu(settings,fn,*args,**kwargs):
    if getattr(settings,'CHAT_BOUNDED_GPU_EXECUTION',False):
        return await get_gpu_executor().run(fn,*args,**kwargs)
    return await asyncio.to_thread(fn,*args,**kwargs)

_conversation_locks: WeakValueDictionary = WeakValueDictionary()

def serialized_conversation(method):
    @wraps(method)
    async def wrapped(self,payload,*args,**kwargs):
        if not getattr(self.settings,'CHAT_SERIALIZE_CONVERSATIONS',False):
            return await method(self,payload,*args,**kwargs)
        conversation_id=getattr(payload,'conversation_id',None)
        if not conversation_id:return await method(self,payload,*args,**kwargs)
        # Loop-local: service tests and worker restarts can use separate loops.
        key=(id(asyncio.get_running_loop()),conversation_id)
        lock=_conversation_locks.get(key)
        if lock is None:lock=asyncio.Lock();_conversation_locks[key]=lock
        async with lock:return await method(self,payload,*args,**kwargs)
    return wrapped
