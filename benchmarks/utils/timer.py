# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import time
from datetime import timedelta

class Timer:

    def __init__(self, parent_timer=None, max_time=None):
        self.parent = parent_timer
        self.max_time = max_time
        self.elapsed = 0
        self.started = False
        self.paused  = False
        self.stopped = False

    def add(self, dt):
        if self.started and not self.paused:
            raise Exception('Cannot add while timer is running or stopped')
        if self.stopped:
            raise Exception('Cannot add when timer has been stopped')
        if not self.started:
            self.started = True
            self.paused  = True
        self.elapsed += dt
        if self.parent:
            self.parent.add(dt)

    def start(self):
        t0 = time.perf_counter()
        self._start(t0)

    def pause(self):
        t = time.perf_counter()
        self._pause(t)

    def resume(self):
        t0 = time.perf_counter()
        self._resume(t0)

    def stop(self):
        t = time.perf_counter()
        self._stop(t)

    def _stop(self, t):
        if self.stopped:
            raise Exception('Timer has already been stopped')
        if not self.started:
            raise Exception('Timer has not been started')
        self._pause(t, ok_if_already_paused=True)
        self.stopped = True

    def get_elapsed(self, pretty=False):
        if not self.started:
            raise Exception('Timer has not been started')
        if self.stopped or self.paused:
            elapsed = self.elapsed
        else:
            elapsed = time.perf_counter() - self.t0
        if pretty:
            return self.human_duration(elapsed)
        else:
            return elapsed

    def is_over_limit(self):
        if self.max_time is None:
            raise Exception('No max_time has been set')
        return self.get_elapsed() > self.max_time

    def _start(self, t0, propagate=True):
        if self.stopped:
            raise Exception('Timer has been stopped')
        if self.started:
            raise Exception('Timer has already started')
        self.t0 = t0
        self.started = True
        if propagate and self.parent:
            self.parent._resume(t0)

    def _pause(self, t, ok_if_already_paused=False):
        if self.stopped:
            raise Exception('Timer has been stopped')
        if not self.started:
            raise Exception('Timer has not been started')
        if self.paused:
            if not ok_if_already_paused:
                raise Exception('Timer has already been paused')
        else:
            self.elapsed += t - self.t0
            self.paused = True
        if self.parent:
            self.parent._pause(t, ok_if_already_paused)

    def _resume(self, t0):
        if self.stopped:
            raise Exception('Timer has been stopped')
        if not self.started:
            self._start(t0, propagate=False)
        elif not self.paused:
            raise Exception('Timer has not been paused')
        self.t0 = t0
        self.paused = False
        if self.parent:
            self.parent._resume(t0)


    @staticmethod
    def human_duration(seconds: float, s_prec: int = 2) -> str:
        td = timedelta(seconds=seconds)
        days = td.days
        rem = td.seconds + td.microseconds / 1e6
        h, rem = divmod(rem, 3600)
        m, s = divmod(rem, 60)
        parts = []
        if days: parts.append(f"{days}d")
        if h:    parts.append(f"{int(h)}h")
        if m:    parts.append(f"{int(m)}m")
        parts.append(f"{s:.{s_prec}f}s")
        return " ".join(parts)

