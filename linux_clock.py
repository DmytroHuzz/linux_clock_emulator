"""
Linux-like clock interaction emulator (conceptual).

Goals:
- Use Linux-ish names and call shapes:
    clock_gettime(CLOCK_*)
    clock_settime(CLOCK_REALTIME, ...)
- Model the typical Linux mental model:
    monotonic is accumulated from a clocksource counter (e.g., TSC)
    realtime = monotonic + wall_offset
    boottime = monotonic + suspend_accum
- Show suspend/resume effect:
    MONOTONIC pauses during suspend
    BOOTTIME includes suspended time via suspend_accum

This is NOT an accurate kernel reimplementation; it's a readable model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

NS_PER_SEC = 1_000_000_000


# -----------------------------
# Linux-like clock ids
# -----------------------------

CLOCK_REALTIME = 0
CLOCK_MONOTONIC = 1
CLOCK_BOOTTIME = 7


# -----------------------------
# Linux-ish timespec
# -----------------------------

# Why does timespec exist instead of just using plain integers?
#
# In Linux, time is never passed around as a single number.  The kernel and
# user-space communicate through `struct timespec { time_t tv_sec; long tv_nsec; }`,
# a pair of (seconds, nanoseconds).  This is the type returned by clock_gettime()
# and accepted by clock_settime() — it is the actual API contract.
#
# We keep this class so the code mirrors the real interface you would see when
# programming against POSIX clocks in C.  Internally the model reasons in plain
# nanoseconds (easier to add/compare), so from_ns() and to_ns() convert between
# the two representations.


@dataclass(frozen=True)
class timespec:
    """Mirrors Linux `struct timespec` — the currency of clock_gettime / clock_settime."""

    tv_sec: int  # whole seconds
    tv_nsec: int  # remaining nanoseconds [0, 999_999_999]

    def __post_init__(self) -> None:
        if self.tv_nsec < 0 or self.tv_nsec >= NS_PER_SEC:
            raise ValueError("tv_nsec must be in [0, 1e9)")

    @staticmethod
    def from_ns(ns: int) -> "timespec":
        """Convenience: single nanosecond value -> split (sec, nsec)."""
        return timespec(tv_sec=ns // NS_PER_SEC, tv_nsec=ns % NS_PER_SEC)

    def to_ns(self) -> int:
        """Convenience: split (sec, nsec) -> single nanosecond value."""
        return self.tv_sec * NS_PER_SEC + self.tv_nsec


def ns_to_iso8601_utc(ns_since_epoch: int) -> str:
    dt = datetime.fromtimestamp(ns_since_epoch / NS_PER_SEC, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


# -----------------------------
# Hardware-ish components
# -----------------------------


@dataclass
class HardwareClockRTC:
    """
    Battery-backed RTC / persistent clock.
    Keeps wall time across power-off.  Stored as Unix epoch ns.
    """

    epoch_ns: int

    def read_ns(self) -> int:
        return self.epoch_ns

    def advance(self, delta_ns: int) -> None:
        self.epoch_ns += delta_ns


@dataclass
class CPUClockTSC:
    """
    TSC-like 64-bit counter.
    Increments only while the CPU is running.
    """

    hz: int
    ticks: int = 0

    def read(self) -> int:
        return self.ticks

    def advance(self, delta_ns: int) -> None:
        delta_ticks = (self.hz * delta_ns) // NS_PER_SEC
        self.ticks += delta_ticks


# -----------------------------
# Kernel timekeeper (Linux-ish API surface)
# -----------------------------


@dataclass
class KernelTimekeeper:
    """
    Linux-like kernel timekeeping model.

    State:
    - tsc_last: last TSC snapshot
    - mult/shift: calibrated ticks->ns conversion (fixed-point)
    - mono_ns: monotonic time since boot (ns), excludes suspend
    - suspend_ns: accumulated suspended time (ns)
    - wall_off_ns: wall-clock offset so realtime = monotonic + wall_off

    Key properties:
    - clock_settime affects CLOCK_REALTIME by changing wall_off_ns
    - CLOCK_MONOTONIC is never stepped; it advances by accumulating deltas
    - CLOCK_BOOTTIME = CLOCK_MONOTONIC + suspend_ns
    """

    tsc: CPUClockTSC
    rtc: HardwareClockRTC

    mult: int
    shift: int

    tsc_last: int = 0
    mono_ns: int = 0
    suspend_ns: int = 0
    wall_off_ns: int = 0

    _rtc_at_suspend: int | None = None

    # ---- boot / initialization ----

    def boot_init_timekeeping(self) -> None:
        """
        Early boot: read RTC to seed wall time, snapshot TSC,
        set monotonic = 0, compute wall offset.
        """
        self.tsc_last = self.tsc.read()
        self.mono_ns = 0
        self.suspend_ns = 0
        self.wall_off_ns = self.rtc.read_ns()  # mono is 0 at boot

    # ---- core internal mechanics ----

    def _ticks_to_ns(self, delta_ticks: int) -> int:
        return (delta_ticks * self.mult) >> self.shift

    def _update_timekeeping(self) -> None:
        """Accumulate TSC deltas into monotonic time."""
        tsc_now = self.tsc.read()
        delta_ticks = tsc_now - self.tsc_last
        self.tsc_last = tsc_now
        self.mono_ns += self._ticks_to_ns(delta_ticks)

    # ---- Linux-like API ----

    def clock_gettime(self, clock_id: int) -> timespec:
        self._update_timekeeping()

        if clock_id == CLOCK_MONOTONIC:
            return timespec.from_ns(self.mono_ns)
        if clock_id == CLOCK_BOOTTIME:
            return timespec.from_ns(self.mono_ns + self.suspend_ns)
        if clock_id == CLOCK_REALTIME:
            return timespec.from_ns(self.mono_ns + self.wall_off_ns)

        raise ValueError(f"Unsupported clock_id: {clock_id}")

    def clock_settime(self, clock_id: int, ts: timespec) -> None:
        """Only CLOCK_REALTIME is settable — adjusts wall offset, not monotonic."""
        if clock_id != CLOCK_REALTIME:
            raise PermissionError("Only CLOCK_REALTIME is settable")

        self._update_timekeeping()
        self.wall_off_ns = ts.to_ns() - self.mono_ns

    # ---- power management hooks ----

    def pm_suspend_enter(self) -> None:
        """Flush deltas and snapshot RTC before suspending."""
        self._update_timekeeping()
        self._rtc_at_suspend = self.rtc.read_ns()

    def pm_suspend_exit(self) -> None:
        """
        On resume: compute suspended duration from RTC,
        accumulate into suspend_ns, and re-sync wall offset.
        """
        if self._rtc_at_suspend is None:
            raise RuntimeError("Resume without matching suspend")

        rtc_resume = self.rtc.read_ns()
        sleep_delta = rtc_resume - self._rtc_at_suspend

        self.suspend_ns += sleep_delta
        # Keep realtime aligned with RTC (mono didn't advance during suspend)
        self.wall_off_ns = rtc_resume - self.mono_ns

        self._rtc_at_suspend = None
        self.tsc_last = self.tsc.read()


# --------------------------------------------------------
# Demo — simulates a full machine lifecycle
#
# The flow mirrors what happens on a real Linux box:
#   1. Hardware exists (RTC + TSC) before the kernel starts
#   2. Kernel boots and initialises timekeeping from hardware
#   3. Normal running: TSC ticks, clocks advance
#   4. Suspend/resume: TSC stops but RTC keeps ticking
#   5. User-space steps wall clock via clock_settime
# --------------------------------------------------------


def main() -> None:
    # -- helpers --

    def advance(ms: int, cpu_running: bool = True) -> None:
        """Simulate real-world time passing."""
        delta_ns = ms * 1_000_000
        rtc.advance(delta_ns)  # RTC always ticks (battery-backed)
        if cpu_running:
            tsc.advance(delta_ns)  # TSC only while CPU is powered

    def show(label: str) -> None:
        rt = k.clock_gettime(CLOCK_REALTIME).to_ns()
        mo = k.clock_gettime(CLOCK_MONOTONIC).to_ns()
        bt = k.clock_gettime(CLOCK_BOOTTIME).to_ns()
        print(f"\n{label}")
        print(f"  REALTIME  : {ns_to_iso8601_utc(rt)}  ({rt} ns)")
        print(f"  MONOTONIC : {mo // 1_000_000} ms")
        print(f"  BOOTTIME  : {bt // 1_000_000} ms")

    # -- 1. Hardware exists before boot --

    start = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)
    rtc_start_ns = int(start.timestamp() * NS_PER_SEC)

    rtc = HardwareClockRTC(epoch_ns=rtc_start_ns)
    tsc = CPUClockTSC(hz=2_500_000_000)

    # -- 2. Kernel boots and calibrates timekeeping --

    # Fixed-point multiplier: delta_ns ≈ delta_ticks * (1e9 / hz)
    shift = 32
    mult = (NS_PER_SEC << shift) // tsc.hz

    k = KernelTimekeeper(tsc=tsc, rtc=rtc, mult=mult, shift=shift)
    k.boot_init_timekeeping()
    show("After boot")

    # -- 3. Normal running: 250 ms of CPU time --

    advance(250)
    show("After 250ms running")

    # -- 4. Suspend / resume: 2 s real-world, CPU paused --

    k.pm_suspend_enter()
    advance(2000, cpu_running=False)
    k.pm_suspend_exit()
    show("After 2s suspend/resume (MONOTONIC paused, BOOTTIME advanced)")

    # -- 5. User-space steps wall clock forward via clock_settime --

    current_rt = k.clock_gettime(CLOCK_REALTIME).to_ns()
    k.clock_settime(CLOCK_REALTIME, timespec.from_ns(current_rt + 5 * NS_PER_SEC))
    show("After clock_settime(+5s) (REALTIME jumped; MONOTONIC unchanged)")

    # -- 6. Another 100 ms of normal running --

    advance(100)
    show("After 100ms running")


if __name__ == "__main__":
    main()
