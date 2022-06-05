from TT_ALS import test

__all__ = [
        "papi_high",
        "papi_low",
        "events",
        "consts",
        "exceptions",
        ]
# This file is automatically generated by the
# 'tools/generate_papi_events.py' script.
# DO NOT EDIT!


# flake8: noqa


PAPI_PRESET_MASK = 0x80000000 if not 0x80000000 & 0x80000000 else 0x80000000 | ~0x7FFFFFFF

PAPI_NATIVE_MASK = 0x40000000 if not 0x40000000 & 0x80000000 else 0x40000000 | ~0x7FFFFFFF

PAPI_UE_MASK = 0xC0000000 if not 0xC0000000 & 0x80000000 else 0xC0000000 | ~0x7FFFFFFF

PAPI_PRESET_AND_MASK = 0x7FFFFFFF

#: this masks just the native bit
PAPI_NATIVE_AND_MASK = 0xBFFFFFFF

PAPI_UE_AND_MASK = 0x3FFFFFFF

#: The maxmimum number of preset events
PAPI_MAX_PRESET_EVENTS = 128

#: The maxmimum number of user defined events
PAPI_MAX_USER_EVENTS = 50

#: The maximum length of the operation string for user defined events
USER_EVENT_OPERATION_LEN = 512

#: Level 1 data cache misses
PAPI_L1_DCM = 0x00 | PAPI_PRESET_MASK

#: Level 1 instruction cache misses
PAPI_L1_ICM = 0x01 | PAPI_PRESET_MASK

#: Level 2 data cache misses
PAPI_L2_DCM = 0x02 | PAPI_PRESET_MASK

#: Level 2 instruction cache misses
PAPI_L2_ICM = 0x03 | PAPI_PRESET_MASK

#: Level 3 data cache misses
PAPI_L3_DCM = 0x04 | PAPI_PRESET_MASK

#: Level 3 instruction cache misses
PAPI_L3_ICM = 0x05 | PAPI_PRESET_MASK

#: Level 1 total cache misses
PAPI_L1_TCM = 0x06 | PAPI_PRESET_MASK

#: Level 2 total cache misses
PAPI_L2_TCM = 0x07 | PAPI_PRESET_MASK

#: Level 3 total cache misses
PAPI_L3_TCM = 0x08 | PAPI_PRESET_MASK

#: Snoops
PAPI_CA_SNP = 0x09 | PAPI_PRESET_MASK

#: Request for shared cache line (SMP)
PAPI_CA_SHR = 0x0A | PAPI_PRESET_MASK

#: Request for clean cache line (SMP)
PAPI_CA_CLN = 0x0B | PAPI_PRESET_MASK

#: Request for cache line Invalidation (SMP)
PAPI_CA_INV = 0x0C | PAPI_PRESET_MASK

#: Request for cache line Intervention (SMP)
PAPI_CA_ITV = 0x0D | PAPI_PRESET_MASK

#: Level 3 load misses
PAPI_L3_LDM = 0x0E | PAPI_PRESET_MASK

#: Level 3 store misses
PAPI_L3_STM = 0x0F | PAPI_PRESET_MASK

#: Cycles branch units are idle
PAPI_BRU_IDL = 0x10 | PAPI_PRESET_MASK

#: Cycles integer units are idle
PAPI_FXU_IDL = 0x11 | PAPI_PRESET_MASK

#: Cycles floating point units are idle
PAPI_FPU_IDL = 0x12 | PAPI_PRESET_MASK

#: Cycles load/store units are idle
PAPI_LSU_IDL = 0x13 | PAPI_PRESET_MASK

#: Data translation lookaside buffer misses
PAPI_TLB_DM = 0x14 | PAPI_PRESET_MASK

#: Instr translation lookaside buffer misses
PAPI_TLB_IM = 0x15 | PAPI_PRESET_MASK

#: Total translation lookaside buffer misses
PAPI_TLB_TL = 0x16 | PAPI_PRESET_MASK

#: Level 1 load misses
PAPI_L1_LDM = 0x17 | PAPI_PRESET_MASK

#: Level 1 store misses
PAPI_L1_STM = 0x18 | PAPI_PRESET_MASK

#: Level 2 load misses
PAPI_L2_LDM = 0x19 | PAPI_PRESET_MASK

#: Level 2 store misses
PAPI_L2_STM = 0x1A | PAPI_PRESET_MASK

#: BTAC miss
PAPI_BTAC_M = 0x1B | PAPI_PRESET_MASK

#: Prefetch data instruction caused a miss
PAPI_PRF_DM = 0x1C | PAPI_PRESET_MASK

#: Level 3 Data Cache Hit
PAPI_L3_DCH = 0x1D | PAPI_PRESET_MASK

#: Xlation lookaside buffer shootdowns (SMP)
PAPI_TLB_SD = 0x1E | PAPI_PRESET_MASK

#: Failed store conditional instructions
PAPI_CSR_FAL = 0x1F | PAPI_PRESET_MASK

#: Successful store conditional instructions
PAPI_CSR_SUC = 0x20 | PAPI_PRESET_MASK

#: Total store conditional instructions
PAPI_CSR_TOT = 0x21 | PAPI_PRESET_MASK

#: Cycles Stalled Waiting for Memory Access
PAPI_MEM_SCY = 0x22 | PAPI_PRESET_MASK

#: Cycles Stalled Waiting for Memory Read
PAPI_MEM_RCY = 0x23 | PAPI_PRESET_MASK

#: Cycles Stalled Waiting for Memory Write
PAPI_MEM_WCY = 0x24 | PAPI_PRESET_MASK

#: Cycles with No Instruction Issue
PAPI_STL_ICY = 0x25 | PAPI_PRESET_MASK

#: Cycles with Maximum Instruction Issue
PAPI_FUL_ICY = 0x26 | PAPI_PRESET_MASK

#: Cycles with No Instruction Completion
PAPI_STL_CCY = 0x27 | PAPI_PRESET_MASK

#: Cycles with Maximum Instruction Completion
PAPI_FUL_CCY = 0x28 | PAPI_PRESET_MASK

#: Hardware interrupts
PAPI_HW_INT = 0x29 | PAPI_PRESET_MASK

#: Unconditional branch instructions executed
PAPI_BR_UCN = 0x2A | PAPI_PRESET_MASK

#: Conditional branch instructions executed
PAPI_BR_CN = 0x2B | PAPI_PRESET_MASK

#: Conditional branch instructions taken
PAPI_BR_TKN = 0x2C | PAPI_PRESET_MASK

#: Conditional branch instructions not taken
PAPI_BR_NTK = 0x2D | PAPI_PRESET_MASK

#: Conditional branch instructions mispred
PAPI_BR_MSP = 0x2E | PAPI_PRESET_MASK

#: Conditional branch instructions corr. pred
PAPI_BR_PRC = 0x2F | PAPI_PRESET_MASK

#: FMA instructions completed
PAPI_FMA_INS = 0x30 | PAPI_PRESET_MASK

#: Total instructions issued
PAPI_TOT_IIS = 0x31 | PAPI_PRESET_MASK

#: Total instructions executed
PAPI_TOT_INS = 0x32 | PAPI_PRESET_MASK

#: Integer instructions executed
PAPI_INT_INS = 0x33 | PAPI_PRESET_MASK

#: Floating point instructions executed
PAPI_FP_INS = 0x34 | PAPI_PRESET_MASK

#: Load instructions executed
PAPI_LD_INS = 0x35 | PAPI_PRESET_MASK

#: Store instructions executed
PAPI_SR_INS = 0x36 | PAPI_PRESET_MASK

#: Total branch instructions executed
PAPI_BR_INS = 0x37 | PAPI_PRESET_MASK

#: Vector/SIMD instructions executed (could include integer)
PAPI_VEC_INS = 0x38 | PAPI_PRESET_MASK

#: Cycles processor is stalled on resource
PAPI_RES_STL = 0x39 | PAPI_PRESET_MASK

#: Cycles any FP units are stalled
PAPI_FP_STAL = 0x3A | PAPI_PRESET_MASK

#: Total cycles executed
PAPI_TOT_CYC = 0x3B | PAPI_PRESET_MASK

#: Total load/store inst. executed
PAPI_LST_INS = 0x3C | PAPI_PRESET_MASK

#: Sync. inst. executed
PAPI_SYC_INS = 0x3D | PAPI_PRESET_MASK

#: L1 D Cache Hit
PAPI_L1_DCH = 0x3E | PAPI_PRESET_MASK

#: L2 D Cache Hit
PAPI_L2_DCH = 0x3F | PAPI_PRESET_MASK

#: L1 D Cache Access
PAPI_L1_DCA = 0x40 | PAPI_PRESET_MASK

#: L2 D Cache Access
PAPI_L2_DCA = 0x41 | PAPI_PRESET_MASK

#: L3 D Cache Access
PAPI_L3_DCA = 0x42 | PAPI_PRESET_MASK

#: L1 D Cache Read
PAPI_L1_DCR = 0x43 | PAPI_PRESET_MASK

#: L2 D Cache Read
PAPI_L2_DCR = 0x44 | PAPI_PRESET_MASK

#: L3 D Cache Read
PAPI_L3_DCR = 0x45 | PAPI_PRESET_MASK

#: L1 D Cache Write
PAPI_L1_DCW = 0x46 | PAPI_PRESET_MASK

#: L2 D Cache Write
PAPI_L2_DCW = 0x47 | PAPI_PRESET_MASK

#: L3 D Cache Write
PAPI_L3_DCW = 0x48 | PAPI_PRESET_MASK

#: L1 instruction cache hits
PAPI_L1_ICH = 0x49 | PAPI_PRESET_MASK

#: L2 instruction cache hits
PAPI_L2_ICH = 0x4A | PAPI_PRESET_MASK

#: L3 instruction cache hits
PAPI_L3_ICH = 0x4B | PAPI_PRESET_MASK

#: L1 instruction cache accesses
PAPI_L1_ICA = 0x4C | PAPI_PRESET_MASK

#: L2 instruction cache accesses
PAPI_L2_ICA = 0x4D | PAPI_PRESET_MASK

#: L3 instruction cache accesses
PAPI_L3_ICA = 0x4E | PAPI_PRESET_MASK

#: L1 instruction cache reads
PAPI_L1_ICR = 0x4F | PAPI_PRESET_MASK

#: L2 instruction cache reads
PAPI_L2_ICR = 0x50 | PAPI_PRESET_MASK

#: L3 instruction cache reads
PAPI_L3_ICR = 0x51 | PAPI_PRESET_MASK

#: L1 instruction cache writes
PAPI_L1_ICW = 0x52 | PAPI_PRESET_MASK

#: L2 instruction cache writes
PAPI_L2_ICW = 0x53 | PAPI_PRESET_MASK

#: L3 instruction cache writes
PAPI_L3_ICW = 0x54 | PAPI_PRESET_MASK

#: L1 total cache hits
PAPI_L1_TCH = 0x55 | PAPI_PRESET_MASK

#: L2 total cache hits
PAPI_L2_TCH = 0x56 | PAPI_PRESET_MASK

#: L3 total cache hits
PAPI_L3_TCH = 0x57 | PAPI_PRESET_MASK

#: L1 total cache accesses
PAPI_L1_TCA = 0x58 | PAPI_PRESET_MASK

#: L2 total cache accesses
PAPI_L2_TCA = 0x59 | PAPI_PRESET_MASK

#: L3 total cache accesses
PAPI_L3_TCA = 0x5A | PAPI_PRESET_MASK

#: L1 total cache reads
PAPI_L1_TCR = 0x5B | PAPI_PRESET_MASK

#: L2 total cache reads
PAPI_L2_TCR = 0x5C | PAPI_PRESET_MASK

#: L3 total cache reads
PAPI_L3_TCR = 0x5D | PAPI_PRESET_MASK

#: L1 total cache writes
PAPI_L1_TCW = 0x5E | PAPI_PRESET_MASK

#: L2 total cache writes
PAPI_L2_TCW = 0x5F | PAPI_PRESET_MASK

#: L3 total cache writes
PAPI_L3_TCW = 0x60 | PAPI_PRESET_MASK

#: FM ins
PAPI_FML_INS = 0x61 | PAPI_PRESET_MASK

#: FA ins
PAPI_FAD_INS = 0x62 | PAPI_PRESET_MASK

#: FD ins
PAPI_FDV_INS = 0x63 | PAPI_PRESET_MASK

#: FSq ins
PAPI_FSQ_INS = 0x64 | PAPI_PRESET_MASK

#: Finv ins
PAPI_FNV_INS = 0x65 | PAPI_PRESET_MASK

#: Floating point operations executed
PAPI_FP_OPS = 0x66 | PAPI_PRESET_MASK

#: Floating point operations executed; optimized to count scaled single precision vector operations
PAPI_SP_OPS = 0x67 | PAPI_PRESET_MASK

#: Floating point operations executed; optimized to count scaled double precision vector operations
PAPI_DP_OPS = 0x68 | PAPI_PRESET_MASK

#: Single precision vector/SIMD instructions
PAPI_VEC_SP = 0x69 | PAPI_PRESET_MASK

#: Double precision vector/SIMD instructions
PAPI_VEC_DP = 0x6A | PAPI_PRESET_MASK

#: Reference clock cycles
PAPI_REF_CYC = 0x6B | PAPI_PRESET_MASK

#: This should always be last!
PAPI_END = 0x6C | PAPI_PRESET_MASK

"""
This module binds `PAPI High Level API
<http://icl.cs.utk.edu/projects/papi/wiki/PAPIC:PAPI.3#High_Level_Functions>`_.
Despite our desire to stay as close as possible as the original C API, we had
to make a lot of change to make this API more *pythonic*. If you are used to
the C API, please read carefully this documentation.
Example using :py:func:`flops`:
::
    from pypapi import papi_high
    # Starts counters
    papi_high.flops()  # -> Flops(0, 0, 0, 0)
    # Read values
    result = papi_high.flops()  # -> Flops(rtime, ptime, flpops, mflops)
    print(result.mflops)
    # Stop counters
    papi_high.stop_counters()   # -> []
Example counting some events:
::
    from pypapi import papi_high
    from pypapi import events as papi_events
    # Starts some counters
    papi_high.start_counters([
        papi_events.PAPI_FP_OPS,
        papi_events.PAPI_TOT_CYC
    ])
    # Reads values from counters and reset them
    results = papi_high.read_counters()  # -> [int, int]
    # Reads values from counters and stop them
    results = papi_high.stop_counters()  # -> [int, int]
"""


from ._papi import lib, ffi
from .papi_high_types import Flips, Flops, IPC, EPC
from .exceptions import papi_error


_counter_count = 0


# int PAPI_accum_counters(long long *values, int array_len);
@papi_error
def accum_counters(values):
    """accum_counters(values)
    Add current counts to the given list and reset counters.
    :param list(int) values: Values to which the counts will be added.
    :returns: A new list with added counts.
    :rtype: list(int)
    :raises PapiInvalidValueError: One or more of the arguments is invalid.
    :raises PapiSystemError: A system or C library call failed inside PAPI.
    """
    cvalues = ffi.new("long long[]", values)
    rcode = lib.PAPI_accum_counters(cvalues, len(values))
    return rcode, ffi.unpack(cvalues, len(values))


# int PAPI_num_counters(void);
def num_counters():
    """Get the number of hardware counters available on the system.
    :rtype: int
    :raises PapiInvalidValueError: ``papi.h`` is different from the version
        used to compile the PAPI library.
    :raises PapiNoMemoryError: Insufficient memory to complete the operation.
    :raises PapiSystemError: A system or C library call failed inside PAPI.
    """
    return lib.PAPI_num_counters()


# int PAPI_num_components(void);
def num_components():
    """Get the number of components available on the system.
    :rtype: int
    """
    return lib.PAPI_num_components()


# int PAPI_read_counters(long long * values, int array_len);
@papi_error
def read_counters():
    """read_counters()
    Get current counts and reset counters.
    :rtype: list(int)
    :raises PapiInvalidValueError: One or more of the arguments is invalid
        (this error should not happen with PyPAPI).
    :raises PapiSystemError: A system or C library call failed inside PAPI.
    """
    values = ffi.new("long long[]", _counter_count)
    rcode = lib.PAPI_read_counters(values, _counter_count)
    return rcode, ffi.unpack(values, _counter_count)


# int PAPI_start_counters(int *events, int array_len);
@papi_error
def start_counters(events):
    """start_counters(events)
    Start counting hardware events.
    :param list events: a list of events to count (from :doc:`events`)
    :raises PapiInvalidValueError: One or more of the arguments is invalid.
    :raises PapiIsRunningError: Counters have already been started, you must
        call :py:func:`stop_counters` before you call this function again.
    :raises PapiSystemError: A system or C library call failed inside PAPI.
    :raises PapiNoMemoryError: Insufficient memory to complete the operation.
    :raises PapiConflictError: The underlying counter hardware cannot count
        this event and other events in the EventSet simultaneously.
    :raises PapiNoEventError: The PAPI preset is not available on the
        underlying hardware.
    """
    global _counter_count
    _counter_count = len(events)

    events_ = ffi.new("int[]", events)
    array_len = len(events)

    rcode = lib.PAPI_start_counters(events_, array_len)

    return rcode, None


# int PAPI_stop_counters(long long * values, int array_len);
@papi_error
def stop_counters():
    """stop_counters()
    Stop counters and return current counts.
    :returns: the current counts (if counter started with
              :py:func:`start_counters`)
    :rtype: list
    :raises PapiInvalidValueError: One or more of the arguments is invalid
        (this error should not happen with PyPAPI).
    :raises PapiNotRunningError: The EventSet is not started yet.
    :raise PapiNoEventSetError: The EventSet has not been added yet.
    """
    global _counter_count
    array_len = _counter_count
    _counter_count = 0

    values = ffi.new("long long[]", array_len)

    rcode = lib.PAPI_stop_counters(values, array_len)

    return rcode, ffi.unpack(values, array_len)


# int PAPI_flips(float *rtime, float *ptime, long long *flpins, float *mflips);
@papi_error
def flips():
    """flips()
    Simplified call to get Mflips/s (floating point instruction rate), real
    and processor time.
    :rtype: pypapi.papi_high_types.Flips
    :raises PapiInvalidValueError: The counters were already started by
        something other than :py:func:`flips`.
    :raises PapiNoEventError: The floating point operations or total cycles
        event does not exist.
    :raises PapiNoMemoryError: Insufficient memory to complete the operation.
    """
    rtime = ffi.new("float*", 0)
    ptime = ffi.new("float*", 0)
    flpins = ffi.new("long long*", 0)
    mflips = ffi.new("float*", 0)

    rcode = lib.PAPI_flops(rtime, ptime, flpins, mflips)

    return rcode, Flips(
            ffi.unpack(rtime, 1)[0],
            ffi.unpack(ptime, 1)[0],
            ffi.unpack(flpins, 1)[0],
            ffi.unpack(mflips, 1)[0]
            )


# int PAPI_flops(float *rtime, float *ptime, long long *flpops, float *mflops);
@papi_error
def flops():
    """flops()
    Simplified call to get Mflops/s (floating point operation rate), real
    and processor time.
    :rtype: pypapi.papi_high_types.Flops
    :raises PapiInvalidValueError: The counters were already started by
        something other than :py:func:`flops`.
    :raises PapiNoEventError: The floating point instructions or total cycles
        event does not exist.
    :raises PapiNoMemoryError: Insufficient memory to complete the operation.
    """
    rtime = ffi.new("float*", 0)
    ptime = ffi.new("float*", 0)
    flpops = ffi.new("long long*", 0)
    mflops = ffi.new("float*", 0)

    rcode = lib.PAPI_flops(rtime, ptime, flpops, mflops)

    return rcode, Flops(
            ffi.unpack(rtime, 1)[0],
            ffi.unpack(ptime, 1)[0],
            ffi.unpack(flpops, 1)[0],
            ffi.unpack(mflops, 1)[0]
            )


# int PAPI_ipc(float *rtime, float *ptime, long long *ins, float *ipc);
@papi_error
def ipc():
    """ipc()
    Gets instructions per cycle, real and processor time.
    :rtype: pypapi.papi_high_types.IPC
    :raises PapiInvalidValueError: The counters were already started by
        something other than :py:func:`ipc`.
    :raises PapiNoEventError: The total instructions or total cycles event does
        not exist.
    :raises PapiNoMemoryError: Insufficient memory to complete the operation.
    """
    rtime = ffi.new("float*", 0)
    ptime = ffi.new("float*", 0)
    ins = ffi.new("long long*", 0)
    ipc_ = ffi.new("float*", 0)

    rcode = lib.PAPI_ipc(rtime, ptime, ins, ipc_)

    return rcode, IPC(
            ffi.unpack(rtime, 1)[0],
            ffi.unpack(ptime, 1)[0],
            ffi.unpack(ins, 1)[0],
            ffi.unpack(ipc_, 1)[0]
            )


# int PAPI_epc(int event, float *rtime, float *ptime, long long *ref,
#              long long *core, long long *evt, float *epc);
@papi_error
def epc(event=0):
    """epc(event=0)
    Gets (named) events per cycle, real and processor time, reference and
    core cycles.
    :param int event: The target event (from :doc:`events`, default:
        :py:const:`pypapi.events.PAPI_TOT_INS`).
    :rtype: pypapi.papi_high_types.EPC
    """
    rtime = ffi.new("float*", 0)
    ptime = ffi.new("float*", 0)
    ref = ffi.new("long long*", 0)
    core = ffi.new("long long*", 0)
    evt = ffi.new("long long*", 0)
    epc_ = ffi.new("float*", 0)

    rcode = lib.PAPI_epc(event, rtime, ptime, ref, core, evt, epc_)

    return rcode, EPC(
            ffi.unpack(rtime, 1)[0],
            ffi.unpack(ptime, 1)[0],
            ffi.unpack(ref, 1)[0],
            ffi.unpack(core, 1)[0],
            ffi.unpack(evt, 1)[0],
            ffi.unpack(epc_, 1)[0]
            )


def flop(function):
    high.start_counters([events.PAPI_FP_OPS, ])
    function
    x = high.stop_counters()
    print(x)

flop