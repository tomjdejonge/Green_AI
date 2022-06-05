from pypapi import events, papi_high as high

high.start_counters([events.PAPI_FP_OPS,])
# Do something
x=high.stop_counters()