## PFFDTD benchmarks

Notes:

- 0.1s simulated in Musikverein model with 11 RLC branches per material
- Speed of sound 343.2m/s
- Cartesian scheme runs use 7.75PPW and CFL=0.577
- FCC scheme runs use 5.6PPW and CFL=0.999
- MVPS = Npts*Nsamples/run-time/1e6
- MVPS/GPU is MVPS per GPU
- Min/s is minutes of compute time need for one second of audio output
- All runs in single precision
- All GPUs had PCIe 3.0 16x width lanes, except for Ampere cards (PCIe 4.0 16x)
- K80 is dual-GPU card.  Used one GPU per card, except for 16x run
