# CUDA-union-find-decoder

## Build Instructions

Give build script executable permissions and run:

```
sudo chmod +x build.sh
./build.sh
```

---

## Run Instructions

### CPU Version

```
./toric_cpu
```

### GPU Version

```
./toric_gpu
```

---

## Notes

- When tested with 500 runs on RTX GeForce 3060 GPU, average time per lattice is 0.32 ms.
- On AMD Ryzen 7 4800U CPU,average tiime per lattice is 1.8 ms with optimization flags.
- This is roughly a 5x speedup for a single lattice, and can go higher for throughput
