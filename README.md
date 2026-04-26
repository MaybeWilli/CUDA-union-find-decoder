# CUDA-union-find-decoder

GPU-accelerated quantum error correction code based on a parallelized version of the union-find decoder algorithm described Delfosse & Nickerson's 2017 paper “Almost-linear time decoding algorithm for topological codes” (https://arxiv.org/abs/1709.06218)

Decoder's peeling step is based on Defosse and Zémor's 2020 paper "Linear-time maximum likelihood decoding of surface codes over the quantum erasure channel" (https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033042)

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
- This is roughly a 5x speedup for a single lattice, and can go much higher for throughput
