# Latency Jitter Detection Pipeline

This project simulates queueing latency from packet traces, generates per-flow jitter ground truth, and evaluates a sketch-based online detector.

## What the code does

The pipeline is split into four parts:

1. `queuing_simulation.py`
   Reads a CAIDA-style `.dat` trace, simulates a FIFO output queue, and writes packet latency records.
2. `ground_truth.py`
   Applies a per-flow EWMA-based jitter detector to create ground-truth labels.
3. `sketch.py`
   Implements `JitterSketch`, a memory-efficient sketch for online jitter detection.
4. `main.py`
   Runs the full experiment end to end and reports evaluation metrics such as precision, recall, F1 score, and throughput.

## Project structure

```text
latency_jitter/
|-- main.py
|-- queuing_simulation.py
|-- ground_truth.py
|-- sketch.py
|-- data/
|-- README.md
|-- .gitignore
```

## Requirements

- Python 3.10 or newer
- No third-party packages are required for the current codebase

## How to run

Run the full pipeline with a real CAIDA trace:

```powershell
python main.py --dat path\to\trace.dat
```

Run a quick synthetic smoke test:

```powershell
python main.py --synthetic --num-pkts 200000
```





