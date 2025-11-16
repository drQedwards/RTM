# RTM — Recursive Transformer Model  
### Self-Referential Inference • Persistent Memory • KV-Slotting • ARC-AGI Performance

Author: Dr. Josef Kurk Edwards (drQedwards)  
Status: Research Release  
License: MIT

---

RTM is an experimental transformer architecture that fuses:

- Recursive forward passes (self-calling inference path)  
- PMLL — Persistent Memory Logic Loop  
- KV-Slotting for long-context efficiency  
- ARC-AGI benchmark evaluation modules  

RTM explores the hypothesis:

*A transformer equipped with recursive inference and persistent memory can solve abstraction tasks that defeat standard models.*

---

## Features

### Recursive Core  
The model calls itself under a continuation policy, enabling multistep reasoning chains.

### PMLL Integration  
Hash-chained semantic memory with Race/Trace retrieval and compressed persistence.

### KV-Slotting  
Selective KV-cache rehydration for long-horizon inference without memory blowup.

### ARC-AGI Evaluation  
Includes adapters, evaluation loops, scoring, and ablation hooks.

---

## Installation

```bash
git clone https://github.com/drQedwards/RTM.git
cd RTM
unzip recursive_transformer.zip -d rtm_src
pip install -r requirements.txt
```

---

## ARC-AGI Benchmarking

The `benchmarks/arc/` module provides:

- Task loading  
- Model ↔ ARC adapters  
- Scoring  
- Recursion-depth analysis  
- Memory hit-rate analysis  

Run:

```bash
python benchmarks/arc/run_arc.py --model rtm --config configs/arc/rtm_default.yaml
```

---

## Roadmap

- Automated ARC reporting  
- Memory graph visualizer  
- Docker reproducibility  
- TechRxiv whitepaper

---

## License  
MIT License. See LICENSE for details.
