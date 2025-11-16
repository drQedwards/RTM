# RTM — Recursive Transformer Model  
### Self-Referential Inference • Persistent Memory • KV-Slotting • ARC-AGI Performance

Author: Dr. Josef Kurk Edwards (drQedwards)  
Status: Research Release  
License: MIT

---
# Recursive Transformer Model (RTM)

## Overview
The **Recursive Transformer Model (RTM)** extends standard transformers with **persistent memory**, **temporal decay**, **consensus validation**, and **contradiction detection**, enabling stateful reasoning across inference sessions. This directly addresses the problem of **nostalgic incorrectness**—a model’s tendency to retain outdated or contradicted beliefs.

RTM draws from Dr. Josef “Q.” Edwards’ 2025 research on **Persistent Memory Logic Loops (PMLL)** and the Enhanced Reconsideration System (ERS). RTM is complementary to the Tiny Recursion Model (TRM), and a hybrid TRM–RTM architecture is possible.

---

## Features

### ✅ Persistent Memory Store
RTM stores past outputs in vectorized memory blocks with:
- **Timestamped entries**
- **Confidence scores**
- **Embedding-based similarity lookup**

### ✅ Temporal Decay
Older or unsupported memories naturally lose confidence, reducing stale knowledge.

### ✅ Consensus Strengthening
Related memories reinforce one another, improving model reliability.

### ✅ Contradiction Detection
Conflicting memories penalize each other’s confidence, reducing knowledge drift.

### ✅ PMLL (Persistent Memory Logic Loop) Scaffold
The repository includes a placeholder for:
- Lattice-based tensor routing  
- Memory hashing  
- Commitment and re-evaluation cycles  

---

## Installation

```bash
git clone <your repo URL>
cd recursive_transformer
pip install -r requirements.txt
```

Requirements:
- Python 3.10+
- numpy
- scikit-learn
- dateutil

---

## Basic Usage

### Example: using a dummy echo model

```python
from recursive_transformer import RecursiveTransformerModel

def echo_model(prompt: str):
    return prompt

model = RecursiveTransformerModel(base_model=echo_model)

result, confidence = model.generate("What is memory?")
print(result, confidence)
```

---

## File Structure

```
recursive_transformer/
│
├── recursive_transformer/
│   ├── memory.py
│   ├── model.py
│   ├── consensus.py
│   ├── contradiction.py
│   ├── decay.py
│   ├── pmll.py
│   └── __init__.py
│
└── examples/
    └── simple_example.py
```

---

## Comparison: TRM vs RTM vs Hybrid

| Feature | TRM | RTM | Hybrid |
|--------|-----|-----|---------|
| Primary skill | Recursive reasoning | Persistent state | Both |
| Memory | Stateless | Persistent & decaying | Persistent + recursive |
| Best for | ARC puzzles | Knowledge systems | AGI-level reasoning |
| Parameter size | ~7M | Model-dependent | Model-dependent |

---

## Citation

If you use RTM in research:

```
Edwards, J. (2025). The Recursive Transformer Model: Architecture, Theory, and Implementation with Persistent Memory Logic Loops.
```

---

## License
MIT License


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
