# JAX Transpose Test Harness

A tiny project to develop and **test your JAX transpose** implementation without giving away the solution.

## Quick start

```bash
# 1) Create a fresh venv (recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install deps (CPU-only)
pip install -r requirements.txt

# 3) Run tests (they will fail until you implement the function)
pytest -q
```

## Where to code

Edit `src/jax_transpose/transpose.py` and implement `transpose_matrix`.  
Do NOT change the function name or signature.

## Notes
- Tests check shape/value correctness, involution (T(T(X)) == X), grad/VJP property, JIT/vmap behavior, and dtype preservation.
- If you want GPU JAX later, follow official JAX installation docs for your CUDA/cuDNN version.
