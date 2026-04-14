
# TVM Instructions
This document provides specific instructions for AI models to follow when generating or modifying code within the TVM project, particularly in the `tvm-ansor` directory. Adhering to these guidelines ensures consistency, quality, and maintainability of the codebase.
## General Guidelines
1. Set environment.
    - source /root/work/venv/bin/activate
    - export TVM_HOME=/root/work/tvm-ansor
    - export PYTHONPATH=$TVM_HOME/python
    - export TVM_LIBRARY_PATH=$TVM_HOME/build-release
2. If any information required for implementation or modification is uncertain (e.g., function behavior, call paths, expected inputs/outputs, side effects, configuration flags, build/runtime assumptions), **do not proceed by inductive guessing** or “likely” reasoning, and **do not write a plan based on assumptions**.  
   Instead, you **must**:
   - **Locate the authoritative source of truth in this repository** (relevant Python/C++ files, headers, build scripts, and tests) and **read the actual code**.
   - Identify the exact symbols and their definitions (e.g., class/function signatures, where they are declared/defined, and where they are called).
   - Trace the concrete control/data flow needed for the change (call chain, ownership/lifetime expectations, invariants).
   - Derive the implementation plan **deductively from the code** and reference the specific files/functions you relied on.
   - Only after the above, apply changes.  
   If the code cannot be found locally, you must state precisely what is missing and why you cannot proceed without it (no speculative planning).
3. **Never** delete any existing files under any circumstance.