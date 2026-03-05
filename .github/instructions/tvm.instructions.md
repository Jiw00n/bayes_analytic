---
applyTo: '**'
---
Coding standards, domain knowledge, and preferences that AI should follow.


# TVM Instructions
This document provides specific instructions for AI models to follow when generating or modifying code within the TVM project, particularly in the `tvm-ansor` directory. Adhering to these guidelines ensures consistency, quality, and maintainability of the codebase.
## General Guidelines
1. Before starting any experiment, run the function `release` (already defined in .bashrc).
2. Always use the command `python /root/work/tvm-ansor/gallery/tune_network_cuda.py` to run the script.
3. To build, always run ninja inside /root/work/tvm-ansor/build-release.
4. The experiment produces a very large amount of output and takes a long time, but errors usually occur within the first 10 seconds. To see errors, apply a 10-second timeout(or a little bit longer) when running.
5. If any information required for implementation or modification is uncertain (e.g., function behavior, call paths, expected inputs/outputs, side effects, configuration flags, build/runtime assumptions), **do not proceed by inductive guessing** or “likely” reasoning, and **do not write a plan based on assumptions**.  
   Instead, you **must**:
   - **Locate the authoritative source of truth in this repository** (relevant Python/C++ files, headers, build scripts, and tests) and **read the actual code**.
   - Identify the exact symbols and their definitions (e.g., class/function signatures, where they are declared/defined, and where they are called).
   - Trace the concrete control/data flow needed for the change (call chain, ownership/lifetime expectations, invariants).
   - Derive the implementation plan **deductively from the code** and reference the specific files/functions you relied on.
   - Only after the above, apply changes.  
   If the code cannot be found locally, you must state precisely what is missing and why you cannot proceed without it (no speculative planning).
6. **Never** delete any existing files under any circumstance.