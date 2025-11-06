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
4. The experiment produces a very large amount of output and takes a long time, but errors usually occur within the first 10 seconds. To see errors, apply a 10-second timeout when running.
5. Do not modify the code unless I explicitly request it. First, provide solutions.
6. **Never** delete any existing files under any circumstance.