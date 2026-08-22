# AI Agent Instructions

The active engine (`up_cpor/`, `cpp/`) is a native Python/C++ implementation and needs no C#/.NET/Mono to build or run; the legacy `CPORLib`/`TestCPORLib` C# reference project is kept in-tree for comparison but is not part of the native build. When operating autonomously, follow these guidelines:

## Environment Setup
- **Python**: Use the provided Conda environment `environment.yml`.
- **C++**: CMake 3.15+ and a C++20 compiler, plus Z3 development headers (`libz3-dev` on Debian/Ubuntu, `z3` via Homebrew on macOS) to build `cpp/`.
- **Testing**: Tests use the `pytest` package. To test meta planners, you must install the `up-tamer` and `up-pyperplan` Python packages. Legacy C#-dependent tests live under `legacy_tests/` and are outside the default `pytest` collection path.

## Development Workflow
- **Recompilation is mandatory**: You MUST run `./build.sh` after making *any* modifications to the code for the changes to take effect.