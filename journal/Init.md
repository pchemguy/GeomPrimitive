<!--
https://chatgpt.com/c/68ff6930-6074-832c-bc10-0f783c79789c
-->

## **Prompt: Bootstrap a Python Reusable Library (Conda + uv + Hatch + Ruff)**

I want you to **bootstrap a reusable Python library project** that is clean, maintainable, and reproducible across environments.  
The project must follow **modern Python packaging conventions** and integrate **Conda/Mamba** for environment isolation, **uv** for dependency management, and **hatch** for packaging, versioning, and environment orchestration.

Use **Ruff** as the single unified tool for linting, formatting, and type checking.  
Do **not** include GitHub Actions, Makefiles, or documentation frameworks (MkDocs/Sphinx).  
Everything should be self-contained and easy to bootstrap locally or in CI systems.

---

### **1. Project Overview**

- **Purpose:** Reusable Python library    
- **Python version:** 3.9+
- **Primary environment manager:** Conda / Mamba
- **Dependency resolver:** `uv`
- **Build & lifecycle manager:** `hatch`
- **Code quality tool:** `ruff` (formatter, linter, type checker)

---

### **2. Project Structure**

Follow modern `src/` layout:

```
project_name/
├── src/project_name/
│   ├── __init__.py
│   ├── core.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_core.py
├── env/
│   ├── environment.yml
│   └── uv.lock
├── demo/
│   ├── demo1.py
│   └── demo2.py
├── pyproject.toml
├── README.md
├── .gitignore
└── LICENSE
```

---

### **3. Environment and Dependency Management**

**Conda environment file:**

```yaml
name: project_name
channels:
  - conda-forge
dependencies:
  - python>=3.9
  - pip
  - uv
  - hatch
  - ruff
  - pytest
```

**uv.lock**

- Generated automatically by `uv pip compile` or equivalent.
- Keep version-locked for reproducibility.

---

### **4. Build and Project Configuration (`pyproject.toml`)**

Use **Hatch** for project metadata, builds, and environment management.

Example:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "project_name"
version = "0.1.0"
description = "A reusable Python library."
authors = [{ name = "Your Name", email = "you@example.com" }]
readme = "README.md"
requires-python = ">=3.9"
dependencies = []

[tool.hatch.envs.default]
dependencies = [
  "ruff",
  "pytest",
  "uv",
]

[tool.hatch.envs.default.scripts]
test = "pytest"
lint = "ruff check src tests"
format = "ruff format src tests"
typecheck = "ruff check --select TYP src"

[tool.ruff]
line-length = 88
target-version = "py39"
select = ["E", "F", "I", "UP", "B", "ANN"]
ignore = ["D"]
fix = true
format = "ruff"

[tool.ruff.lint]
extend-select = ["TYP"]

[tool.hatch.build.targets.sdist]
include = ["src", "README.md", "LICENSE"]

[tool.hatch.build.targets.wheel]
include = ["src", "README.md", "LICENSE"]
```

---

### **5. Testing and Quality**

- Framework: `pytest`
- Command: `hatch run test`
- Ruff handles:
    - `ruff check` → linting & static analysis
    - `ruff format` → auto-formatting
    - `ruff check --select TYP` → type analysis

---

### **6. Documentation & Metadata**

- Only minimal `README.md` with:
    - Purpose
    - Environment setup (`mamba env create -f env/environment.yml`)
    - Usage example
    - Development commands (`hatch run lint`, etc.)
- Add `LICENSE` (MIT by default).
- No Sphinx or MkDocs.

---

### **7. Output Requirements**

When bootstrapping:
- Output full directory tree.
- Include all file contents with realistic placeholders.
- Ensure configurations are syntactically correct and runnable.
- Provide a one-line example function and corresponding test.

---

### **TL;DR Version**

> Bootstrap a reusable Python library using **Conda/Mamba** for environment management, **uv** for fast dependency resolution, and **hatch** for packaging, lifecycle, and scripts.  
> Use **ruff** for formatting, linting, and type checking.  
> No Makefile, GitHub Actions, or Sphinx.  
> Include `environment.yml`, `uv.lock`, `pyproject.toml`, and minimal working code/tests.

---

> [!TODO]
> 
> Verify that with `pip install -e .` package can be installed during the dev phase in editable mode for running demos.

---
