# gshapes

A reusable Python library for geometric shapes.

## Setup

This project uses Conda/Mamba for environment management and Hatch for task running.

1.  **Create the Conda Environment**

    First, create the Conda environment using the provided `environment.yml` file. This will install Python, `uv`, and `hatch`.

    ```bash
    conda env create -f env/environment.yml
    ```

2.  **Activate the Environment**

    Activate the newly created environment:

    ```bash
    conda activate gshapes
    ```

## Development

All development tasks are managed through `hatch` scripts defined in `pyproject.toml`.

### Running Tests

To run the test suite, use the following command:

```bash
hatch run test
```

### Code Quality

This project uses `ruff` for linting, formatting, and type checking.

*   **Format code:**
    ```bash
    hatch run format
    ```

*   **Lint code:**
    ```bash
    hatch run lint
    ```

*   **Run all checks (format, lint, test):**
    ```bash
    hatch run all
    ```

### Running Demos

To run the demo scripts, you can use `hatch` to execute them within the project's environment:

```bash
hatch run python demo/demo1.py
hatch run python demo/demo2.py
```
