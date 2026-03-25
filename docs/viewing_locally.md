# Viewing Locally

To view the documentation on your local host, follow these steps:

1.  **Open a Terminal**: Open your command line interface (e.g., PowerShell, Cmd, or Bash).
2.  **Activate Environment**: Ensure you are in the correct conda environment:
    ```bash
    conda activate cavsim3d
    ```
3.  **Run MkDocs Serve**:
    ```bash
    mkdocs serve
    ```
4.  **Open Browser**: Navigate to `http://127.0.0.1:8000` in your web browser.

The server will automatically reload whenever you make changes to the documentation files (`.md`) or the configuration (`mkdocs.yml`).

!!! tip "Troubleshooting"
    If the port `8000` is already in use, you can specify a different port:
    ```bash
    mkdocs serve -a 127.0.0.1:8001
    ```
