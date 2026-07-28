"""``python -m testbench`` — serve the app locally (``--dev`` for hot reload)."""

from ngapp.cli.serve_standalone import main

main(app_module="testbench.appconfig")
