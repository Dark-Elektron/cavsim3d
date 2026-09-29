# ============================================
# 1. Start from a base image with conda
# ============================================
FROM condaforge/miniforge3:latest

# ============================================
# 2. Set metadata
# ============================================
LABEL maintainer="Sosoho-Abasi Udongwo <numurho@gmail.com>"
LABEL description="cavsim3d - 3D EM cavity simulation"

# ============================================
# 3. Set environment variables
# ============================================
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1
ENV DEBIAN_FRONTEND=noninteractive

# ============================================
# 4. Install system-level dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 5. Create conda environment
# ============================================
RUN conda create -n cavsim3d python=3.11 -y

# Use this environment for all subsequent commands
SHELL ["conda", "run", "-n", "cavsim3d", "/bin/bash", "-c"]

# ============================================
# 6. Install conda-only packages
# ============================================
RUN conda install -y -c conda-forge \
    pythonocc-core \
    pythreejs \
    ipywidgets \
    && conda clean -afy

# ============================================
# 7. Install pip packages
# ============================================
RUN pip install --no-cache-dir \
    ngsolve \
    pytest

# ============================================
# 8. Copy your project into the container
# ============================================
WORKDIR /app
COPY . /app

# ============================================
# 9. Install your project
# ============================================
RUN pip install --no-cache-dir -e .

# ============================================
# 10. Default command when container starts
# ============================================
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cavsim3d"]
CMD ["python"]