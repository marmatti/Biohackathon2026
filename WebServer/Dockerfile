# Start with a standard, lightweight Ubuntu Linux image
FROM ubuntu:22.04

# Stop interactive prompts from halting the build process
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install necessary system dependencies (wget, python, and libraries for OpenCV)
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Download and extract the official Linux Headless version of Ilastik
# (We place it in the /opt/ directory, which is standard for add-on software)
RUN wget https://files.ilastik.org/ilastik-1.4.0-Linux.tar.bz2 -O ilastik.tar.bz2 && \
    tar -xjf ilastik.tar.bz2 && \
    rm ilastik.tar.bz2 && \
    mv ilastik-1.4.0-Linux /opt/ilastik

# 3. Set up your Python environment
WORKDIR /app
RUN pip3 install opencv-python-headless

# 4. Copy your specific files into the Docker container
COPY filament_model_v1.ilp /app/
COPY deploy.py /app/

# 5. Create directories where you will mount your images later
RUN mkdir -p /app/data/input
RUN mkdir -p /app/data/output

# 6. Tell Docker what to do when the container starts
CMD ["python3", "deploy.py"]
