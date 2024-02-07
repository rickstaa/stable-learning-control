# Use the official Miniconda3 base image
FROM continuumio/miniconda3:latest

# Set environment variables for Conda
ENV CONDA_HOME="/opt/conda"
ENV PATH="$CONDA_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a Conda environment with Python 3.10
RUN conda create -n slc python=3.10

# Activate the Conda environment
SHELL ["conda", "run", "-n", "slc", "/bin/bash", "-c"]

# Copy the repo code into the Docker image
COPY . /stable-learning-control

# Clone stable-gym package
RUN git clone -b han2020 https://github.com/rickstaa/stable-gym.git

# Set the working directory
WORKDIR /stable-learning-control

# Install mpi4py using Conda
# NOTE: Done since the pip version of mpi4py does not to seem to work in a Conda environment.
RUN conda install mpi4py

# Install slc package
RUN pip install -e .[mujoco]

# Install the stable-gym package
RUN pip install -e ../stable-gym

# Add Conda activation to .bashrc (optional)
RUN echo "source activate slc" >> /root/.bashrc 

# Start the experiments
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "slc", "python", "-m", "stable_learning_control.run"]
