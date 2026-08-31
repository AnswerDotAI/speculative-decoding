# Speculative Decoding: An Interactive Tutorial

Speculative decoding is a widely used method to speed up generation of tokens by LLMs. It is an ongoing, active area of machine learning research, and it is widely used in production systems (including at NVIDIA, Google, and DeepSeek).

This tutorial assumes you are familiar with Python and probability distributions, and that you know
the rough idea of how an LLM generates text one token at a time. It does not assume any background with speculative decoding.

The tutorial has three parts. The first offers an interactive visualization of what the
algorithm does. The second checks on a toy data set that speculative decoding samples from the
same distribution as the slower model it replaces. The third works out where the speedup
comes from, and closes with an exercise.

## What you need

Python 3.12 or newer. All the code runs on a laptop. You do not need a GPU, a model
download, or an API key.

## Setup

Create and activate a virtual environment in this folder, then install the dependencies.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Running the tutorial

Start Jupyter from inside this folder:

    jupyter lab speculative-decoding-tutorial.ipynb

Starting it from anywhere else will break the visualization, because the notebook imports
`specdec_viz_implementation.py` from the folder it is launched in.

Run the cells in order.

## The interactive visualization

The visualization runs on a small web server started by the notebook itself, and it
displays at `http://localhost:8000`.

If port 8000 is already in use on your machine, the visualization will not appear. You can update the `PORT` variable to a free port in the cell that starts the server.

If you cannot run the server at all, the tutorial plays a screen recording of the
visualization instead, from `spec_dec_video.mp4`.

## Files

- `speculative-decoding-tutorial.ipynb` — the tutorial
- `specdec_viz_implementation.py` — the visualization, imported by the tutorial
- `spec_dec_video.mp4` — a screen recording of the visualization, played by the tutorial
- `specdec-viz-implementation.ipynb` — how the visualization was built, for readers who
  want to modify it
- `requirements.txt` — the dependencies
- `speculative-decoding.pdf` — the two-page summary for the NeurIPS 2026 Education Track
