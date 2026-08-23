# Speculative Decoding: An Interactive Tutorial

An interactive tutorial on speculative decoding, aimed at advanced undergraduates.

It assumes you are familiar with Python and probability distributions, and that you know
roughly how an LLM generates text one token at a time. It does not assume you have met
speculative decoding before.

The tutorial has three parts. The first builds an interactive visualization of the
algorithm. The second checks on a toy data set that speculative decoding samples from the
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

If port 8000 is already in use on your machine, the visualization will not appear. Either
stop whatever is using the port, or set `PORT` to a free port in the cell that starts the
server.

Stop the server with `server.stop()` when you have finished with it. If the visualization
misbehaves, stop the server and run the cells again.

If you cannot run the server at all, the folder includes a screen recording of the
visualization instead.

## Files

- `speculative-decoding-tutorial.ipynb` — the tutorial
- `specdec_viz_implementation.py` — the visualization, imported by the tutorial
- `specdec-viz-implementation.ipynb` — how the visualization was built, for readers who
  want to modify it
- `requirements.txt` — the dependencies
