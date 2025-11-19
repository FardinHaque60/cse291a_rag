# phase 2 pipeline

- folder contains the pipeline for all the changes needed in phase 2
- full pipeline can be run with `0_runner.py`
- pipeline stages are prefixed with a number with file format: `pn_{stage}.py`
- `data_load.py` is used to load data into qdrant vector storage, the runner stages will use the data in storage for inferencing
- see `phase_2_eval` dir to see how pipeline is being used and how it is performing