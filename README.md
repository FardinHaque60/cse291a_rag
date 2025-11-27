# cse291a_rag
RAG project for cse 291a fall 2025

## setting up the environment
### running with docker (recommended for non-MacOS users)
1. to build the container run `docker build -t retail_rag .`
2. create a `.env` file and paste the contents from [this document](https://docs.google.com/document/d/1IB_VThi-pA60TgTRRDWw4OTMAn-SzDxKJkJTCiWKcZA/edit?usp=sharing). 
3. run the container `docker run -it --env-file .env retail_rag`. this will start a shell inside the docker container to run commands from

### running locally with venv (recomended for MacOS users)
repo uses a python `venv` to manage dependencies. python version `>= 3.12.8` needed. this set up has only been tested on MacOS.
follow the below steps for the initial `venv` set up once the repo is cloned on your computer.
1. run `python -m venv venv` to create the venv
2. run `source venv/bin/activate` to activate the venv
3. run `pip install -r requirements.txt` to install all the needed dependencies
4. create a `.env` file and copy contents from [this document](https://docs.google.com/document/d/1IB_VThi-pA60TgTRRDWw4OTMAn-SzDxKJkJTCiWKcZA/edit?usp=sharing) into the `.env` file.

clean up: to deactivate the `venv` once finished using, run `deactivate` to turn it off.

## running the program
1. for reproducing metrics using our prompts:
    - phase 1: run `python eval/evaluation.py`
    - phase 2: run `python eval/evaluation2.py`
These commands will generate a file in `eval/out` that you can inspect with metrics.
2. for running with your own queries:
    - phase 1: go to `phase_1_pipeline/inference.py` and input your queries into the `QUERIES` variable on line 15. Run the code using `python phase_1_pipeline/inference.py`. Inspect the results in `phase_1_pipeline/results/`
    - phase 2: go to `phase_2_pipeline/p0_runner.py` and input your queries into the `QUERIES` variable on line 14. Run the code using `python phase_2_pipeline/p0_runner.py`

## more details
visit the `README.md` of any directories for more information (ex. `phase_1_pipeline/`)