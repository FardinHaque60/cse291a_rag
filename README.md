# cse291a_rag
RAG project for cse 291a fall 2025

## running the program
### running with docker
1. to build the container ```docker build -t my-python-app .```
2. run the container ```docker run -it --env-file .env my-python-app```. this will start a shell inside the docker container to run commands from

### running locally with venv
repo uses a python `venv` to manage dependencies. python version `>= 3.12.8` needed.
follow the below steps for the initial `venv` set up once the repo is cloned on your computer.
1. run `python -m venv venv` to create the venv
2. run `source venv/bin/activate` to activate the venv
3. run `pip install -r requirements.txt` to install all the needed dependencies
4. [optional: only needed if you are trying to run open source RAGs from phase 1] add environment variables to a file called `.env`. copy contents from [this document](https://docs.google.com/document/d/1IB_VThi-pA60TgTRRDWw4OTMAn-SzDxKJkJTCiWKcZA/edit?usp=sharing) into the `.env` file.

clean up: to deactivate the `venv` once finished using, run `deactivate` to turn it off.

## using open source rags (phase 1)
visit the `README.md` of any directories for open source RAGs (ex. `phase_1_pipeline/`)