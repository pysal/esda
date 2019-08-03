# build the container
container:
	docker build -t libpysal .

# run jupyter notebook for development
nb:
	docker run --rm -p 8888:8888 -v ${PWD}:/home/jovyan libpysal

# run a shell for development
cli:
	docker run -it -p 8888:8888 -v ${PWD}:/home/jovyan libpysal /bin/bash

cov:
	pytest --cov-report html --cov=esda
