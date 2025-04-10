#Start with this version of Python. Slim because it is faster and will only add what we need 
FROM python:3.11-slim

#To set the working directory inside our container. This tells Docker that everything will happen in /app folder
WORKDIR /app

#Install system dependencies so that it can build some packages 
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

#Install Poetry -- I need this for my pyproject.toml file that has the dependencies 
RUN curl -sSL https://install.python-poetry.org | python3 -

#Add poetry to PATH to make sure that the commands for poetry are available 
ENV PATH="/root/.local/bin:$PATH"

#This will copy only the dependencies. So if my code change, but not the dependency, no need to reinstall everything
COPY pyproject.toml poetry.lock* /app/

#Install dependencies that are in my pyproject.toml file 
RUN poetry install --no-root --only main

#Copy the rest of my code into the container 
COPY . /app

#Make sure models dir exists since this is where my Models will be stored 
RUN mkdir -p /app/models

#Expose port for FastAPI
EXPOSE 8000

#Command that runs when i launch the container 
CMD ["poetry", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

