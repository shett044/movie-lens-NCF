# FROM bitnami/pytorch
FROM pytorch/pytorch:latest

RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*


RUN pip install pipenv
WORKDIR /movie-lens-NeuMLP
COPY ["Pipfile", "Pipfile.lock" ,"./"]
RUN pipenv install --ignore-pipfile --deploy --system
COPY ["ml-100k", "./ml-100k/"]
COPY ["models", "./models/"]
COPY ["argparser.py", "boilerplate.py", "data_utils.py", "evaluators.py", "main.py", "trainer.py", "utils.py",  "./"]

# Train the model
# CMD [ "python", "./main.py"]