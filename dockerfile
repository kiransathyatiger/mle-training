#Deriving the latest base image
FROM python:latest


# Any working directory can be chosen as per choice like '/' or '/home' etc

WORKDIR /

#to COPY the remote file at working directory in container

COPY main.py ./
COPY ingestdata.py ./
COPY data/raw/housing/housing.csv ./data/raw/housing/housing.csv
COPY score.py ./
COPY train.py ./
COPY logger.py ./
COPY data/processed/housing_train.csv ./
COPY data/processed/housing_test.csv ./

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install the required packages

RUN pip install numpy \
    pandas \
    scikit-learn\
    logger
   	     



#CMD instruction should be used to run the software
#contained by your image, along with any arguments.


CMD [ "python", "./main.py"]