FROM python:latest
RUN pip install python-nexus numpy 
RUN pip install --upgrade pandas
RUN pip install scikit-learn
WORKDIR /src
