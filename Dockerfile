FROM python:3.7.12

# Update and install system packages
RUN apt-get update -y && \
    apt-get install --no-install-recommends -y -q \
        git libpq-dev python-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --upgrade pip

# Install poetry
RUN python -m pip install 'poetry>=1.1.13'

# Set environment variables
ENV PROJECT_DIR /groupby

RUN mkdir -p $PROJECT_DIR
COPY . $PROJECT_DIR

# Set working directory
WORKDIR $PROJECT_DIR
RUN POETRY_VIRTUALENVS_CREATE=false python -m poetry install --no-interaction

# Run dbt
ENTRYPOINT ["python", "-m", "groupby_capstone", "webserver"]
