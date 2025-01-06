ARG sourceimage=traderpilotorg/traderpilot
ARG sourcetag=develop
FROM ${sourceimage}:${sourcetag}

# Install dependencies
COPY requirements-plot.txt /traderpilot/

RUN pip install -r requirements-plot.txt --user --no-cache-dir
