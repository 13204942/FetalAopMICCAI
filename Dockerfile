FROM pytorch/pytorch:latest

RUN useradd -ms /bin/bash algorithm
USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm model/ /opt/algorithm/model/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm segmentation.py /opt/algorithm/

RUN python3 -m pip install --user -r requirements.txt

ENTRYPOINT python3 -m process $0 $@

LABEL nl.diagnijmegen.rse.algorithm.name=seg_algorithm