FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY pipeline_stable.py .
COPY annotate.html .
COPY med_names.txt .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

COPY spottings.csv .

RUN mkdir -p /data/annotations /data/cropped /images

ENV DATA_DIR=/data
ENV IMAGES_DIR=/images
ENV PORT=8090

EXPOSE 8090

ENTRYPOINT ["./entrypoint.sh"]