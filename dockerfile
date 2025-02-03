FROM python:3.13-slim

WORKDIR /app

COPY app.py /app/
COPY naive_bayes_model.joblib /app/
COPY label_encoders.joblib /app/

COPY static /app/static
COPY templates /app/templates

COPY modules.txt /app/modules.txt
RUN pip install --no-cache-dir -r /app/modules.txt

EXPOSE 80

CMD ["python3", "app.py"]
