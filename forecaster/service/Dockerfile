
# 
FROM python:3.11.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt
# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

ARG TOKEN_DSIPTS
ENV TOKEN_DSIPTS=${TOKEN_DSIPTS}

RUN pip install --force dsipts==1.1.0 --index-url ${TOKEN_DSIPTS}

# 

copy ./app /code/app

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
