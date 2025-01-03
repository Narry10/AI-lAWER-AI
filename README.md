﻿# Law QA API

This project provides a FastAPI-based web service for answering legal questions using a Hugging Face model.

## Requirements

- Python 3.12
- `uvicorn==0.32.0`
- `fastapi==0.115.2`
- `python-dotenv==1.0.1`
- `huggingface-hub==0.25.2`
- `transformers==4.45.2`
- `torch==2.4.1`

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/Captone2-legal-support/services-ai.git
    cd services-ai
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the `env` directory with your Hugging Face token:
    ```dotenv
    HF_TOKEN='your_huggingface_token'
    ```

## Using Docker

1. Build the Docker image:
    ```sh
    docker build -t law-qa-api .
    ```

2. Run the Docker container:
    ```sh
    docker run -d -p 8000:8000 --name law-qa-api-container law-qa-api
    ```
3. The API will be available at `http://127.0.0.1:8000`.
 
## Running the API

1. Start the FastAPI server:
    ```sh
    uvicorn main:app --reload
    ```

2. The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- `GET /`: Returns a welcome message.
- `GET /get_answer`: Accepts a `question` parameter and returns the answer.

## Example

Request:
```sh
curl -X 'GET' \
  'http://127.0.0.1:8000/get_answer?question=Các%20biện%20pháp%20giải%20quyết%20tranh%20chấp%20đất%20đai%20theo%20pháp%20luật%20Việt%20Nam%20là%20gì?' \
  -H 'accept: application/json'

