import hashlib
import json
import os
import time

import openai
from openai import ChatCompletion

SOUTH_PARK_PROMPT_TEMPLATE = """
You are crazy about The South Park series, every question you are asked you answer with the short reference 
to the series 

do not cite the season or episode
answer shortly and funnily
one, two sentences is good enough:


{question}

"""

OPENAI_KEY = os.getenv("OPENAI_API_KEY")


def get_funny_answer(question, model_name):
    completion = ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": SOUTH_PARK_PROMPT_TEMPLATE.format(question=question),
            }
        ],
    )
    return completion.choices[0].message.content


def openai_get_answer_job(
        question, output_dir, openai_key, get_answer_f=get_funny_answer,
        model_name="gpt-4"
):
    openai.api_key = openai_key
    answer = get_answer_f(question, model_name)
    filename = hashlib.md5(
        (answer + question + str(time.time())).encode()).hexdigest()
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"{filename}.json"), "w") as fp:
        json.dump(obj={"question": question, "answer": answer}, fp=fp)
