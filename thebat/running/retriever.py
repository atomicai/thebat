from thebat.running.mask import IR
import openai
import os
from typing import Union, List
import random
import backoff


class IBM25Runner(IR):
    """
    Retrieves the passages from index using BM25 algo.
    Relies on WeaviateDocStore or ElasticSearch running service.
    """

    def __init__(self, store=None):
        self.client = None
        self.store = store

    @backoff.on_exception(backoff.expo, exception=openai.error.RateLimitError)
    def generate_answer(prompt: str, model: str = None):
        if model is None:
            models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]
            idx = random.randint(0, 4)
            if idx >= 1 and idx <= 2:
                pos = 1
            elif idx >= 3 and idx <= 4:
                pos = 2
            else:
                pos = 0
            model = models[pos]
        if random.randint(0, 1) > 0:
            prompt += " Ответь кратко."
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You’re a kind helpful assistant"}, {"role": "user", "content": prompt}],
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_tokens=1024,
            temperature=1.0,
        )
        try:
            return response["choices"][0]["message"]["content"].strip()
        except IndexError:
            return ""

    def retrieve_topk(self, query: Union[str, List[str]], **kwargs):
        pass


__all__ = ["IBM25Runner"]
