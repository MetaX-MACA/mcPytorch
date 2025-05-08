import requests
import multiprocessing
from pathlib import Path

import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class APIClient:
    def __init__(self, url='http://10.6.26.44:5000/api/gpu/perf/metrics'):
        self.url = url

    def post_data(self, data):
        response = requests.post(self.url, json=data)
        print(self.url)
        print(data)
        print(response.text)
        return response.status_code

    @timer_decorator
    def post_file(self, filename):
        files = {'file': open(filename, 'r')}
        response = requests.post(self.url, files=files)
        status_code = response.status_code
        print(response.status_code)
        if status_code == 200:
            print(response.text)
        return response.status_code

    def post_data_multiprocessing(self, data_list):
        with multiprocessing.Pool() as pool:
            results = pool.map(self.post_data, data_list)
        return results

