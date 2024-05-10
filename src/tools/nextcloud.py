import os
from dotenv import find_dotenv, load_dotenv
from webdav3.client import Client

class NextCloud:
    def __init__(self):
        load_dotenv(find_dotenv())
        options = {
            'webdav_hostname': os.getenv('STORAGE_URL'),
            'webdav_login': os.getenv('STORAGE_LOGIN'),
            'webdav_password': os.getenv('STORAGE_PASSWORD')
        }
        self.client = Client(options)
    
    def upload(self, remote_cat: str, file_path: str):
        self.client.upload(f'thesis/{remote_cat}', file_path)

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    options = {
        'webdav_hostname': os.getenv('STORAGE_URL'),
        'webdav_login': os.getenv('STORAGE_LOGIN'),
        'webdav_password': os.getenv('STORAGE_PASSWORD')
    }

    client = Client(options)

    client.upload("thesis/demo/a.mp3", "/home/mszawerd/Learn/thesis/musical-generative-models-conditioning/notebooks/audiocraft/musicgen_out.wav")
