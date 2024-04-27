import os

from dotenv import find_dotenv, load_dotenv
from webdav3.client import Client

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    options = {
        'webdav_hostname': os.getenv('STORAGE_URL'),
        'webdav_login': os.getenv('STORAGE_LOGIN'),
        'webdav_password': os.getenv('STORAGE_PASSWORD')
    }

    client = Client(options)

    client.download_sync("Storage-Share.png", "file.png")
