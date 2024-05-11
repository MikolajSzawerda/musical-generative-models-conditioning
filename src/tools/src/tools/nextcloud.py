import os
from dotenv import find_dotenv, load_dotenv
from webdav3.client import Client
import logging

class NextCloud:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        load_dotenv(find_dotenv())
        options = {
            'webdav_hostname': os.getenv('STORAGE_URL'),
            'webdav_login': os.getenv('STORAGE_LOGIN'),
            'webdav_password': os.getenv('STORAGE_PASSWORD')
        }
        self.client = Client(options)
    
    def upload(self, remote_cat: str, file_path: str):
        self.client.upload(f'thesis/{remote_cat}', file_path)
    
    def sync(self, remote_directory, local_directory, subdirs=tuple()):
        def _sync(rd, ld):
            self.client.push(rd, ld)
            self.client.pull(rd, ld)
        logging.info(f'Start syncing {remote_directory} <-> {local_directory}')
        _sync(f'thesis/{remote_directory}', local_directory)
        content = self.client.list(f'thesis/{remote_directory}')
        for subdir in subdirs:
            logging.info(f'Start syncing subdirectory {subdir}')
            
            p = f'thesis/{remote_directory}/{subdir}'
            p_l = f'{local_directory}/{subdir}'
            if f'{subdir}/' not in content:
                logging.info(f"Created remote subdir {subdir}")
                self.client.mkdir(p)
            sub_content = self.client.list(p)
            _sync(f'thesis/{remote_directory}/{subdir}', p_l)
            for f in os.listdir(p_l):
                if f not in sub_content:
                    logging.info(f'Uploaded file {f} to remote')
                    self.client.upload_sync(f'{p}/{f}', f'{p_l}/{f}')
            
            
            
        

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    options = {
        'webdav_hostname': os.getenv('STORAGE_URL'),
        'webdav_login': os.getenv('STORAGE_LOGIN'),
        'webdav_password': os.getenv('STORAGE_PASSWORD')
    }
    nc = NextCloud()
