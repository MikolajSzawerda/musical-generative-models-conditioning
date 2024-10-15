import os
from dotenv import find_dotenv, load_dotenv
from webdav3.client import Client
import logging
import argparse
from rich.console import Console

def _validate_env():
    if not os.getenv('STORAGE_URL'):
        raise Exception("no storage url provided")
    if not os.getenv('STORAGE_LOGIN'):
        raise Exception("no storage login provided")
    if not os.getenv('STORAGE_PASSWORD'):
        raise Exception("no storage password provided")


class NextCloud:
    def __init__(self, logger=None):
        logging.basicConfig(level=logging.INFO)
        load_dotenv(find_dotenv())
        _validate_env()
        self.client = Client({
            'webdav_hostname': os.getenv('STORAGE_URL'),
            'webdav_login': os.getenv('STORAGE_LOGIN'),
            'webdav_password': os.getenv('STORAGE_PASSWORD')
        })
        self.log: Console = logger
    def _log(self, *args, **kwargs):
        if self.log:
            self.log.print(*args, **kwargs)
    def _create_remote_directory(self, remote_path):
        if self.client.check(remote_path):
            return
        self._log("[yellow]Have to create recursivly directories")
        directories = remote_path.strip('/').split('/')
        current_dir = ''
        for directory in directories:
            current_dir += f'{directory}/'
            if not self.client.check(current_dir):
                self.client.mkdir(current_dir)

    def upload(self, src: str, dest: str):
        path = f'thesis/storage/{dest}'
        self._create_remote_directory(path)
        if not self.client.push(path, src):
            self._log("[yellow]No files to upload")
        self._log("[green]Upload finished :smiley:")

    def download(self, src: str, dest: str):
        path = f'thesis/storage/{src}'
        if not self.client.pull(path, dest):
            self._log("[yellow]No files to download")
        self._log("[green]Download finished :smiley:")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("method")
    parser.add_argument("src")
    parser.add_argument("dest", nargs='?')
    args = parser.parse_args()
    load_dotenv(find_dotenv())
    _validate_env()
    options = {
        'webdav_hostname': os.getenv('STORAGE_URL'),
        'webdav_login': os.getenv('STORAGE_LOGIN'),
        'webdav_password': os.getenv('STORAGE_PASSWORD')
    }
    nc = NextCloud(Console())
    if args.method == 'push':
        nc.upload(args.src, args.dest if args.dest else args.src)
    if args.method == 'pull':
        nc.download(args.src, args.dest if args.dest else args.src)
