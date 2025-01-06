import os
import sys
import logging
from contextlib import contextmanager

@contextmanager
def suppress_all_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_log_level = logging.getLogger().getEffectiveLevel()
    
    null_file = open(os.devnull, 'w')
    
    try:
        sys.stdout = null_file
        sys.stderr = null_file
        
        logging.disable(logging.CRITICAL)
        
        yield 
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.disable(logging.DEBUG)
        null_file.close()



