import os

LIBRARY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(LIBRARY_ROOT)
DATAFILES_ROOT = os.path.join(PROJECT_ROOT, 'sdss_data')
