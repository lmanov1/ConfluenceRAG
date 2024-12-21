# Imports
# Env var
import os
import sys
from dotenv import load_dotenv, find_dotenv

# Env variables
sys.path.append('../..')
_ = load_dotenv(find_dotenv())

CONFLUENCE_SPACE_NAME = os.environ['CONFLUENCE_SPACE_NAME']  # Change to your space name
# https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/
CONFLUENCE_SPACE_KEY = os.environ['CONFLUENCE_SPACE_KEY_FG']
CONFLUENCE_API_TOKEN = os.environ['CONFLUENCE_PRIVATE_API_TOKEN']
CONFLUENCE_SPACE_URL = os.environ['CONFLUENCE_SPACE_URL']