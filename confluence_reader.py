import os
from onyx_confluence import OnyxConfluence
from atlassian import Confluence
import os
import sys
import requests
from dotenv import load_dotenv, find_dotenv
from config import (CONFLUENCE_API_TOKEN, CONFLUENCE_SPACE_URL)

import shutil
from convert_tools import (process_non_text_document)
#import logging
from bs4 import BeautifulSoup
from injest import preprocess_and_split_text

# Initialize the Confluence clients
client = OnyxConfluence(
    url=CONFLUENCE_SPACE_URL,
    cloud=False,
    token=CONFLUENCE_API_TOKEN
)

confluence_client = Confluence(
    url=CONFLUENCE_SPACE_URL,
    cloud=False,
    token=CONFLUENCE_API_TOKEN
)
    
def preprocess_html(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # Get text content
    text_content = soup.get_text(separator=' ', strip=True)

    # Further cleaning (optional)
    # Remove excessive whitespace
    text_content = ' '.join(text_content.split())

    return text_content


# Function to save content to a file
def save_to_file(file_name, content, the_output_folder , as_binary=False ):

    file_name = file_name.replace(' ', '_').lower()
    file_path = os.path.join(the_output_folder, file_name)
    print(f"Going to save to {file_path}")
    try:
        if as_binary == True:
            with open(file_path, "wb") as file:
                file.write(content)
        else:
            content = content.lower()
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)

        print(f"Saved to {file_path}")
    except Exception as e:
        raise e
        #print(f"An error occurred while saving {file_path}: {e}")
    return file_path

def get_link(url , api_token = CONFLUENCE_API_TOKEN):
    # Set up the authentication
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
       }

    try:
        # Make the GET request to the Confluence API
        print(f"Going to retrieve {url}")
        response = requests.get(url, headers=headers, verify = False)

        # Check if the request was successful
        if response.status_code == 200:
            print(f"Got ok responce : {response.status_code}")
            print( f"Attachment downloaded")
            return response
        else:
            print(f"Failed to fetch {url}: {response.status_code}")
            return None
    except requests.HTTPError as e:
        if  response.status_code == 404:
            print(f"Attachment not found at {url}")  # noqa: T201
            pass
        else:
            raise e
    except Exception as e:
        raise e



# Function to retrieve and save attachments
def get_and_save_attachments(page_id, page_title):
    all_atachments_text = ""
    try:
        attachments = confluence_client.get_attachments_from_content(page_id)
        for attachment in attachments["results"]:
            print(f"Attachement: {attachment} ")
            attachment_title = attachment['title']
            if not attachment_title:
                attachment_title = attachment["id"]  # if the attachment has no title, use attachment_id as a filename

            attachment_file_name = f"{page_title}_{attachment_title}".replace(' ', '_').lower()
            attachment_name = attachment["_links"]["download"][1:] # strip leading '/'
            print(f"Attachement id {attachment['id']} file name {attachment_file_name}   --> {attachment_name}")

            download_link = CONFLUENCE_SPACE_URL + attachment_name
            r = get_link(f"{download_link}")
            if r:
                attachment_file_name = save_to_file(attachment_file_name, r.content, the_output_folder = attachements_output_folder, as_binary=True)
                text = process_non_text_document(attachment_file_name)
                os.remove(attachment_file_name)
                if text:
                    all_atachments_text += " " + text
                    # scraped_text_path = attachment_file_name.rsplit('.', 1)[0] + ".txt"
                    # scraped_file_name_only = os.path.basename(scraped_text_path)
                    print(f"Scraped from {attachment_file_name}")
                          ##-> to {scraped_file_name_only}")
                    # save_to_file(scraped_file_name_only, text , the_output_folder= output_folder)
                    
        return all_atachments_text
    
    except requests.HTTPError as e:
        if r.response.status_code == 404:
            print(f"Attachment not found at {download_link}")  # noqa: T201
            pass
        else:
            raise e
    except Exception as e:
        #raise e
        print(f"An error occurred while fetching attachments: {e}")


# Function to retrieve and save the content of a page
def get_and_save_page_content(page_id, page_title):
    try:
        print(f"get_and_save_page_content : {page_title}")
        page = client.get_page_by_id(page_id, expand='body.storage')
        html_content = page['body']['storage']['value']
        text_content = preprocess_html(html_content)
        #html_file_name = f"{page_title}.html"
        text_file_name = f"{page_title}.txt"

        #save_to_file(html_file_name, html_content)
        text_file_name = save_to_file(text_file_name, text_content , the_output_folder= output_folder)

        #print(f"Saved page content as HTML: {html_file_name}")
        print(f"Saved page content as text: {text_file_name}")
        return text_content
    except NotADirectoryError:
            raise NotADirectoryError("Verify if directory path is correct and/or if directory exists")
    except PermissionError:
        raise PermissionError(
            "Directory found, but there is a problem with saving file to this directory. Check directory permissions")
    except Exception as e:
        #raise e
        print(f"An error occurred while fetching page content: {e}")
        

# Function to retrieve and save space map
def get_and_save_space(space_key , space_name , tokenizer):
    chunks = []    
    global output_folder
    global attachements_output_folder 
    print(f"space: {space_key} , name {space_name}")
    output_folder = f"OnyxData/{space_name}".replace(' ', '_')
    attachements_output_folder = f"OnyxDataAttachements/{space_name}".replace(' ', '_')
    print(f"==> output_folder {output_folder} , attachements_output_folder {attachements_output_folder}")
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    if os.path.isdir(attachements_output_folder):
        shutil.rmtree(attachements_output_folder)
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
    os.makedirs(attachements_output_folder, exist_ok=True)  # Create output folder for attachements if it doesn't exist
    
    try:
        pages = client.get_all_pages_from_space(space_key, start=0, limit=100)
        space_map_content = f"Space Map for Space Key: {space_key}\n\n"
        #for page in pages[:5]:
        for page in pages:
            page_title = page['title']
            page_id = page['id']
            space_map_content += f"Title: {page_title}, ID: {page_id}\n"
            text = get_and_save_page_content(page_id, page_title)
            if text:
                chunks.extend(preprocess_and_split_text(text, tokenizer ))
                #chunks.extend([text[i:i+chunk_size] for i in range(0, len(text), chunk_size)])
            text = get_and_save_attachments(page_id, page_title)
            if text:
                chunks.extend(preprocess_and_split_text(text, tokenizer))
                #chunks.extend([text[i:i+chunk_size] for i in range(0, len(text), chunk_size)])
        file_name = f"{space_key}_space_map.txt"
        file_name = save_to_file(file_name, space_map_content , the_output_folder = output_folder )
        print(f"Saved space map: {file_name}")
        return chunks
    except Exception as e:
        print(f"An error occurred while fetching the space: {e}")




