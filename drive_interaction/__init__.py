from __future__ import print_function
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

from googleapiclient import discovery

# For authorized data access, we need additional resources http and oauth2client
from httplib2 import Http
from oauth2client import file, client, tools

import os.path
import io


# If modifying these scopes, delete the file token.json.
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIAL = 'credential.json'


def uploadFiles():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file(
            'token.json', DRIVE_SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIAL, DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)
        
        # Upload the presentation to Google Drive
        presentation_file_metadata = {'name': 'presentation.pdf'}

        mimetype = 'application/pdf'
        presentation_media = MediaIoBaseUpload(open("./output.pdf",'rb'), mimetype=mimetype)

        presentation_file = service.files().create(body=presentation_file_metadata,
                                        media_body=presentation_media,
                                        fields='id').execute()
        print('Presentation File ID: %s' % presentation_file.get('id'))

        # Set the permissions of the uploaded file to public
        service.permissions().create(
            fileId=presentation_file['id'],
            body={'type': 'anyone', 'role': 'writer'},
            supportsAllDrives=True
        ).execute()

        presentation_file_link = "https://drive.google.com/file/d/" + presentation_file.get('id') + "/view?usp=sharing"

        # Upload the video to Google Drive
        video_file_metadata = {'name': 'video.mp4'}

        video_mimetype = 'video/*'
        video_media = MediaIoBaseUpload(open("./output.mp4",'rb'), mimetype=video_mimetype)

        video_file = service.files().create(body=video_file_metadata,
                                        media_body=video_media,
                                        fields='id').execute()
        print('Video File ID: %s' % video_file.get('id'))

        # Set the permissions of the uploaded file to public
        service.permissions().create(
            fileId=video_file['id'],
            body={'type': 'anyone', 'role': 'writer'},
            supportsAllDrives=True
        ).execute()

        video_file_link = "https://drive.google.com/file/d/" + video_file.get('id') + "/view?usp=sharing"



        # Call the Drive v3 API
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f'An error occurred: {error}')
        file = None

    return {'presentation_link': presentation_file_link, 'video_link': video_file_link}


if __name__ == "__main__":
    uploadFiles()
