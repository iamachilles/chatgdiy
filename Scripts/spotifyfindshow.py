"""
Spotify Show ID Finder

This script helps you find the Spotify ID for a specific podcast (show).

Usage:
python spotifyfindshow.py

Before running:
1. Ensure you have a .env file in the same directory with your Spotify API credentials:
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret

2. Install required libraries:
   pip install python-dotenv requests

The script will prompt you to enter the name of the podcast you're looking for.
It will then display the Spotify ID for that show, which you can use in other scripts
or add to your .env file as SPOTIFY_SHOW_ID.

Note: This script requires an active internet connection to access the Spotify API.

For more detailed instructions or troubleshooting, refer to the README.md file in the project repository.
"""


import os
from dotenv import load_dotenv
import base64
import requests

# Load environment variables
load_dotenv()

def get_access_token():
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("Error: SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not found in .env file")
        return None

    auth_url = 'https://accounts.spotify.com/api/token'
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    auth_data = {'grant_type': 'client_credentials'}
    auth_headers = {'Authorization': f'Basic {auth_header}'}

    auth_response = requests.post(auth_url, headers=auth_headers, data=auth_data)
    if auth_response.status_code == 200:
        access_token = auth_response.json()['access_token']
        print("Authentication successful. Access token obtained.")
        return access_token
    else:
        print(f"Authentication failed. Status code: {auth_response.status_code}")
        print(f"Error message: {auth_response.text}")
        return None

def get_show_id(podcast_name, access_token):
    encoded_name = requests.utils.quote(podcast_name)
    search_url = f"https://api.spotify.com/v1/search?q={encoded_name}&type=show&market=US"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        results = response.json()
        
        if results['shows']['items']:
            show_id = results['shows']['items'][0]['id']
            show_name = results['shows']['items'][0]['name']
            print(f"Found show: {show_name}")
            print(f"Show ID: {show_id}")
            return show_id
        else:
            print(f"No shows found with the name '{podcast_name}'")
            return None
    else:
        print(f"Error searching for show. Status code: {response.status_code}")
        print(f"Error message: {response.text}")
        return None

# Main execution
if __name__ == "__main__":
    # Get access token
    access_token = get_access_token()
    
    if access_token:
        # Search for podcast
        podcast_name = "The Diary of a CEO"  # Replace with the actual podcast name
        show_id = get_show_id(podcast_name, access_token)
        
        if show_id:
            print(f"You can use this Show ID: {show_id} for further API calls.")
    else:
        print("Failed to obtain access token. Cannot proceed.")