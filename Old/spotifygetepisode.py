import os
from dotenv import load_dotenv
import requests
import base64
import re
import json

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
        return auth_response.json()['access_token']
    else:
        print(f"Authentication failed. Status code: {auth_response.status_code}")
        return None

def get_show_episodes(show_id, access_token, limit=50, offset=0):
    url = f'https://api.spotify.com/v1/shows/{show_id}/episodes'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    params = {
        'limit': limit,
        'offset': offset
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()['items']
    else:
        print(f"Failed to get episodes. Status code: {response.status_code}")
        return None


def get_episode_details(show_id, episode_id, access_token):
    url = f'https://api.spotify.com/v1/shows/{show_id}/episodes'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    params = {
        'limit': 50,
        'offset': 0
    }
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            episodes = response.json()['items']
            for episode in episodes:
                if episode['id'] == episode_id:
                    return episode
            if len(episodes) < 50:
                break
            params['offset'] += 50
        else:
            print(f"Failed to get episodes. Status code: {response.status_code}")
            return None
    print(f"Episode with ID {episode_id} not found in the show's episodes.")
    return None

def parse_timeline(description):
    # Try to find TIMELINE: or TIMELINE section
    timeline_section = re.search(r'(?:TIMELINE:?|TIMELINE\s?:)(.+?)(?=Les anciens épisodes|Vous pouvez|La musique du générique|On a cité|on a parlé|\Z)', description, re.DOTALL | re.IGNORECASE)
    if timeline_section:
        # Extract individual timeline entries
        timeline = re.findall(r'(\d{1,2}:\d{2}:\d{2})\s*:?\s*(.+?)(?=\d{1,2}:\d{2}:\d{2}|\Z)', timeline_section.group(1), re.DOTALL)
        return timeline
    return None

def get_episode_chapters(episode_details):
    if 'chapters' in episode_details:
        return episode_details['chapters']['items']
    else:
        return None

def get_show_details(show_id, access_token):
    url = f'https://api.spotify.com/v1/shows/{show_id}'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get show details. Status code: {response.status_code}")
        return None

def find_episode_by_number(episodes, target_number):
    for episode in episodes:
        # Assuming the episode number is at the start of the name, like "#200 - ..."
        if episode['name'].startswith(f"#{target_number}"):
            return episode
    return None

def save_timeline_to_file(timeline, episode_number, episode_name):
    filename = f"timeline_episode_{episode_number}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Timeline for Episode #{episode_number}: {episode_name}\n\n")
        for time, description in timeline:
            # Clean up the description
            description = re.sub(r'\s+', ' ', description.strip())
            f.write(f"{time}: {description}\n")
    print(f"Timeline saved to {filename}")


# Main execution
if __name__ == "__main__":
    SHOW_ID = os.getenv("SPOTIFY_SHOW_ID")
    if not SHOW_ID:
        print("Error: SPOTIFY_SHOW_ID not found in .env file")
        exit(1)

    TARGET_EPISODE = 250  # CHANGE THE TARGET EPISODE HERE
    access_token = get_access_token()
    
    if access_token:
        all_episodes = []
        offset = 0
        while True:
            episodes = get_show_episodes(SHOW_ID, access_token, offset=offset)
            if not episodes:
                break
            all_episodes.extend(episodes)
            if len(episodes) < 50:  # If we got less than 50 episodes, we've reached the end
                break
            offset += 50

        target_episode = find_episode_by_number(all_episodes, TARGET_EPISODE)
        
        if target_episode:
            print(f"Found episode #{TARGET_EPISODE}: {target_episode['name']}")
            print(f"Episode ID: {target_episode['id']}")
            episode_details = get_episode_details(SHOW_ID, target_episode['id'], access_token)
            if episode_details:
                print("Episode details:")
                print(f"Name: {episode_details['name']}")
                print(f"Release date: {episode_details['release_date']}")
                print(f"Duration: {episode_details['duration_ms']} ms")
                
                timeline = parse_timeline(episode_details['description'])
                if timeline:
                    print("\nTimeline:")
                    for time, description in timeline:
                        # Clean up the description
                        description = re.sub(r'\s+', ' ', description.strip())
                        print(f"{time}: {description}")
                    
                    # Save timeline to file
                    save_timeline_to_file(timeline, TARGET_EPISODE, episode_details['name'])
                else:
                    print("\nNo timeline found in the description.")
            else:
                print("Couldn't get episode details.")
        else:
            print(f"Could not find episode #{TARGET_EPISODE}")
    else:
        print("Failed to obtain access token. Cannot proceed.")