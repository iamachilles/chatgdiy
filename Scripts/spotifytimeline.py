"""
Spotify Episode Timeline Extractor

This script fetches the timeline for a specific episode of a Spotify podcast.

Usage:
python spotifygetepisode.py --episode <episode_number>

Make sure to set up your .env file with the following variables:
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_SHOW_ID=your_show_id

For more detailed instructions, see the README.md file.
"""


import os
import re
import json
import asyncio
import logging
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv
import aiohttp
import base64
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
AUTH_URL = 'https://accounts.spotify.com/api/token'
API_BASE_URL = 'https://api.spotify.com/v1'
TIMELINE_REGEX = r'(?:TIMELINE:?|TIMELINE\s?:)(.+?)(?=Les anciens épisodes|Vous pouvez|La musique du générique|cité|parlé|SHOW NOTES|\Z)'
TIMELINE_ENTRY_REGEX = r'(\d{1,2}:\d{2}:\d{2})\s*:?\s*(.+?)(?=\d{1,2}:\d{2}:\d{2}|\Z)'

class SpotifyAPIError(Exception):
    """Custom exception for Spotify API errors."""
    pass

class SpotifyAPI:
    def __init__(self, client_id: str, client_secret: str):
        """Initialize the SpotifyAPI with client credentials."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None

    async def get_access_token(self) -> None:
        """Authenticate with Spotify and obtain an access token."""
        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        auth_data = {'grant_type': 'client_credentials'}
        auth_headers = {'Authorization': f'Basic {auth_header}'}

        async with aiohttp.ClientSession() as session:
            async with session.post(AUTH_URL, headers=auth_headers, data=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self.access_token = data['access_token']
                else:
                    raise SpotifyAPIError(f"Authentication failed. Status code: {response.status}")

    async def get_show_episodes(self, show_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Fetch episodes for a given show ID."""
        url = f'{API_BASE_URL}/shows/{show_id}/episodes'
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        params = {'limit': limit, 'offset': offset}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['items']
                else:
                    raise SpotifyAPIError(f"Failed to get episodes. Status code: {response.status}")

    async def get_episode_details(self, show_id: str, episode_id: str) -> Optional[Dict[str, Any]]:
        """Fetch details for a specific episode within a show."""
        url = f'{API_BASE_URL}/shows/{show_id}/episodes'
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        params = {'limit': 50, 'offset': 0}

        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        episodes = data['items']
                        for episode in episodes:
                            if episode['id'] == episode_id:
                                return episode
                        if len(episodes) < 50:
                            break
                        params['offset'] += 50
                    else:
                        raise SpotifyAPIError(f"Failed to get episodes. Status code: {response.status}")

        logger.warning(f"Episode with ID {episode_id} not found in the show's episodes.")
        return None

class EpisodeProcessor:
    @staticmethod
    def parse_timeline(description: str) -> Optional[List[Tuple[str, str]]]:
        """Extract timeline entries from the episode description."""
        timeline_section = re.search(TIMELINE_REGEX, description, re.DOTALL | re.IGNORECASE)
        if timeline_section:
            return re.findall(TIMELINE_ENTRY_REGEX, timeline_section.group(1), re.DOTALL)
        return None

    @staticmethod
    def find_episode_by_number(episodes: List[Dict[str, Any]], target_number: int) -> Optional[Dict[str, Any]]:
        """Find an episode by its number in the episode list."""
        return next((episode for episode in episodes if episode['name'].startswith(f"#{target_number}")), None)

    @staticmethod
    def save_timeline_to_file(timeline: List[Tuple[str, str]], episode_number: int, episode_name: str) -> None:
        """Save the parsed timeline to a text file."""
        filename = f"timeline_episode_{episode_number}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Timeline for Episode #{episode_number}: {episode_name}\n\n")
            for time, description in timeline:
                description = re.sub(r'\s+', ' ', description.strip())
                f.write(f"{time}: {description}\n")
        logger.info(f"Timeline saved to {filename}")

async def main(target_episode: int):
    """Main function to orchestrate the episode fetching and timeline extraction process."""
    load_dotenv()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    show_id = os.getenv("SPOTIFY_SHOW_ID")

    if not all([client_id, client_secret, show_id]):
        raise ValueError("SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, or SPOTIFY_SHOW_ID not found in .env file")

    api = SpotifyAPI(client_id, client_secret)
    await api.get_access_token()

    all_episodes = []
    offset = 0
    while True:
        episodes = await api.get_show_episodes(show_id, offset=offset)
        if not episodes:
            break
        all_episodes.extend(episodes)
        if len(episodes) < 50:
            break
        offset += 50

    target_episode_data = EpisodeProcessor.find_episode_by_number(all_episodes, target_episode)
    
    if target_episode_data:
        logger.info(f"Found episode #{target_episode}: {target_episode_data['name']}")
        logger.info(f"Episode ID: {target_episode_data['id']}")
        
        episode_details = await api.get_episode_details(show_id, target_episode_data['id'])
        if episode_details:
            logger.info("Episode details:")
            logger.info(f"Name: {episode_details['name']}")
            logger.info(f"Release date: {episode_details['release_date']}")
            logger.info(f"Duration: {episode_details['duration_ms']} ms")
            
            timeline = EpisodeProcessor.parse_timeline(episode_details['description'])
            if timeline:
                logger.info("\nTimeline:")
                for time, description in timeline:
                    description = re.sub(r'\s+', ' ', description.strip())
                    logger.info(f"{time}: {description}")
                
                EpisodeProcessor.save_timeline_to_file(timeline, target_episode, episode_details['name'])
            else:
                logger.warning("\nNo timeline found in the description.")
        else:
            logger.error("Couldn't get episode details.")
    else:
        logger.error(f"Could not find episode #{target_episode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Spotify podcast episode timeline.")
    parser.add_argument("--episode", type=int, required=True, help="Target episode number")
    args = parser.parse_args()

    asyncio.run(main(args.episode))