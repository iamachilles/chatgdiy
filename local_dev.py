import sys
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PORT = 3000
HOST = '127.0.0.1'  # localhost only, more secure than 0.0.0.0

try:
    # Add the project root to Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    logger.info("Python path updated")

    # Load environment variables
    load_dotenv()
    logger.info("Environment variables loaded")

    # Import the Flask app
    from api.app import app
    logger.info("Flask app imported successfully")

    if __name__ == '__main__':
        logger.info(f"Starting server on http://{HOST}:{PORT}")
        # Disable reloader to prevent double execution
        app.run(
            host=HOST,
            port=PORT,
            debug=True,
            use_reloader=False
        )

except Exception as e:
    logger.error(f"Error during startup: {str(e)}", exc_info=True)
    sys.exit(1)