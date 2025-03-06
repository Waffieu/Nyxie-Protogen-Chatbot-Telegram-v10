import os
import sys
import subprocess
import asyncio
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_startup.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("bot_starter")

async def main():
    logger.info("Starting bot initialization process...")
    
    # Load environment variables
    load_dotenv()
    
    # First check environment variables
    logger.info("Checking environment variables...")
    env_check = subprocess.run(
        [sys.executable, "environment_checker.py"],
        capture_output=True,
        text=True
    )
    
    if env_check.returncode != 0:
        logger.critical("Environment check failed. Please fix the issues above.")
        print(env_check.stdout)
        print(env_check.stderr)
        return False
    
    # Start the bot
    logger.info("Starting bot...")
    try:
        bot_process = subprocess.run(
            [sys.executable, "bot.py"],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.critical(f"Bot process exited with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        return True
    except Exception as e:
        logger.critical(f"Failed to start bot: {e}")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if not result:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Startup process interrupted by user")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        sys.exit(1)
