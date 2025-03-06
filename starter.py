import os
import sys
import subprocess
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('starter_log.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("bot_starter")

def main():
    """Start the bot as a separate process"""
    logger.info("Starting bot process...")
    
    try:
        # Use sys.executable to ensure we use the same Python interpreter
        process = subprocess.Popen(
            [sys.executable, "bot.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Bot process started with PID: {process.pid}")
        
        # Wait for process to complete and capture output
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Bot exited with error code: {process.returncode}")
            logger.error(f"Standard output:\n{stdout}")
            logger.error(f"Error output:\n{stderr}")
        else:
            logger.info("Bot process completed successfully")
            
        return process.returncode
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Starter process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in starter script: {e}")
        sys.exit(1)
