import os
import sys
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from telegram.ext import ApplicationBuilder
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("environment_checker")

async def check_telegram_token(token):
    """Test the Telegram token with a simple API call"""
    try:
        app = ApplicationBuilder().token(token).build()
        bot_info = await app.bot.get_me()
        logger.info(f"✅ Telegram token verified successfully (Bot: {bot_info.username})")
        return True
    except Exception as e:
        logger.error(f"❌ Invalid TELEGRAM_TOKEN: {str(e)}")
        return False

def check_gemini_api_key(api_key):
    """Test the Gemini API key with a simple request"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content("Hello")
        logger.info(f"✅ Gemini API key verified successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Invalid GEMINI_API_KEY: {str(e)}")
        return False

async def main():
    # Load environment variables
    load_dotenv()
    logger.info("Checking environment variables...")
    
    # Required environment variables
    required_vars = {
        "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")
    }
    
    # Check if variables exist
    all_vars_present = True
    for var_name, var_value in required_vars.items():
        if not var_value:
            logger.error(f"❌ Missing {var_name}")
            all_vars_present = False
        else:
            logger.info(f"✅ Found {var_name}")
    
    if not all_vars_present:
        logger.error("Some required environment variables are missing.")
        return False
    
    # Check if the values are valid
    telegram_valid = await check_telegram_token(required_vars["TELEGRAM_TOKEN"])
    gemini_valid = check_gemini_api_key(required_vars["GEMINI_API_KEY"])
    
    if telegram_valid and gemini_valid:
        logger.info("✅ All environment variables are valid!")
        return True
    else:
        logger.error("❌ Some environment variables are invalid.")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    if not result:
        sys.exit(1)
    else:
        print("\nYou can now run the bot with: python bot.py")
