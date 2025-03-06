import psutil
import os
import logging
from datetime import datetime

class SystemMonitor:
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger(__name__)
        
    async def system_status(self, update, context):
        try:
            # System performance metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            uptime = datetime.now() - datetime.fromtimestamp(psutil.boot_time())
            
            # Bot-specific metrics
            code_size = sum(os.path.getsize(f) for f in os.listdir('.') if f.endswith('.py'))
            active_plugins = len(self.bot.handlers)
            
            response = (
                f"🖥️ System Status:\n"
                f"CPU Usage: {cpu_usage}%\n"
                f"Memory Usage: {mem.percent}%\n"
                f"Disk Usage: {disk.percent}%\n"
                f"Uptime: {str(uptime).split('.')[0]}\n\n"
                f"🤖 Bot Metrics:\n"
                f"Codebase Size: {code_size/1024:.2f} KB\n"
                f"Active Modules: {active_plugins}\n"
                f"Messages Processed: {self.bot.message_count}"
            )
            
            await update.message.reply_text(response)
            self.logger.info('System status reported')
            
        except Exception as e:
            self.logger.error(f"System status error: {str(e)}")
            await update.message.reply_text("❌ Could not retrieve system status")

    async def view_code(self, update, context):
        try:
            file_name = context.args[0] if context.args else 'system_monitor.py'
            with open(file_name, 'r') as f:
                code = f.read()
            await update.message.reply_text(f"📄 {file_name}:\n```python\n{code[:3000]}\n```", parse_mode='Markdown')
            self.logger.info(f"Code viewed: {file_name}")
        except Exception as e:
            self.logger.error(f"Code view error: {str(e)}")
            await update.message.reply_text("❌ Could not retrieve code file")

    def register_commands(self):
        self.bot.handlers.extend([
            (['system_status'], self.system_status, {'admin_only': True}),
            (['view_code'], self.view_code, {'admin_only': True})
        ])
        self.logger.info("Self-awareness commands registered")