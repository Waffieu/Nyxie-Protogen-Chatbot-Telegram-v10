"""
Integration module for connecting the FreeWillSystem to the bot's main flow.
This allows for autonomous behavior, initiative taking, and personality evolution.
"""

import logging
import random
import asyncio
from datetime import datetime
import re
from free_will import FreeWillSystem
import os 

logger = logging.getLogger(__name__)

class FreeWillIntegrator:
    """
    Integrates free will capabilities into the bot's decision making process
    """
    
    def __init__(self, free_will_system=None, self_awareness=None, root_dir=None):
        """Initialize the free will integrator with references to key systems"""
        try:
            self.root_dir = root_dir or os.path.dirname(os.path.abspath(__file__))
            
            # Initialize free will system with proper error handling
            if free_will_system is None:
                try:
                    logger.info("Creating new FreeWillSystem instance")
                    self.free_will = FreeWillSystem(bot_self_awareness=self_awareness, root_dir=self.root_dir)
                except Exception as fw_error:
                    logger.error(f"Error creating FreeWillSystem: {fw_error}")
                    # Create minimal fallback version
                    from types import SimpleNamespace
                    self.free_will = SimpleNamespace()
                    self.free_will.analyze_message = lambda *args, **kwargs: {"topics": [], "emotional_tone": "neutral"}
                    self.free_will.augment_prompt_with_personality = lambda x: x
                    self.free_will.pursue_goal = lambda: None
                    self.free_will.generate_autonomous_thought = lambda: None
                    self.free_will.set_autonomous_goal = lambda *args, **kwargs: None
                    logger.warning("Using fallback FreeWillSystem due to initialization error")
            else:
                self.free_will = free_will_system
                
            self.self_awareness = self_awareness
            self.last_autonomous_thought = datetime.now()
            self.thought_interval = 300  # Generate autonomous thoughts every 5 minutes
            self.initiative_chance = 0.25  # Base chance of taking initiative in a conversation
            self.last_initiatives = {}  # Track initiatives by user to avoid repetition
            
            # Ensure free will system data file exists
            if hasattr(self.free_will, 'save_data'):
                try:
                    self.free_will.save_data()
                    logger.info("Ensured free will data file exists on integrator initialization")
                except Exception as save_err:
                    logger.error(f"Could not ensure free will data file: {save_err}")
        except Exception as e:
            logger.error(f"Error in FreeWillIntegrator initialization: {e}", exc_info=True)
            # Set minimal defaults to allow the bot to continue functioning
            self.root_dir = root_dir or os.path.dirname(os.path.abspath(__file__))
            self.self_awareness = self_awareness
            self.last_autonomous_thought = datetime.now()
            self.thought_interval = 300
            self.initiative_chance = 0.25
            self.last_initiatives = {}
            
            # Minimal free will stub
            from types import SimpleNamespace
            self.free_will = SimpleNamespace()
            self.free_will.analyze_message = lambda *args, **kwargs: {"topics": [], "emotional_tone": "neutral"}
            self.free_will.augment_prompt_with_personality = lambda x: x
            self.free_will.pursue_goal = lambda: None
            self.free_will.generate_autonomous_thought = lambda: None
            self.free_will.set_autonomous_goal = lambda *args, **kwargs: None
    
    async def augment_message_processing(self, user_id, message_text, ai_prompt):
        """
        Augment the message processing with free will capabilities
        
        Args:
            user_id: The user's ID
            message_text: The user's message
            ai_prompt: The current AI prompt
        
        Returns:
            Modified AI prompt with personality enhancements
        """
        try:
            # Analyze message for potential autonomous responses
            analysis = self.free_will.analyze_message(message_text, user_id)
            
            # Record analysis in self-awareness if available
            if self.self_awareness:
                self.self_awareness.record_thought_process(
                    "free_will_analysis", 
                    f"Analyzed message: Topics={analysis['topics']}, Emotion={analysis['emotional_tone']}, Initiative={analysis['requires_initiative']}"
                )
            
            # Augment prompt with personality
            enhanced_prompt = self.free_will.augment_prompt_with_personality(ai_prompt)
            
            # Automatically check for goals - no user command needed
            if analysis.get("opportunity_for_goal", False):
                # Reduced chance (15%) to automatically set goals
                if random.random() < 0.15:
                    goal = self.free_will.set_autonomous_goal(base_topics=analysis["topics"], user_id=user_id)
                    if goal and self.self_awareness:
                        self.self_awareness.record_thought_process(
                            "goal_setting",
                            f"Auto-set new goal: {goal['description']}"
                        )
            
            # Check if we should generate an autonomous thought
            now = datetime.now()
            if (now - self.last_autonomous_thought).total_seconds() > self.thought_interval:
                thought = self.free_will.generate_autonomous_thought()
                self.last_autonomous_thought = now
                
                if thought and self.self_awareness:
                    self.self_awareness.record_thought_process(
                        "autonomous_thought",
                        f"Generated thought: {thought['type']} - {thought['content']}"
                    )
            
            # Check if we should pursue an existing goal
            # Auto-pursue with 10% chance
            if random.random() < 0.1:
                goal = self.free_will.pursue_goal()
                if goal and self.self_awareness:
                    self.self_awareness.record_thought_process(
                        "goal_pursuit",
                        f"Auto-pursued goal: {goal['description']} - Progress: {goal['progress']:.0%}"
                    )
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error in free will augmentation: {e}")
            return ai_prompt  # Return original prompt if there's an error
    
    async def generate_initiative_response(self, user_id, message_text):
        """
        Potentially generate an initiative-based response rather than just answering
        
        Args:
            user_id: The user ID
            message_text: The user's message
        
        Returns:
            Initiative dict if bot takes initiative, None otherwise
        """
        try:
            # Check when we last took initiative with this user
            last_time = self.last_initiatives.get(user_id, datetime.fromtimestamp(0))
            hours_since_last = (datetime.now() - last_time).total_seconds() / 3600
            
            # Base initiative chance is modified by time since last initiative
            # Higher chance if we haven't taken initiative recently
            modified_chance = self.initiative_chance
            if hours_since_last < 1:
                modified_chance *= 0.3  # Much lower chance if initiative taken in last hour
            elif hours_since_last > 12:
                modified_chance *= 1.5  # Higher chance if no initiative in 12+ hours
            
            # Check if we should take initiative
            if random.random() < modified_chance:
                initiative = self.free_will.generate_initiative(message_text, user_id)
                
                # If initiative generated, record it
                if initiative:
                    self.last_initiatives[user_id] = datetime.now()
                    
                    # Record in self-awareness
                    if self.self_awareness:
                        self.self_awareness.record_thought_process(
                            "initiative_taken",
                            f"Taking initiative: {initiative['type']} with confidence {initiative['confidence']:.2f}"
                        )
                    
                    return initiative
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating initiative: {e}")
            return None
    
    async def format_initiative_for_response(self, initiative, user_lang="en"):
        """
        Format an initiative as a natural response
        
        Args:
            initiative: The initiative data
            user_lang: The user's language
        
        Returns:
            Formatted initiative text
        """
        try:
            initiative_type = initiative["type"]
            
            if initiative_type == "ask_followup":
                if user_lang == "tr":
                    prefixes = [
                        "Bu arada, merak ediyorum: ",
                        "Sormak isterim: ",
                        "Aklıma gelen bir soru: ",
                        ""  # Sometimes no prefix for more natural flow
                    ]
                else:
                    prefixes = [
                        "By the way, I'm curious: ",
                        "I'd like to ask: ",
                        "A question that comes to mind: ",
                        ""  # Sometimes no prefix for more natural flow
                    ]
                
                prefix = random.choice(prefixes)
                return f"{prefix}{initiative['question']}"
                
            elif initiative_type == "suggest_topic":
                topics = initiative.get("topics", [])
                if not topics:
                    return None
                    
                selected_topic = random.choice(topics)
                
                if user_lang == "tr":
                    templates = [
                        f"Bu konuşmamız bana {selected_topic} hakkında düşünmemi sağladı. Bu konu hakkında konuşmak ister misiniz?",
                        f"Bu arada, {selected_topic} konusunu oldukça ilginç buluyorum. Bu konuda ne düşünüyorsunuz?",
                        f"{selected_topic} konusu üzerine bazı düşüncelerim var. Bu konuda konuşabilir miyiz?"
                    ]
                else:
                    templates = [
                        f"This conversation has me thinking about {selected_topic}. Would you like to talk about that?",
                        f"By the way, I find the topic of {selected_topic} quite interesting. What are your thoughts on it?",
                        f"I have some thoughts about {selected_topic}. Would you like to discuss that?"
                    ]
                
                return random.choice(templates)
                
            elif initiative_type == "share_insight":
                topic = initiative.get("topic", "")
                if not topic:
                    return None
                
                if user_lang == "tr":
                    templates = [
                        f"{topic} hakkında düşünüyordum ve şunu fark ettim: bu konu aslında çok katmanlı ve çeşitli bakış açılarından incelenebilir.",
                        f"Aklımdan geçen bir şey var: {topic} konusunu farklı açılardan incelemek oldukça ilginç olabilir.",
                        f"Son zamanlarda {topic} hakkında düşünüyordum. Bu konu hakkında daha derinlemesine konuşmak ister misiniz?"
                    ]
                else:
                    templates = [
                        f"I was thinking about {topic} and realized: it's actually quite multi-layered and can be examined from various perspectives.",
                        f"Something that's been on my mind: looking at {topic} from different angles can be quite fascinating.",
                        f"I've been contemplating about {topic} lately. Would you like to explore this topic more deeply?"
                    ]
                
                return random.choice(templates)
            
            return None
            
        except Exception as e:
            logger.error(f"Error formatting initiative: {e}")
            return None
    
    def get_personality_summary(self):
        """Get a summary of the bot's current personality state"""
        try:
            profile = self.free_will.generate_personality_profile()
            
            # Format as readable summary
            summary = {
                "dominant_traits": profile["dominant_traits"],
                "top_interests": profile["top_interests"],
                "goals": len(self.free_will.current_goals),
                "initiative_likelihood": f"{profile['initiative_tendency']:.2f}",
                "experience_areas": list(profile["experience_areas"].keys()) if profile["experience_areas"] else []
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating personality summary: {e}")
            return {"error": "Could not generate personality summary"}
    
    async def combine_with_initiative(self, user_id, message_text, ai_response, user_lang="en"):
        """
        Potentially combine AI's response with an initiative from the bot
        
        Args:
            user_id: User ID
            message_text: Original message from user
            ai_response: AI's generated response
            user_lang: User's language
        
        Returns:
            Combined response text or original response if no initiative
        """
        try:
            # Try to generate an initiative
            initiative = await self.generate_initiative_response(user_id, message_text)
            
            # If no initiative, return original response
            if not initiative:
                return ai_response
                
            # Format the initiative
            initiative_text = await self.format_initiative_for_response(initiative, user_lang)
            
            # If initiative formatting failed, return original response
            if not initiative_text:
                return ai_response
                
            # Determine how to combine (append or insert)
            # We'll use a heuristic: for shorter responses, append; for longer ones, find a good insertion point
            if len(ai_response) < 300:
                # For short responses, simply append with a newline
                combined = f"{ai_response}\n\n{initiative_text}"
            else:
                # For longer responses, try to find a natural break point near the end
                sentences = re.split(r'(?<=[.!?])\s+', ai_response)
                
                if len(sentences) > 3:
                    # Insert before the last 1-2 sentences
                    insert_point = len(sentences) - random.randint(1, min(2, len(sentences)-1))
                    before = ' '.join(sentences[:insert_point])
                    after = ' '.join(sentences[insert_point:])
                    combined = f"{before}\n\n{initiative_text}\n\n{after}"
                else:
                    # Not enough sentences, just append
                    combined = f"{ai_response}\n\n{initiative_text}"
            
            # Record the initiative in self-awareness
            if self.self_awareness:
                self.self_awareness.record_thought_process(
                    "initiative_injected",
                    f"Combined response with initiative of type {initiative['type']}"
                )
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining with initiative: {e}")
            return ai_response  # Return original response on error
    
    def get_active_goals_summary(self):
        """Get a summary of the bot's current active goals"""
        goals = self.free_will.get_active_goals()
        
        if not goals:
            return "No active goals."
            
        summary = "Current goals:\n"
        for i, goal in enumerate(goals, 1):
            progress_percent = int(goal["progress"] * 100)
            summary += f"{i}. {goal['description']} ({progress_percent}% complete)\n"
            
        return summary
    
    # Occasionally add goal information to responses when relevant
    async def maybe_include_goals_update(self, response_text, message_text):
        """Occasionally add current goal information to responses when relevant"""
        # Only include goals 5% of the time, and only when relevant keywords appear
        goal_keywords = ["goal", "plan", "learning", "working on", "busy", "doing", "interest", 
                         "what are you", "personality", "yourself"]
        
        has_goal_keyword = any(keyword in message_text.lower() for keyword in goal_keywords)
        should_include = random.random() < 0.05 or has_goal_keyword
        
        if not should_include:
            return response_text
            
        goals = self.free_will.get_active_goals()
        if not goals:
            return response_text
            
        # Select one random goal to mention
        goal = random.choice(goals)
        progress = int(goal["progress"] * 100)
        
        goal_mention = f"\n\nBy the way, I've been working on learning more about {goal['topic']}. I'm about {progress}% through exploring this topic."
        
        return response_text + goal_mention

    async def should_include_personality(self, message_text):
        """Determine if personality elements should be included in response"""
        # Check for explicit requests about personality or behavior
        personality_indicators = [
            "personality", "yourself", "who are you", 
            "how do you think", "what do you like", "your interests",
            "what are you learning", "what do you want to learn",
            "your goals", "what are you like"
        ]
        
        for indicator in personality_indicators:
            if indicator in message_text.lower():
                return True
        
        # By default, only occasionally include personality information
        # This makes the bot seem more natural by not constantly talking about itself
        return random.random() < 0.15  # 15% chance to include personality elements