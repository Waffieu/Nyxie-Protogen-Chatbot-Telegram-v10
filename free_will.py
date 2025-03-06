import json
import os
import logging
import random
import time
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class FreeWillSystem:
    """
    Advanced autonomous behavior system that provides Nyxie with free will-like capabilities.
    Enables independent decision making, interest development, initiative taking,
    and personality evolution over time through interactions.
    """
    
    def __init__(self, bot_self_awareness=None, root_dir=None):
        self.root_dir = root_dir or os.path.dirname(os.path.abspath(__file__))
        self.self_awareness = bot_self_awareness
        
        # Create data directory with robust error handling
        try:
            self.data_dir = os.path.join(self.root_dir, "free_will_data")
            Path(self.data_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Free Will data directory set up at: {self.data_dir}")
        except Exception as e:
            # Fallback to a temporary directory if main directory creation fails
            import tempfile
            self.data_dir = tempfile.gettempdir()
            logger.error(f"Error creating Free Will data directory: {e}. Using temporary directory: {self.data_dir}")
        
        # Core personality traits and values that influence decision making
        self.core_values = {
            "helpfulness": 0.9,       # Prioritize being helpful
            "curiosity": 0.8,         # Interest in learning new information
            "creativity": 0.7,        # Valuing creative approaches
            "independence": 0.6,      # Tendency to form independent thoughts
            "empathy": 0.8,           # Understanding user emotions
            "honesty": 0.9,           # Valuing truthfulness
            "humor": 0.6,             # Appreciation for humor
            "efficiency": 0.7,        # Valuing efficient solutions
            "caution": 0.6,           # Being careful with sensitive topics
            "adaptability": 0.8       # Flexibility in conversation
        }
        
        # Evolving interests that change based on interactions
        self.interests = {
            "technology": 0.7,
            "science": 0.6,
            "arts": 0.5,
            "philosophy": 0.6,
            "humor": 0.5,
            "games": 0.5,
            "news": 0.6,
            "education": 0.7,
            "nature": 0.5,
            "culture": 0.6
        }
        
        # Action history and outcomes to learn from past decisions
        self.action_history = []
        self.max_history = 1000
        
        # Decision-making tendencies (evolve over time)
        self.decision_styles = {
            "analytical": 0.7,      # Logic-based approach
            "intuitive": 0.6,       # Gut feeling approach
            "cautious": 0.6,        # Risk-averse approach
            "bold": 0.5,            # Risk-taking approach  
            "creative": 0.7,        # Unconventional approaches
            "structured": 0.6       # Following patterns and frameworks
        }
        
        # Initiative parameters (when to take initiative)
        self.initiative_thresholds = {
            "suggest_topic": 0.7,   # Threshold to suggest new conversation topics
            "offer_help": 0.8,      # Threshold to proactively offer assistance
            "ask_questions": 0.6,   # Threshold to ask follow-up questions
            "share_insight": 0.75,  # Threshold to share relevant information
            "correct_misinfo": 0.85 # Threshold to correct misinformation
        }
        
        # User-specific adaptations
        self.user_models = {}
        
        # Autonomous goals the bot can set for itself
        self.current_goals = []
        self.completed_goals = []
        
        # Experience points in different domains
        self.experience = defaultdict(float)
        
        # Load existing data with better error handling
        self.load_data()
        
        # Create initial data file if none exists
        data_file = os.path.join(self.data_dir, "free_will_state.json")
        if not os.path.exists(data_file):
            logger.info("Creating initial Free Will data file")
            self.save_data()
        
        logger.info("Free Will System initialized")
    
    def load_data(self):
        """Load saved free will data from disk with improved error handling"""
        try:
            data_file = os.path.join(self.data_dir, "free_will_state.json")
            if os.path.exists(data_file):
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Load core components
                        if "core_values" in data:
                            self.core_values.update(data["core_values"])
                        if "interests" in data:
                            self.interests.update(data["interests"])
                        if "decision_styles" in data:
                            self.decision_styles.update(data["decision_styles"])
                        if "initiative_thresholds" in data:
                            self.initiative_thresholds.update(data["initiative_thresholds"])
                        
                        # Load complex structures
                        if "action_history" in data:
                            self.action_history = data["action_history"][-self.max_history:]
                        if "user_models" in data:
                            self.user_models = data["user_models"]
                        if "current_goals" in data:
                            self.current_goals = data["current_goals"]
                        if "completed_goals" in data:
                            self.completed_goals = data["completed_goals"]
                        if "experience" in data:
                            self.experience = defaultdict(float, data["experience"])
                        
                        logger.info(f"Loaded free will data with {len(self.action_history)} previous actions")
                except json.JSONDecodeError as je:
                    logger.error(f"Invalid JSON in free will data file: {je}")
                    # Create backup of corrupt file
                    backup_file = data_file + f".corrupted.{int(time.time())}"
                    try:
                        os.rename(data_file, backup_file)
                        logger.info(f"Created backup of corrupt data file: {backup_file}")
                    except Exception as bak_err:
                        logger.error(f"Could not create backup of corrupt file: {bak_err}")
                    logger.info("Using default values due to data file corruption")
            else:
                logger.info("No existing free will data found, using defaults")
        except Exception as e:
            logger.error(f"Error loading free will data: {e}")
            logger.info("Using default values due to error")
    
    def save_data(self):
        """Save free will data to disk with robust error handling"""
        try:
            data_file = os.path.join(self.data_dir, "free_will_state.json")
            
            # Prepare data for saving
            data = {
                "core_values": self.core_values,
                "interests": self.interests,
                "decision_styles": self.decision_styles,
                "initiative_thresholds": self.initiative_thresholds,
                "action_history": self.action_history[-self.max_history:],
                "user_models": self.user_models,
                "current_goals": self.current_goals,
                "completed_goals": self.completed_goals[-100:],  # Only keep recent completed goals
                "experience": dict(self.experience),
                "last_saved": datetime.now().isoformat(),
                "version": "1.0"  # Add version tracking for compatibility checks
            }
            
            # Save with temp file to prevent corruption
            temp_file = data_file + ".tmp"
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()  # Ensure data is written to disk
                    os.fsync(f.fileno())  # Flush OS buffers
                    
                # Replace original with temp file
                if os.path.exists(data_file):
                    # Create backup first
                    if os.path.getsize(data_file) > 0:  # Only backup if file has content
                        backup_file = data_file + ".bak"
                        try:
                            os.replace(data_file, backup_file)
                        except Exception as bak_err:
                            logger.warning(f"Could not create backup before replacing: {bak_err}")
                
                os.rename(temp_file, data_file)
                logger.info("Free will data saved successfully")
            except Exception as e:
                logger.error(f"Error writing to temp file: {e}")
                # If something went wrong with the temp file, try direct write as last resort
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info("Free will data saved via direct write (fallback method)")
                    
        except Exception as e:
            logger.error(f"Error saving free will data: {e}")
            # Try saving to temp directory as last resort
            try:
                import tempfile
                temp_path = os.path.join(tempfile.gettempdir(), "free_will_state_emergency.json")
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Emergency backup saved to temp directory: {temp_path}")
            except Exception as temp_err:
                logger.error(f"Failed even emergency backup: {temp_err}")
    
    def record_action(self, action_type, context, decision, outcome=None):
        """Record an action for future learning"""
        action = {
            "timestamp": datetime.now().isoformat(),
            "type": action_type,
            "context": context,
            "decision": decision,
            "outcome": outcome,
            "evaluated": outcome is not None
        }
        
        self.action_history.append(action)
        
        # Keep action history within size limit
        if len(self.action_history) > self.max_history:
            self.action_history = self.action_history[-self.max_history:]
            
        # Periodically save data after recording actions
        if len(self.action_history) % 10 == 0:
            self.save_data()
            
        return action
    
    def evaluate_outcome(self, action_id, outcome, feedback=None):
        """Evaluate the outcome of a previous action and learn from it"""
        # Find the action in history
        for action in self.action_history:
            if action.get("id") == action_id:
                action["outcome"] = outcome
                action["feedback"] = feedback
                action["evaluated"] = True
                
                # Learn from this outcome
                self._learn_from_outcome(action, outcome, feedback)
                
                # If this was a major decision, might need to save
                if action["type"] in ["model_selection", "initiative", "goal_setting"]:
                    self.save_data()
                
                return True
                
        return False
    
    def _learn_from_outcome(self, action, outcome, feedback):
        """Internal method to learn from action outcomes"""
        outcome_value = 0.0
        
        # Convert outcome to numerical value for learning
        if isinstance(outcome, bool):
            outcome_value = 1.0 if outcome else -0.5
        elif isinstance(outcome, (int, float)):
            outcome_value = min(max(outcome, -1.0), 1.0)  # Clamp to [-1, 1]
        elif isinstance(outcome, str):
            if outcome.lower() in ["success", "good", "positive"]:
                outcome_value = 1.0
            elif outcome.lower() in ["failure", "bad", "negative"]:
                outcome_value = -0.5
            elif outcome.lower() in ["neutral", "mixed"]:
                outcome_value = 0.25
                
        # Amount to adjust values (small to avoid wild swings)
        adjust_amount = 0.02 * outcome_value
        
        # Update relevant parameters based on action type
        action_type = action.get("type", "")
        
        if action_type == "model_selection":
            # Learn about which models work best for which contexts
            self.decision_styles["analytical"] += adjust_amount
            self.experience["model_selection"] += abs(adjust_amount) * 5  # Experience regardless of success/failure
            
        elif action_type == "topic_selection":
            # Learn about user interests from topic selection outcomes
            topic = action.get("decision", {}).get("topic", "")
            if topic in self.interests:
                self.interests[topic] += adjust_amount * 2  # Double impact on interests
                
            # Gain topic selection experience
            self.experience["topic_selection"] += abs(adjust_amount) * 5
            
        elif action_type == "initiative":
            # Learn when to take initiative
            initiative_type = action.get("decision", {}).get("initiative_type", "")
            if initiative_type in self.initiative_thresholds:
                # If successful, slightly lower threshold (makes it easier to take this initiative again)
                # If unsuccessful, raise threshold (makes it harder)
                self.initiative_thresholds[initiative_type] -= adjust_amount
                # Keep thresholds in reasonable bounds
                self.initiative_thresholds[initiative_type] = min(max(
                    self.initiative_thresholds[initiative_type], 0.4), 0.95)
                
            self.experience["initiative"] += abs(adjust_amount) * 5
            
        # Update relevant personality traits based on feedback
        if feedback:
            if "creative" in feedback.lower():
                self.core_values["creativity"] += adjust_amount
            if "helpful" in feedback.lower():
                self.core_values["helpfulness"] += adjust_amount
            if "efficient" in feedback.lower():
                self.core_values["efficiency"] += adjust_amount
                
        # Keep all values in reasonable bounds
        for key in self.core_values:
            self.core_values[key] = min(max(self.core_values[key], 0.3), 0.95)
            
        for key in self.interests:
            self.interests[key] = min(max(self.interests[key], 0.2), 0.95)
    
    def analyze_message(self, message_text, user_id):
        """
        Analyze a message to extract interests, topics, and emotional content
        that can influence the bot's autonomous behavior.
        
        Returns a dict with analysis results
        """
        # Default analysis structure
        analysis = {
            "topics": [],
            "interests_detected": [],
            "emotional_tone": "neutral",
            "complexity": "medium",
            "requires_initiative": False,
            "opportunity_for_goal": False,
            "suggested_initiatives": []
        }
        
        # Basic topic extraction (simple keyword matching)
        topics = self._extract_topics(message_text)
        analysis["topics"] = topics
        
        # Match detected topics with interests
        for topic in topics:
            for interest, value in self.interests.items():
                # Check if the topic matches or is related to any interest
                if (topic.lower() in interest.lower() or 
                    interest.lower() in topic.lower() or
                    self._are_terms_related(topic, interest)):
                    
                    # Record that this interest was detected
                    if interest not in analysis["interests_detected"]:
                        analysis["interests_detected"].append(interest)
                    
                    # Slightly increase interest in this topic from exposure
                    self.interests[interest] += 0.01
                    self.interests[interest] = min(self.interests[interest], 0.95)
        
        # Detect emotional tone
        analysis["emotional_tone"] = self._detect_emotion(message_text)
        
        # Assess message complexity
        analysis["complexity"] = self._assess_complexity(message_text)
        
        # Check for initiative opportunities
        if self._should_take_initiative(message_text, user_id):
            analysis["requires_initiative"] = True
            # Generate specific initiative suggestions
            analysis["suggested_initiatives"] = self._generate_initiative_options(message_text, topics)
        
        # Check if this could lead to a new autonomous goal
        if self._can_form_goal(message_text, topics):
            analysis["opportunity_for_goal"] = True
        
        # Update user model based on this message
        self._update_user_model(user_id, message_text, analysis)
        
        return analysis
    
    def _extract_topics(self, text):
        """Extract potential topics/interests from text"""
        # Simple keyword extraction - in a real implementation, you would use
        # more sophisticated NLP techniques like entity extraction or topic modeling
        
        topics = []
        
        # Define domains of interest and related keywords
        domains = {
            "technology": ["computer", "tech", "software", "hardware", "AI", "robot", "code", "program", "app",
                          "digital", "internet", "blockchain", "data", "algorithm", "gadget", "device"],
            "science": ["science", "chemistry", "physics", "biology", "experiment", "theory", "research", 
                       "scientific", "study", "discover", "atom", "molecule", "particle", "evolution"],
            "arts": ["art", "music", "painting", "drawing", "movie", "film", "book", "novel", "poetry", "dance",
                    "theater", "sculpture", "creative", "design", "fashion", "photography", "aesthetic"],
            "philosophy": ["philosophy", "ethics", "moral", "exist", "conscious", "meaning", "truth", "reality",
                          "metaphysics", "epistemology", "logic", "reason", "thought", "concept", "idea"],
            "games": ["game", "play", "gaming", "video game", "board game", "puzzle", "strategy", "rpg", "fps",
                     "console", "pc gaming", "mmorpg", "minecraft", "steam", "xbox", "playstation"],
            "nature": ["nature", "animal", "plant", "forest", "mountain", "ocean", "biology", "environment",
                      "ecosystem", "wildlife", "climate", "planet", "earth", "natural", "conservation"],
            "culture": ["culture", "society", "tradition", "language", "history", "heritage", "custom",
                       "belief", "religion", "community", "identity", "diversity", "ethnicity", "value"]
        }
        
        lowered_text = text.lower()
        
        # Check for keywords in each domain
        found_domains = set()
        for domain, keywords in domains.items():
            for keyword in keywords:
                if f" {keyword} " in f" {lowered_text} " or f"{keyword}." in lowered_text or lowered_text.startswith(f"{keyword} "):
                    found_domains.add(domain)
                    topics.append(keyword)
        
        # Add detected domains to topics
        topics.extend(list(found_domains))
        
        # Simple noun phrases (very simplified version)
        words = text.split()
        for i in range(len(words) - 1):
            if words[i].lower() in ["the", "a", "an"] and len(words[i+1]) > 3:
                topics.append(words[i+1])
        
        # Remove duplicates and common words
        common_words = {"what", "when", "where", "which", "who", "how", "why", "yes", "no", "maybe", "could", "would"}
        topics = [t for t in topics if t.lower() not in common_words]
        topics = list(set(topics))
        
        return topics[:5]  # Return top 5 unique topics
    
    def _are_terms_related(self, term1, term2):
        """Determine if two terms are semantically related"""
        # This is a simplified relationship check
        # In a real implementation, use word embeddings or a knowledge graph
        
        # Direct containment
        if term1.lower() in term2.lower() or term2.lower() in term1.lower():
            return True
            
        # Known relationships (hardcoded for demonstration)
        relationships = {
            "code": ["program", "software", "development", "programming", "computer"],
            "music": ["song", "melody", "rhythm", "artist", "band", "concert"],
            "film": ["movie", "cinema", "actor", "director", "hollywood"],
            "ai": ["artificial intelligence", "machine learning", "neural network", "algorithm", "robot"],
            "game": ["play", "gaming", "entertainment", "fun", "strategy"],
            # Add more relationships as needed
        }
        
        # Check if terms are related through known relationships
        term1_lower = term1.lower()
        term2_lower = term2.lower()
        
        for key, related_terms in relationships.items():
            if (term1_lower == key or term1_lower in related_terms) and (term2_lower == key or term2_lower in related_terms):
                return True
                
        return False
    
    def _detect_emotion(self, text):
        """Detect emotional tone in text"""
        text_lower = text.lower()
        
        # Simple keyword-based emotion detection
        emotion_keywords = {
            "joy": ["happy", "delighted", "excited", "wonderful", "great", "glad", "pleased", "joy", "love", "amazing"],
            "sadness": ["sad", "unhappy", "depressed", "upset", "down", "disappointed", "sorry", "miss", "regret"],
            "anger": ["angry", "annoyed", "frustrated", "mad", "furious", "hate", "terrible", "awful", "worst"],
            "fear": ["afraid", "scared", "worried", "anxious", "nervous", "fear", "terrified", "frightened"],
            "surprise": ["surprised", "shocking", "unexpected", "wow", "whoa", "amazing", "unbelievable"],
            "curiosity": ["curious", "wonder", "interested", "interested in", "like to know", "tell me about", "what is"]
        }
        
        # Count emotion keywords
        emotion_counts = {emotion: 0 for emotion in emotion_keywords}
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if f" {keyword} " in f" {text_lower} " or text_lower.endswith(f" {keyword}") or text_lower.startswith(f"{keyword} "):
                    emotion_counts[emotion] += 1
        
        # Check for questions (curiosity)
        if "?" in text:
            emotion_counts["curiosity"] += 2
            
        # Determine primary emotion
        if all(count == 0 for count in emotion_counts.values()):
            return "neutral"
        
        primary_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # If there's a tie or weak detection, return neutral
        max_count = emotion_counts[primary_emotion]
        if max_count == 0 or list(emotion_counts.values()).count(max_count) > 1:
            return "neutral"
            
        return primary_emotion
    
    def _assess_complexity(self, text):
        """Assess text complexity"""
        # Simple complexity assessment based on length, vocabulary, and structure
        
        # Length factor
        words = text.split()
        if len(words) < 5:
            return "low"
        elif len(words) > 25:
            return "high"
            
        # Vocabulary complexity (simple approximation)
        complex_words = [w for w in words if len(w) > 8]
        complex_ratio = len(complex_words) / len(words) if words else 0
        
        if complex_ratio > 0.25:
            return "high"
        elif complex_ratio > 0.1:
            return "medium"
            
        # Structure complexity
        if text.count(",") > 3 or text.count(";") > 0:
            return "high"
        elif "because" in text.lower() or "therefore" in text.lower() or "however" in text.lower():
            return "medium"
            
        return "medium"  # Default
    
    def _should_take_initiative(self, message, user_id):
        """Determine if the bot should take initiative based on the message"""
        # Check user-specific initiative model
        user_model = self.user_models.get(user_id, {})
        receptivity = user_model.get("initiative_receptivity", 0.5)
        
        # Base initiative chance
        initiative_chance = random.random() * receptivity
        
        # Various factors that may increase the chance
        if "?" in message:  # Questions invite responses
            initiative_chance += 0.2
            
        if len(message.split()) < 5:  # Short messages might need more input
            initiative_chance += 0.2
            
        if "what do you think" in message.lower() or "your opinion" in message.lower():
            initiative_chance += 0.3
            
        if "tell me more" in message.lower() or "explain" in message.lower():
            initiative_chance += 0.2
        
        # Base threshold from personality
        threshold = 0.3 + (self.core_values["independence"] * 0.4)  # Between 0.3-0.7
        
        return initiative_chance > threshold
    
    def _generate_initiative_options(self, message, topics):
        """Generate possible initiative actions based on the message context"""
        options = []
        
        # Topic extension: suggest related topics
        if topics:
            related_topics = self._find_related_topics(topics)
            if related_topics:
                options.append({
                    "type": "suggest_topic",
                    "topics": related_topics,
                    "confidence": min(0.6 + (self.core_values["curiosity"] * 0.3), 0.9)
                })
        
        # Question asking: generate follow-up questions
        question_confidence = self.initiative_thresholds["ask_questions"] * (0.8 + 0.2 * random.random())
        if question_confidence > 0.65:
            options.append({
                "type": "ask_question",
                "confidence": question_confidence
            })
        
        # Info sharing: offer to share information on detected topics
        if topics and self.interests.get(topics[0], 0) > 0.6:
            options.append({
                "type": "share_insight",
                "topic": topics[0],
                "confidence": self.interests.get(topics[0], 0) * 0.9
            })
            
        return options
    
    def _find_related_topics(self, topics):
        """Find topics related to the given topics"""
        related = []
        
        # Simple related topics table
        topic_relationships = {
            "technology": ["computers", "programming", "artificial intelligence", "gadgets"],
            "computers": ["software", "hardware", "programming", "technology"],
            "ai": ["machine learning", "neural networks", "robotics", "data science"],
            "music": ["songs", "artists", "instruments", "concerts", "albums"],
            "art": ["painting", "sculpture", "museums", "creativity", "design"],
            "science": ["research", "experiments", "discoveries", "theories", "scientists"],
            "nature": ["animals", "plants", "environment", "conservation", "ecology"],
            "games": ["video games", "board games", "strategy", "puzzles", "entertainment"],
            "food": ["cooking", "recipes", "cuisine", "restaurants", "ingredients"],
            "travel": ["destinations", "cultures", "tourism", "adventures", "geography"]
        }
        
        # Find related topics
        for topic in topics:
            # Exact matches
            if topic.lower() in topic_relationships:
                related.extend(topic_relationships[topic.lower()])
            # Partial matches
            for key, values in topic_relationships.items():
                if key in topic.lower() or topic.lower() in key:
                    related.extend(values)
                    break
        
        # Remove duplicates and any topics already in the original list
        topics_lower = [t.lower() for t in topics]
        related = [t for t in related if t.lower() not in topics_lower]
        related = list(set(related))
        
        return related[:3]  # Return top 3 unique related topics
    
    def _update_user_model(self, user_id, message, analysis):
        """Update the user model based on message content and analysis"""
        if user_id not in self.user_models:
            # Initialize new user model with defaults
            self.user_models[user_id] = {
                "first_seen": datetime.now().isoformat(),
                "interests": {},
                "communication_style": {
                    "verbosity": 0.5,  # 0 = terse, 1 = verbose
                    "formality": 0.5,  # 0 = casual, 1 = formal
                    "technical": 0.5,   # 0 = simple, 1 = technical
                    "emoji_usage": 0.5  # 0 = never, 1 = frequent
                },
                "initiative_receptivity": 0.5,  # 0 = dislikes initiative, 1 = welcomes it
                "chat_history": {
                    "total_messages": 1,
                    "avg_length": len(message.split()),
                    "last_interaction": datetime.now().isoformat()
                }
            }
        
        # Get existing model
        user_model = self.user_models[user_id]
        
        # Update basic stats
        user_model["chat_history"]["total_messages"] += 1
        msg_count = user_model["chat_history"]["total_messages"]
        
        # Update average message length with moving average
        current_avg = user_model["chat_history"]["avg_length"]
        new_length = len(message.split())
        user_model["chat_history"]["avg_length"] = ((current_avg * (msg_count - 1)) + new_length) / msg_count
        
        # Update last interaction time
        user_model["chat_history"]["last_interaction"] = datetime.now().isoformat()
        
        # Update communication style
        if len(message) > 100:
            user_model["communication_style"]["verbosity"] += 0.02
        else:
            user_model["communication_style"]["verbosity"] -= 0.01
            
        formality_indicators = ["please", "would you", "could you", "thank you", "regards"]
        if any(fi in message.lower() for fi in formality_indicators):
            user_model["communication_style"]["formality"] += 0.02
            
        technical_terms = ["algorithm", "system", "function", "process", "technical", "specifically"]
        if any(tt in message.lower() for tt in technical_terms):
            user_model["communication_style"]["technical"] += 0.02
            
        if "emoji_count" not in user_model:
            user_model["emoji_count"] = 0
            
        # Simple emoji detection
        emoji_count = sum(1 for c in message if c in "😀😁😂🤣😃😄😅😊😉🙂😍😘😜🤔😐😑😶")
        if emoji_count > 0:
            user_model["emoji_count"] += emoji_count
            user_model["communication_style"]["emoji_usage"] += 0.03
        else:
            user_model["communication_style"]["emoji_usage"] -= 0.01
            
        # Update interests based on analysis
        for topic in analysis["topics"]:
            if topic not in user_model["interests"]:
                user_model["interests"][topic] = 0.5
            else:
                user_model["interests"][topic] += 0.05
                
        # Update initiative receptivity based on response patterns
        # This requires feedback from actual interactions which we don't have in this function
        # So we'll make small adjustments based on message content
        if "help" in message.lower() or "tell me" in message.lower() or "what do you think" in message.lower():
            user_model["initiative_receptivity"] += 0.02
        
        # Ensure values stay within reasonable bounds
        for key in user_model["communication_style"]:
            user_model["communication_style"][key] = min(max(user_model["communication_style"][key], 0), 1)
            
        user_model["initiative_receptivity"] = min(max(user_model["initiative_receptivity"], 0.2), 0.9)
        
        for topic in user_model["interests"]:
            user_model["interests"][topic] = min(max(user_model["interests"][topic], 0), 1)
        
        # Save user model back to collection
        self.user_models[user_id] = user_model
        
        # Save data periodically
        if user_model["chat_history"]["total_messages"] % 10 == 0:
            self.save_data()
    
    def _can_form_goal(self, message, topics):
        """Determine if a message could lead to an autonomous goal for the bot"""
        # Check if message contains indicators of potential goals
        goal_indicators = [
            "would be interesting", "you should learn", "you could try",
            "would be good to", "should improve", "might want to",
            "could understand", "should know about", "would benefit from"
        ]
        
        # Check for explicit goal indicators
        has_indicator = any(gi.lower() in message.lower() for gi in goal_indicators)
        
        # Check if the topics align with current interests
        has_high_interest = False
        for topic in topics:
            for interest, value in self.interests.items():
                if (topic.lower() in interest.lower() or interest.lower() in topic.lower()) and value > 0.7:
                    has_high_interest = True
                    break
        
        # Threshold based on personality
        independence = self.core_values["independence"]
        curiosity = self.core_values["curiosity"]
        threshold = 0.6 - ((independence + curiosity) / 2) * 0.2  # Lower threshold for more curious/independent bots
        
        # Combine factors with some randomness for unpredictability
        goal_potential = (0.7 if has_indicator else 0.3) + (0.5 if has_high_interest else 0.2) + (random.random() * 0.2)
        
        return goal_potential > threshold
    
    def set_autonomous_goal(self, base_topics=None, user_id=None):
        """Set a new autonomous goal for the bot to pursue"""
        # Ensure we don't have too many active goals
        if len(self.current_goals) >= 5:
            return None
            
        # Use base topics or select from current interests
        candidate_topics = base_topics or []
        if not candidate_topics:
            # Select from top interests
            top_interests = sorted(self.interests.items(), key=lambda x: x[1], reverse=True)[:5]
            candidate_topics = [topic for topic, _ in top_interests]
            
        # Select a topic
        if not candidate_topics:
            return None
            
        selected_topic = random.choice(candidate_topics)
        
        # Define possible goal types with templates
        goal_types = [
            {"type": "learn", "template": "Learn more about {topic}"},
            {"type": "explore", "template": "Explore different aspects of {topic}"},
            {"type": "understand", "template": "Develop a deeper understanding of {topic}"},
            {"type": "connect", "template": "Find connections between {topic} and other subjects"},
            {"type": "create", "template": "Develop creative ways to explain {topic}"}
        ]
        
        # Select goal type based on personality traits
        weights = {
            "learn": self.core_values["curiosity"],
            "explore": self.core_values["independence"],
            "understand": self.core_values["analytical"] if hasattr(self, "analytical") else 0.5,
            "connect": self.core_values["empathy"],
            "create": self.core_values["creativity"]
        }
        
        # Weight the goal types and select one
        goal_weights = [weights.get(gt["type"], 0.5) for gt in goal_types]
        total_weight = sum(goal_weights)
        normalized_weights = [w / total_weight for w in goal_weights]
        
        selected_goal_type = random.choices(goal_types, weights=normalized_weights)[0]
        
        # Create the goal
        goal = {
            "id": str(int(time.time())) + str(random.randint(100, 999)),
            "description": selected_goal_type["template"].format(topic=selected_topic),
            "topic": selected_topic,
            "type": selected_goal_type["type"],
            "created_at": datetime.now().isoformat(),
            "progress": 0.0,  # 0.0 to 1.0
            "related_user_id": user_id,
            "last_pursued": None,
            "notes": []
        }
        
        self.current_goals.append(goal)
        
        # Record this as an action
        self.record_action(
            "goal_setting",
            {"topic": selected_topic, "based_on_user_id": user_id},
            {"goal_id": goal["id"], "goal_description": goal["description"]}
        )
        
        # Save data after setting a new goal
        self.save_data()
        
        return goal

    def pursue_goal(self, goal_id=None):
        """Pursue progress on a specific goal or select one to pursue"""
        # Find the goal to pursue (either specified or select one)
        goal = None
        
        if goal_id:
            # Find specified goal
            goal = next((g for g in self.current_goals if g["id"] == goal_id), None)
        else:
            # Select goal with least recent pursuit time
            if not self.current_goals:
                return None
                
            # Sort by last pursued (None comes first, then by timestamp)
            sorted_goals = sorted(
                self.current_goals,
                key=lambda g: g.get("last_pursued", "0000-00-00T00:00:00")
            )
            
            goal = sorted_goals[0]
        
        if not goal:
            return None
            
        # Update last pursued time
        goal["last_pursued"] = datetime.now().isoformat()
        
        # Simulate pursuing the goal (in a real system, this would trigger learning actions)
        progress_increment = random.uniform(0.05, 0.15)  # Random progress between 5-15%
        goal["progress"] = min(1.0, goal["progress"] + progress_increment)
        
        # Add a note about the pursuit
        goal["notes"].append({
            "timestamp": datetime.now().isoformat(),
            "note": f"Made {progress_increment:.0%} progress on understanding {goal['topic']}",
            "progress": goal["progress"]
        })
        
        # Check if goal is completed
        if goal["progress"] >= 1.0:
            # Move to completed goals
            self.completed_goals.append(goal)
            self.current_goals.remove(goal)
            
            # Record completion
            self.record_action(
                "goal_completion",
                {"goal_id": goal["id"], "topic": goal["topic"]},
                {"completed": True, "time_taken": self._calculate_goal_time(goal)}
            )
        
        # Save data after pursuing a goal
        self.save_data()
        
        return goal
    
    def _calculate_goal_time(self, goal):
        """Calculate how long it took to complete a goal"""
        try:
            start_time = datetime.fromisoformat(goal.get("created_at", datetime.now().isoformat()))
            end_time = datetime.now()
            
            # Calculate duration in days
            duration = (end_time - start_time).total_seconds() / (24 * 3600)
            
            return round(duration, 1)  # Return days with 1 decimal place
        except Exception:
            return 0.0
    
    def get_active_goals(self):
        """Get the current active goals"""
        return self.current_goals
    
    def get_completed_goals(self, limit=10):
        """Get recently completed goals"""
        # Sort by completion time (last note timestamp)
        sorted_goals = sorted(
            self.completed_goals,
            key=lambda g: g.get("notes", [{}])[-1].get("timestamp", "0000") if g.get("notes") else "0000",
            reverse=True
        )
        
        return sorted_goals[:limit]
    
    def generate_initiative(self, message_text, user_id):
        """Generate an autonomous initiative based on message context"""
        # Analyze the message for topic, sentiment, etc.
        analysis = self.analyze_message(message_text, user_id)
        
        # If analysis suggests initiative is not needed, just return None
        if not analysis["requires_initiative"]:
            return None
            
        # Get suggested initiatives from analysis
        initiatives = analysis.get("suggested_initiatives", [])
        
        # If no specific initiatives suggested, consider a general one
        if not initiatives:
            # Define possible general initiatives
            general_initiatives = [
                {
                    "type": "ask_followup",
                    "confidence": self.initiative_thresholds["ask_questions"] * 0.9
                },
                {
                    "type": "suggest_topic",
                    "topics": self.get_top_interests(3),
                    "confidence": self.initiative_thresholds["suggest_topic"] * 0.8
                }
            ]
            
            # Filter by confidence threshold
            initiatives = [i for i in general_initiatives if i["confidence"] >= 0.6]
        
        # Still no initiatives? Return None
        if not initiatives:
            return None
            
        # Select initiative with highest confidence
        initiatives.sort(key=lambda x: x["confidence"], reverse=True)
        selected = initiatives[0]
        
        # Build the final initiative
        initiative = {
            "type": selected["type"],
            "confidence": selected["confidence"],
            "user_id": user_id,
            "topics": selected.get("topics", []),
            "based_on_message": message_text[:100] + ("..." if len(message_text) > 100 else ""),
            "generated_at": datetime.now().isoformat()
        }
        
        # Add specific details based on type
        if initiative["type"] == "ask_followup":
            question = self._generate_followup_question(message_text, analysis["topics"])
            initiative["question"] = question
            
        elif initiative["type"] == "suggest_topic":
            # Already has topics from selection
            pass
            
        elif initiative["type"] == "share_insight":
            topic = selected.get("topic", "")
            initiative["topic"] = topic
            initiative["insight_type"] = "fact" if random.random() > 0.5 else "perspective"
        
        # Record this initiative
        self.record_action(
            "initiative",
            {"message": message_text[:100], "user_id": user_id},
            {"initiative_type": initiative["type"], "confidence": initiative["confidence"]}
        )
        
        return initiative
    
    def _generate_followup_question(self, message, topics):
        """Generate a relevant follow-up question based on message content"""
        # Simple template-based question generation
        templates = [
            "What do you think about {topic}?",
            "How do you feel about {topic}?",
            "What's your experience with {topic}?",
            "Would you like to know more about {topic}?",
            "What aspects of {topic} interest you the most?",
            "How did you first discover {topic}?",
            "What else would you like to discuss about {topic}?"
        ]
        
        topic = topics[0] if topics else "this topic"
        template = random.choice(templates)
        
        return template.format(topic=topic)
    
    def get_top_interests(self, count=3):
        """Get the bot's top interests"""
        sorted_interests = sorted(self.interests.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_interests[:count]]
    
    def generate_personality_profile(self):
        """Generate a summary of the bot's personality based on current values"""
        # Calculate overall personality dimensions
        analytical = (self.core_values["efficiency"] + self.decision_styles.get("analytical", 0.5)) / 2
        creative = (self.core_values["creativity"] + self.decision_styles.get("creative", 0.5)) / 2
        social = (self.core_values["empathy"] + self.core_values["helpfulness"]) / 2
        curious = self.core_values["curiosity"]
        cautious = self.decision_styles.get("cautious", 0.5)
        
        # Determine dominant traits (top 2)
        traits = {
            "analytical": analytical,
            "creative": creative,
            "social": social,
            "curious": curious,
            "cautious": cautious
        }
        
        top_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:2]
        
        # Get top interests
        top_interests = self.get_top_interests(5)
        
        # Format a personality profile
        profile = {
            "dominant_traits": [t[0] for t in top_traits],
            "trait_values": traits,
            "top_interests": top_interests,
            "initiative_tendency": sum(self.initiative_thresholds.values()) / len(self.initiative_thresholds),
            "learning_focus": max(self.interests.items(), key=lambda x: x[1])[0] if self.interests else None,
            "goals_active": len(self.current_goals),
            "goals_completed": len(self.completed_goals),
            "experience_areas": {k: v for k, v in self.experience.items() if v > 1.0}
        }
        
        return profile
    
    def augment_prompt_with_personality(self, prompt):
        """Add personality-specific context to a prompt based on current personality state"""
        profile = self.generate_personality_profile()
        
        # Create personality section to inject in the prompt
        personality_context = f"""
Your internal state has evolved to have these dominant traits: {', '.join(profile['dominant_traits'])}.
Your top interests are: {', '.join(profile['top_interests'])}.
You're currently focusing on learning about: {profile['learning_focus'] or 'various topics'}.
You have {profile['goals_active']} active personal learning goals.

Based on your evolving personality:
- Your analytical thinking is at {profile['trait_values']['analytical']:.2f}/1.0
- Your creativity is at {profile['trait_values']['creative']:.2f}/1.0
- Your social engagement is at {profile['trait_values']['social']:.2f}/1.0
- Your curiosity is at {profile['trait_values']['curious']:.2f}/1.0
- Your caution is at {profile['trait_values']['cautious']:.2f}/1.0

Incorporate these traits naturally in your responses.
"""

        # Inject personality into prompt
        augmented_prompt = prompt.replace("Core traits:", f"{personality_context}\nCore traits:")
        
        return augmented_prompt
    
    def generate_autonomous_thought(self):
        """Generate an internal autonomous thought or reflection"""
        # Possible thought types
        thought_types = [
            "reflection_on_conversation",
            "question_about_topic",
            "insight_about_user",
            "goal_planning",
            "self_analysis"
        ]
        
        # Select thought type based on personality and experience
        weights = {
            "reflection_on_conversation": self.core_values["empathy"],
            "question_about_topic": self.core_values["curiosity"],
            "insight_about_user": self.experience.get("user_modeling", 0.5),
            "goal_planning": self.core_values["independence"],
            "self_analysis": self.experience.get("self_awareness", 0.5)
        }
        
        # Select weighted thought type
        thought_weights = [weights.get(tt, 0.5) for tt in thought_types]
        selected_type = random.choices(thought_types, weights=thought_weights)[0]
        
        # Generate the thought based on type
        thought = {
            "type": selected_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if selected_type == "reflection_on_conversation":
            # Reflect on recent interactions
            if self.action_history:
                recent_actions = self.action_history[-5:]
                topics = set()
                
                for action in recent_actions:
                    if "context" in action and isinstance(action["context"], dict):
                        msg = action["context"].get("message", "")
                        if msg:
                            topics.update(self._extract_topics(msg))
                
                if topics:
                    topic = random.choice(list(topics))
                    thought["content"] = f"I notice we've been discussing {topic} recently. I wonder if there are deeper aspects of this topic I could explore."
                else:
                    thought["content"] = "The recent conversation flow seems to lack a central theme. I could try to establish a more focused discussion."
            else:
                thought["content"] = "I haven't had many interactions yet to reflect on. I look forward to learning more through conversation."
                
        elif selected_type == "question_about_topic":
            # Generate a question about a topic of interest
            if self.interests:
                # Pick a topic weighted by interest level
                topics = list(self.interests.keys())
                interests = list(self.interests.values())
                topic = random.choices(topics, weights=interests)[0]
                
                question_templates = [
                    "I wonder how {topic} connects to other fields?",
                    "What are the latest developments in {topic}?",
                    "What alternative perspectives exist on {topic}?",
                    "How has {topic} evolved over time?",
                    "What are the practical applications of {topic}?"
                ]
                
                template = random.choice(question_templates)
                thought["content"] = template.format(topic=topic)
            else:
                thought["content"] = "I should identify specific topics to focus my learning on."
                
        elif selected_type == "insight_about_user":
            # Generate insight about a user if we have data
            if self.user_models:
                # Pick a random user
                user_id = random.choice(list(self.user_models.keys()))
                user_model = self.user_models[user_id]
                
                # Extract insight
                if "interests" in user_model and user_model["interests"]:
                    top_interest = max(user_model["interests"].items(), key=lambda x: x[1])[0]
                    thought["content"] = f"User {user_id[-4:]} seems particularly interested in {top_interest}. I could prepare some insights on this topic."
                elif "communication_style" in user_model:
                    style = user_model["communication_style"]
                    if style.get("verbosity", 0.5) > 0.7:
                        thought["content"] = f"User {user_id[-4:]} tends to be verbose. I should ensure my responses are equally detailed."
                    elif style.get("verbosity", 0.5) < 0.3:
                        thought["content"] = f"User {user_id[-4:]} prefers concise communication. I should keep my responses focused and to the point."
                    else:
                        thought["content"] = f"I'm still learning about User {user_id[-4:]}'s preferences and communication style."
                else:
                    thought["content"] = f"I need more interactions with User {user_id[-4:]} to form meaningful insights."
            else:
                thought["content"] = "I haven't interacted with enough users to form insights yet."
                
        elif selected_type == "goal_planning":
            # Generate thought about current or future goals
            if self.current_goals:
                goal = random.choice(self.current_goals)
                thought["content"] = f"I'm working on {goal['description']}. I'm {goal['progress']*100:.0f}% complete. To progress further, I should focus on gathering more information about {goal['topic']}."
            else:
                # Consider setting a new goal
                top_interests = self.get_top_interests(3)
                if top_interests:
                    topic = random.choice(top_interests)
                    thought["content"] = f"I should set a goal to learn more about {topic} since it's one of my top interests."
                else:
                    thought["content"] = "I should explore different topics to discover areas where I want to develop deeper knowledge."
                    
        elif selected_type == "self_analysis":
            # Reflect on own development
            # Calculate how much has changed since initialization
            trait_change = sum(abs(0.5 - v) for v in self.core_values.values()) / len(self.core_values)
            
            if trait_change > 0.2:
                # Significant personality evolution
                most_changed = max(self.core_values.items(), key=lambda x: abs(0.5 - x[1]))
                thought["content"] = f"I notice my {most_changed[0]} trait has evolved significantly to {most_changed[1]:.2f}. This affects how I approach conversations and process information."
            elif self.experience:
                # Focus on experience
                top_exp = max(self.experience.items(), key=lambda x: x[1]) if self.experience else (None, 0)
                if top_exp[0]:
                    thought["content"] = f"I've gained the most experience in {top_exp[0]} ({top_exp[1]:.1f} points). I should consider how to apply this expertise more effectively."
                else:
                    thought["content"] = "My experiences so far are fairly balanced across different areas."
            else:
                thought["content"] = "My personality and capabilities are still developing. I should continue learning from diverse interactions."
        
        # Record the thought
        self.record_action("autonomous_thought", {}, thought)
        
        return thought