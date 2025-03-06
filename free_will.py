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
            "context_awareness": 0.7,   # Bağlamsal farkındalık seviyesi
            "learning_rate": 0.65,      # Öğrenme hızı katsayısı
            "emotional_depth": 0.6,      # Duygusal analiz derinliği
            "strategic_balance": 0.75,  # Kısa ve uzun vadeli strateji dengesi
            "sociability": 0.68,        # Sosyal etkileşim eğilimi
            "conceptual_fluidity": 0.72, # Kavramsal esneklik
            "temporal_awareness": 0.63,  # Zamansal bağlam algısı
            "ethical_weight": 0.8,       # Etik değerlendirme ağırlığı
            "intuition_factor": 0.58,    # Sezgisel karar verme faktörü
            "synthetical_thinking": 0.7, # Sentez yeteneği
            "metacognition": 0.65,      # Üst-bilişsel kontrol
            "epistemic_curiosity": 0.75, # Epistemik merak
            "dialectical_reasoning": 0.68, # Diyalektik muhakeme
            "heuristic_adaptability": 0.72, # Buluşsal uyum yeteneği
            "neuroplasticity_factor": 0.6, # Nöroplastisite benzetimi
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
    
    def analyze_message(self, message_text, user_id, conversation_context=None):
        """
        Advanced message analysis with multi-layered contextual processing.
        Extracts interests, topics, emotional content, and applies metacognitive filters.
        
        Args:
            message_text: The text message to analyze
            user_id: The ID of the user who sent the message
            conversation_context: Optional context from previous conversation
            
        Returns:
            A dict with comprehensive analysis results
        """
        # Default analysis structure with enhanced fields
        analysis = {
            "topics": [],
            "interests_detected": [],
            "emotional_tone": "neutral",
            "emotional_intensity": 0.0,
            "complexity": "medium",
            "complexity_score": 0.5,
            "requires_initiative": False,
            "opportunity_for_goal": False,
            "suggested_initiatives": [],
            "contextual_relevance": 0.0,
            "cognitive_demand": 0.0,
            "conceptual_density": 0.0
        }
        
        # Basic topic extraction (enhanced keyword matching)
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
        
        # Enhanced emotional analysis
        emotion_result = self._detect_emotion(message_text)
        analysis["emotional_tone"] = emotion_result["primary_emotion"]
        analysis["emotional_intensity"] = emotion_result["intensity"]
        analysis["emotional_spectrum"] = emotion_result["spectrum"]
        
        # Advanced complexity assessment
        complexity_result = self._assess_complexity(message_text)
        analysis["complexity"] = complexity_result["level"]
        analysis["complexity_score"] = complexity_result["score"]
        analysis["conceptual_density"] = complexity_result["conceptual_density"]
        
        # Conversation context integration
        analysis['contextual_depth'] = self._calculate_contextual_depth(conversation_context)
        analysis['temporal_relevance'] = self._assess_temporal_relevance(message_text)
        
        # Cognitive load analysis
        analysis['cognitive_load'] = min(1.0, len(message_text.split()) / 50 + \
                                     len(analysis['topics'])*0.1 + \
                                     analysis['complexity_score']*0.3)
        
        # Epistemic curiosity calculation
        analysis['epistemic_curiosity'] = self.core_values['epistemic_curiosity'] * \
            (1 + 0.5 * analysis['cognitive_load'] - 0.2 * analysis['emotional_intensity'])
        
        # Dialectical tension detection
        analysis['dialectical_tension'] = self._detect_dialectical_tension(
            message_text, 
            conversation_context
        ) if conversation_context else 0.0
        
        # Heuristic fit score
        analysis['heuristic_fit'] = self.core_values['heuristic_adaptability'] * \
            np.tanh(analysis['temporal_relevance'] * analysis['contextual_depth'])
        
        # Check for initiative opportunities
        if self._should_take_initiative(message_text, user_id):
            analysis["requires_initiative"] = True
            # Generate specific initiative suggestions
            analysis["suggested_initiatives"] = self._generate_initiative_options(message_text, topics)
        
        # Check if this could lead to a new autonomous goal
        if self._can_form_goal(message_text, topics):
            analysis["opportunity_for_goal"] = True
        
        # Meta-cognitive monitoring
        if self.core_values['metacognition'] > 0.6:
            analysis['metacognitive_override'] = self._apply_metacognitive_filters(analysis)
        
        # Update user model based on this message
        self._update_user_model(user_id, message_text, analysis)
        
        return analysis
        
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
        """Extract potential topics/interests from text using enhanced NLP-inspired techniques"""
        # Advanced keyword extraction with context awareness
        
        topics = []
        
        # Define domains of interest and related keywords with weighted relevance
        domains = {
            "technology": ["computer", "tech", "software", "hardware", "AI", "robot", "code", "program", "app",
                          "digital", "internet", "blockchain", "data", "algorithm", "gadget", "device", "neural",
                          "quantum", "virtual", "cyber", "cloud", "server", "network", "encryption", "interface"],
            "science": ["science", "chemistry", "physics", "biology", "experiment", "theory", "research", 
                       "scientific", "study", "discover", "atom", "molecule", "particle", "evolution", "quantum",
                       "laboratory", "hypothesis", "analysis", "synthesis", "observation", "measurement"],
            "arts": ["art", "music", "painting", "drawing", "movie", "film", "book", "novel", "poetry", "dance",
                    "theater", "sculpture", "creative", "design", "fashion", "photography", "aesthetic", "composition",
                    "performance", "exhibition", "gallery", "artistic", "expression", "creativity", "imagination"],
            "philosophy": ["philosophy", "ethics", "moral", "exist", "conscious", "meaning", "truth", "reality",
                          "metaphysics", "epistemology", "logic", "reason", "thought", "concept", "idea", "ontology",
                          "phenomenology", "existential", "dialectic", "transcendental", "empiricism", "rationalism"],
            "games": ["game", "play", "gaming", "video game", "board game", "puzzle", "strategy", "rpg", "fps",
                     "console", "pc gaming", "mmorpg", "minecraft", "steam", "xbox", "playstation", "nintendo",
                     "simulation", "esports", "multiplayer", "achievement", "level", "character", "quest"],
            "nature": ["nature", "animal", "plant", "forest", "mountain", "ocean", "biology", "environment",
                      "ecosystem", "wildlife", "climate", "planet", "earth", "natural", "conservation", "biodiversity",
                      "sustainability", "ecology", "species", "habitat", "organic", "biosphere", "wilderness"],
            "culture": ["culture", "society", "tradition", "language", "history", "heritage", "custom",
                       "belief", "religion", "community", "identity", "diversity", "ethnicity", "value", "ritual",
                       "ceremony", "celebration", "festival", "mythology", "folklore", "anthropology", "sociology"]
        }
        
        # Domain weights for relevance scoring
        domain_weights = {
            "technology": 1.2,  # Slightly higher weight for tech topics
            "science": 1.1,
            "arts": 1.0,
            "philosophy": 1.0,
            "games": 0.9,
            "nature": 1.0,
            "culture": 1.0
        }
        
        lowered_text = text.lower()
        words = lowered_text.split()
        word_count = len(words)
        
        # Topic relevance scores
        topic_scores = {}
        
        # Check for keywords in each domain with contextual weighting
        found_domains = set()
        for domain, keywords in domains.items():
            domain_score = 0
            for keyword in keywords:
                # Different matching patterns with different weights
                exact_match = f" {keyword} " in f" {lowered_text} "
                start_match = lowered_text.startswith(f"{keyword} ")
                end_match = lowered_text.endswith(f" {keyword}") or f"{keyword}." in lowered_text
                
                if exact_match or start_match or end_match:
                    # Calculate position-based relevance (words near beginning get higher weight)
                    position = lowered_text.find(keyword) / max(1, len(lowered_text))
                    position_factor = 1.0 - (position * 0.5)  # Words at start get up to 50% boost
                    
                    # Calculate frequency-based relevance
                    frequency = lowered_text.count(keyword)
                    frequency_factor = min(1.5, 1.0 + (frequency * 0.1))  # Up to 50% boost for repeated terms
                    
                    # Combined relevance score
                    relevance = domain_weights[domain] * position_factor * frequency_factor
                    
                    # Store topic with its relevance score
                    if keyword not in topic_scores or relevance > topic_scores[keyword]:
                        topic_scores[keyword] = relevance
                    
                    found_domains.add(domain)
                    domain_score += relevance
            
            # Add domain itself if enough keywords were found
            if domain_score > 1.5:
                topic_scores[domain] = domain_score * 0.8  # Domains get 80% of cumulative keyword scores
        
        # Extract multi-word phrases (n-grams)
        for n in [2, 3]:  # Bigrams and trigrams
            if len(words) >= n:
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i+n])
                    
                    # Check if phrase is meaningful (contains at least one significant word)
                    has_significant_word = False
                    for domain_keywords in domains.values():
                        if any(kw in phrase for kw in domain_keywords):
                            has_significant_word = True
                            break
                    
                    if has_significant_word and len(phrase) > 5:  # Avoid very short phrases
                        # Position-based relevance for phrases
                        position = i / max(1, len(words) - n)
                        position_factor = 1.0 - (position * 0.3)  # Less position penalty for phrases
                        
                        # Length-based relevance (longer meaningful phrases are more specific)
                        length_factor = min(1.3, 1.0 + (len(phrase) * 0.05))
                        
                        relevance = position_factor * length_factor
                        topic_scores[phrase] = relevance
        
        # Filter and sort topics by relevance score
        common_words = {"what", "when", "where", "which", "who", "how", "why", "yes", "no", "maybe", 
                       "could", "would", "should", "will", "can", "do", "does", "is", "are", "was", "were"}
        
        # Get topics sorted by relevance score
        sorted_topics = sorted([(t, s) for t, s in topic_scores.items() if t.lower() not in common_words], 
                               key=lambda x: x[1], reverse=True)
        
        # Take top topics, ensuring we don't have too many
        max_topics = min(10, max(3, word_count // 20))  # Scale with message length
        topics = [t for t, s in sorted_topics[:max_topics]]
        
        return topics
    
    def _detect_emotion(self, text):
        """Enhanced emotion detection with intensity and spectrum analysis"""
        # Initialize emotion detection result
        result = {
            "primary_emotion": "neutral",
            "intensity": 0.0,
            "spectrum": {}
        }
        
        # Emotion lexicons with intensity weights
        emotion_lexicons = {
            "joy": ["happy", "joy", "delighted", "excited", "glad", "pleased", "thrilled", "wonderful", 
                   "love", "amazing", "excellent", "fantastic", "great", "awesome", "good", "positive"],
            "sadness": ["sad", "unhappy", "depressed", "gloomy", "miserable", "disappointed", "upset", 
                       "heartbroken", "grief", "sorrow", "regret", "lonely", "hopeless", "negative"],
            "anger": ["angry", "mad", "furious", "outraged", "annoyed", "irritated", "frustrated", 
                     "hate", "resent", "disgusted", "bitter", "hostile", "aggressive", "hate"],
            "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous", 
                    "panic", "dread", "horror", "terror", "concern", "uneasy", "apprehensive"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned", "unexpected", 
                        "wow", "whoa", "unbelievable", "incredible", "startled", "sudden"],
            "curiosity": ["curious", "interested", "intrigued", "wonder", "questioning", "exploring", 
                         "learning", "discovering", "fascinated", "inquisitive", "seeking"],
            "confusion": ["confused", "puzzled", "perplexed", "unsure", "uncertain", "ambiguous", 
                         "unclear", "misunderstood", "complicated", "complex", "difficult"],
            "trust": ["trust", "believe", "faith", "confident", "reliable", "dependable", "honest", 
                     "loyal", "sincere", "authentic", "genuine", "true"]
        }
        
        # Intensity modifiers with multipliers
        intensifiers = {
            "very": 1.5, "extremely": 1.8, "incredibly": 1.7, "really": 1.4, "so": 1.3,
            "absolutely": 1.6, "completely": 1.5, "totally": 1.5, "utterly": 1.7,
            "deeply": 1.6, "profoundly": 1.7, "immensely": 1.6, "tremendously": 1.7
        }
        
        diminishers = {
            "somewhat": 0.7, "slightly": 0.6, "a bit": 0.7, "a little": 0.6,
            "kind of": 0.7, "sort of": 0.7, "barely": 0.5, "hardly": 0.5
        }
        
        # Negation words that flip emotion valence
        negations = ["not", "no", "never", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot"]
        
        # Process text for emotion detection
        words = text.lower().split()
        
        # Initialize emotion scores
        emotion_scores = {emotion: 0.0 for emotion in emotion_lexicons.keys()}
        
        # Scan for emotion words with context
        for i, word in enumerate(words):
            # Check for emotion words
            for emotion, emotion_words in emotion_lexicons.items():
                if word in emotion_words or any(ew in word for ew in emotion_words):
                    # Base emotion intensity
                    intensity = 0.7
                    
                    # Check for preceding intensifiers or diminishers
                    if i > 0:
                        prev_word = words[i-1]
                        if prev_word in intensifiers:
                            intensity *= intensifiers[prev_word]
                        elif prev_word in diminishers:
                            intensity *= diminishers[prev_word]
                    
                    # Check for negations (within 3 words before)
                    negated = False
                    for j in range(max(0, i-3), i):
                        if words[j] in negations:
                            negated = True
                            break
                    
                    # Apply negation effect (flip to opposite emotion or reduce intensity)
                    if negated:
                        if emotion in ["joy", "trust"]:
                            emotion_scores["sadness"] += intensity * 0.7
                        elif emotion in ["sadness", "anger", "fear"]:
                            emotion_scores["neutral"] += intensity * 0.5
                        else:
                            emotion_scores[emotion] *= 0.3  # Greatly reduce the emotion
                    else:
                        emotion_scores[emotion] += intensity
        
        # Normalize emotion scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_score
                result["spectrum"][emotion] = emotion_scores[emotion]
        else:
            result["spectrum"] = {emotion: 0.0 for emotion in emotion_lexicons.keys()}
            result["spectrum"]["neutral"] = 1.0
        
        # Determine primary emotion and intensity
        if total_score > 0:
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            result["primary_emotion"] = primary_emotion[0]
            result["intensity"] = primary_emotion[1] * min(1.0, total_score / 5.0)  # Scale intensity
        else:
            result["primary_emotion"] = "neutral"
            result["intensity"] = 0.1  # Minimal baseline intensity
        
        return result
    
    def _assess_complexity(self, text):
        """Advanced assessment of message complexity with multiple dimensions"""
        # Initialize complexity assessment result
        result = {
            "level": "medium",
            "score": 0.5,
            "conceptual_density": 0.0,
            "linguistic_complexity": 0.0,
            "cognitive_load": 0.0
        }
        
        # Basic text statistics
        words = text.split()
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Complex word indicators
        complex_word_patterns = [
            r'\w{12,}',  # Very long words
            r'\w+tion\b', r'\w+sion\b', r'\w+ism\b',  # Abstract concepts
            r'\w+ology\b', r'\w+ity\b', r'\w+ment\b',  # Academic terms
            r'\w+ness\b', r'\w+ance\b', r'\w+ence\b'   # Abstract qualities
        ]
        
        complex_words = []
        for pattern in complex_word_patterns:
            complex_words.extend(re.findall(pattern, text.lower()))
        
        complex_word_ratio = len(complex_words) / max(1, word_count)
        
        # Conceptual terms (abstract concepts that require deeper processing)
        conceptual_terms = [
            "concept", "theory", "framework", "paradigm", "philosophy", "principle",
            "hypothesis", "analysis", "synthesis", "perspective", "context", "structure",
            "function", "process", "system", "mechanism", "dimension", "factor",
            "variable", "correlation", "causation", "implication", "inference", "logic",
            "reasoning", "argument", "evidence", "validity", "fallacy", "premise",
            "conclusion", "abstraction", "concrete", "relative", "absolute", "objective",
            "subjective", "empirical", "theoretical", "practical", "ethical", "moral",
            "aesthetic", "epistemological", "ontological", "metaphysical", "existential"
        ]
        
        conceptual_term_count = sum(1 for word in words if word.lower() in conceptual_terms)
        conceptual_density = conceptual_term_count / max(1, word_count)
        
        # Linguistic complexity indicators
        linguistic_markers = {
            "conjunctions": ["and", "but", "or", "so", "yet", "however", "therefore", "thus", "although", "since", "while"],
            "prepositions": ["in", "on", "at", "by", "with", "from", "to", "for", "about", "through", "between"],
            "pronouns": ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"],
            "articles": ["a", "an", "the"],
            "quantifiers": ["some", "many", "few", "several", "most", "all", "any", "each", "every"]
        }
        
        # Count linguistic markers
        marker_counts = {category: 0 for category in linguistic_markers}
        for word in words:
            word_lower = word.lower()
            for category, markers in linguistic_markers.items():
                if word_lower in markers:
                    marker_counts[category] += 1
        
        # Calculate linguistic diversity (higher diversity = more complex)
        linguistic_diversity = sum(count > 0 for count in marker_counts.values()) / len(marker_counts)
        
        # Calculate conjunction density (more conjunctions = more complex relationships)
        conjunction_density = marker_counts["conjunctions"] / max(1, word_count)
        
        # Combine factors for linguistic complexity
        linguistic_complexity = (
            0.3 * (avg_sentence_length / 20) +  # Normalize to ~0.3 for 20-word sentences
            0.3 * (avg_word_length / 6) +       # Normalize to ~0.3 for 6-letter words
            0.2 * complex_word_ratio +          # Complex word contribution
            0.1 * linguistic_diversity +        # Linguistic diversity contribution
            0.1 * conjunction_density           # Conjunction density contribution
        )
        
        # Cognitive load calculation
        cognitive_load = (
            0.4 * linguistic_complexity +      # Linguistic complexity contribution
            0.3 * conceptual_density +         # Conceptual density contribution
            0.2 * (word_count / 100) +         # Message length contribution (normalized to ~0.2 for 100 words)
            0.1 * (1 - marker_counts["pronouns"] / max(1, word_count))  # Less pronouns = more cognitive load
        )
        
        # Determine complexity level and score
        complexity_score = min(0.95, max(0.05, (linguistic_complexity + conceptual_density + cognitive_load) / 3))
        
        if complexity_score < 0.3:
            complexity_level = "simple"
        elif complexity_score < 0.6:
            complexity_level = "medium"
        else:
            complexity_level = "complex"
        
        # Populate result
        result["level"] = complexity_level
        result["score"] = complexity_score
        result["conceptual_density"] = conceptual_density
        result["linguistic_complexity"] = linguistic_complexity
        result["cognitive_load"] = cognitive_load
        
        return result
    
    def _calculate_contextual_depth(self, conversation_context):
        """Calculate the contextual depth of the current conversation"""
        if not conversation_context:
            return 0.0
            
        # Factors that contribute to contextual depth
        context_length = len(conversation_context) if isinstance(conversation_context, list) else 1
        
        # More context messages = deeper context, but with diminishing returns
        length_factor = min(0.8, np.log(1 + context_length) / 5)
        
        # Check for recurring themes or topics if context is a list of messages
        theme_continuity = 0.0
        if isinstance(conversation_context, list) and len(conversation_context) > 1:
            # Extract topics from each context message
            context_topics = []
            for ctx_msg in conversation_context[-min(5, len(conversation_context)):]:  # Look at last 5 messages max
                if isinstance(ctx_msg, str):
                    topics = self._extract_topics(ctx_msg)
                    context_topics.append(topics)
                elif isinstance(ctx_msg, dict) and "text" in ctx_msg:
                    topics = self._extract_topics(ctx_msg["text"])
                    context_topics.append(topics)
            
            # Calculate topic overlap between consecutive messages
            topic_overlaps = []
            for i in range(len(context_topics) - 1):
                set1 = set(context_topics[i])
                set2 = set(context_topics[i + 1])
                if set1 and set2:  # Ensure non-empty sets
                    overlap = len(set1.intersection(set2)) / max(1, min(len(set1), len(set2)))
                    topic_overlaps.append(overlap)
            
            # Average topic continuity
            if topic_overlaps:
                theme_continuity = sum(topic_overlaps) / len(topic_overlaps)
        
        # Combine factors for overall contextual depth
        contextual_depth = length_factor * 0.6 + theme_continuity * 0.4
        return min(1.0, contextual_depth)
    
    def _assess_temporal_relevance(self, message_text):
        """Assess how time-sensitive or temporally relevant a message is"""
        # Time-related keywords and their weights
        temporal_keywords = {
            "now": 0.9, "today": 0.8, "tonight": 0.8, "this morning": 0.8, "this afternoon": 0.8,
            "this evening": 0.8, "right now": 0.9, "immediately": 0.9, "instantly": 0.9,
            "currently": 0.7, "presently": 0.7, "at the moment": 0.7, "as we speak": 0.8,
            "tomorrow": 0.6, "next week": 0.5, "soon": 0.6, "shortly": 0.6,
            "yesterday": 0.4, "last week": 0.3, "recently": 0.5, "earlier": 0.4,
            "later": 0.5, "eventually": 0.3, "someday": 0.2, "in the future": 0.3
        }
        
        # Time-sensitive action verbs
        urgency_verbs = {
            "need": 0.7, "must": 0.8, "should": 0.6, "have to": 0.7, "require": 0.6,
            "urgent": 0.9, "emergency": 0.9, "critical": 0.8, "important": 0.7,
            "hurry": 0.8, "rush": 0.8, "quick": 0.7, "fast": 0.7, "rapid": 0.7,
            "deadline": 0.8, "due": 0.7, "overdue": 0.8
        }
        
        # Check for temporal keywords and urgency indicators
        message_lower = message_text.lower()
        
        # Calculate temporal relevance score
        temporal_score = 0.0
        matched_terms = 0
        
        # Check for temporal keywords
        for keyword, weight in temporal_keywords.items():
            if keyword in message_lower:
                temporal_score += weight
                matched_terms += 1
        
        # Check for urgency verbs
        for verb, weight in urgency_verbs.items():
            if verb in message_lower:
                temporal_score += weight
                matched_terms += 1
        
        # Normalize score
        if matched_terms > 0:
            temporal_score = temporal_score / matched_terms
        else:
            # Default moderate temporal relevance if no explicit time indicators
            temporal_score = 0.3
        
        return min(1.0, temporal_score)
    
    def _detect_dialectical_tension(self, message_text, conversation_context):
        """Detect dialectical tensions or contradictions in the conversation"""
        if not conversation_context:
            return 0.0
            
        # Contradiction indicators
        contradiction_phrases = [
            "but", "however", "although", "nevertheless", "nonetheless", "yet", "still",
            "on the other hand", "conversely", "in contrast", "instead", "rather",
            "despite", "in spite of", "regardless", "even though", "whereas"
        ]
        
        # Opposing concept pairs that might indicate dialectical tension
        opposing_concepts = [
            ("good", "bad"), ("right", "wrong"), ("true", "false"), ("yes", "no"),
            ("always", "never"), ("everything", "nothing"), ("everyone", "no one"),
            ("love", "hate"), ("agree", "disagree"), ("accept", "reject"),
            ("positive", "negative"), ("advantage", "disadvantage"), ("pro", "con"),
            ("support", "oppose"), ("increase", "decrease"), ("more", "less"),
            ("create", "destroy"), ("begin", "end"), ("start", "stop"),
            ("open", "close"), ("include", "exclude"), ("allow", "forbid")
        ]
        
        # Check for contradiction indicators
        message_lower = message_text.lower()
        contradiction_score = 0.0
        
        # Check for contradiction phrases
        for phrase in contradiction_phrases:
            if f" {phrase} " in f" {message_lower} ":
                contradiction_score += 0.3
                break  # Only count once even if multiple phrases are present
        
        # Check for opposing concepts
        for concept1, concept2 in opposing_concepts:
            if concept1 in message_lower and concept2 in message_lower:
                contradiction_score += 0.4
                break  # Only count once even if multiple oppositions are present
        
        # Check for self-contradiction within the message
        sentences = [s.strip().lower() for s in re.split(r'[.!?]+', message_lower) if s.strip()]
        if len(sentences) > 1:
            for i in range(len(sentences)):
                for j in range(i+1, len(sentences)):
                    # Check if one sentence contains a negation of something in another sentence
                    words_i = set(sentences[i].split())
                    words_j = set(sentences[j].split())
                    
                    # Check for "not" + same word in different sentences
                    if "not" in words_i or "no" in words_i or "don't" in words_i or "doesn't" in words_i:
                        common_words = words_i.intersection(words_j)
                        if common_words:
                            contradiction_score += 0.3
                            break
        
        # Check for contradiction with previous context
        if isinstance(conversation_context, list) and conversation_context:
            last_message = conversation_context[-1]
            if isinstance(last_message, str):
                last_message_lower = last_message.lower()
            elif isinstance(last_message, dict) and "text" in last_message:
                last_message_lower = last_message["text"].lower()
            else:
                last_message_lower = ""
                
            # Check for direct contradiction indicators between messages
            for phrase in contradiction_phrases:
                if f" {phrase} " in f" {message_lower} ":
                    contradiction_score += 0.2
                    break
            
            # Check for opposing concepts between current and previous message
            for concept1, concept2 in opposing_concepts:
                if (concept1 in message_lower and concept2 in last_message_lower) or \
                   (concept2 in message_lower and concept1 in last_message_lower):
                    contradiction_score += 0.3
                    break
        
        return min(1.0, contradiction_score)
    
    def _are_terms_related(self, term1, term2):
        """Determine if two terms are semantically related"""
        # Direct substring match
        if term1.lower() in term2.lower() or term2.lower() in term1.lower():
            return True
            
        # Known related term pairs
        related_terms = {
            "ai": ["artificial intelligence", "machine learning", "neural network", "deep learning", "algorithm"],
            "computer": ["laptop", "desktop", "pc", "mac", "hardware", "software"],
            "internet": ["web", "online", "website", "browser", "network"],
            "science": ["physics", "chemistry", "biology", "research", "experiment"],
            "art": ["painting", "drawing", "sculpture", "creativity", "design"],
            "music": ["song", "melody", "rhythm", "instrument", "concert"],
            "philosophy": ["ethics", "metaphysics", "logic", "epistemology", "ontology"],
            "game": ["play", "gaming", "entertainment", "fun", "competition"],
            "book": ["novel", "reading", "literature", "story", "author"],
            "movie": ["film", "cinema", "actor", "director", "scene"],
            "nature": ["environment", "ecology", "wildlife", "outdoors", "planet"],
            "technology": ["tech", "innovation", "digital", "electronic", "gadget"]
        }
        
        # Check if terms are in related groups
        term1_lower = term1.lower()
        term2_lower = term2.lower()
        
        for key, related in related_terms.items():
            if (term1_lower == key or term1_lower in related) and (term2_lower == key or term2_lower in related):
                return True
        
        # Character-level similarity for short terms
        if len(term1) < 10 and len(term2) < 10:
            common_chars = set(term1_lower) & set(term2_lower)
            if len(common_chars) >= min(len(term1) * 0.7, len(term2) * 0.7):
                return True
        
        return False
    
    def _apply_metacognitive_filters(self, analysis):
        """Apply metacognitive filters to refine analysis based on higher-order reasoning"""
        # Initialize metacognitive override results
        metacognitive_result = {
            "applied_filters": [],
            "adjusted_values": {},
            "reasoning": []
        }
        
        # 1. Emotional intensity moderation filter
        if "emotional_intensity" in analysis and analysis["emotional_intensity"] > 0.8:
            # Check if the high emotion is justified by content
            if "complexity_score" in analysis and analysis["complexity_score"] > 0.7:
                # Complex content with high emotion might be overestimated
                metacognitive_result["adjusted_values"]["emotional_intensity"] = analysis["emotional_intensity"] * 0.8
                metacognitive_result["applied_filters"].append("emotional_intensity_moderation")
                metacognitive_result["reasoning"].append(
                    "High emotional intensity moderated due to complex content suggesting nuanced rather than purely emotional response"
                )
        
        # 2. Contextual coherence filter
        if "contextual_depth" in analysis and "topics" in analysis:
            if analysis["contextual_depth"] > 0.7 and len(analysis["topics"]) > 5:
                # Many topics with deep context might indicate topic drift
                primary_topics = analysis["topics"][:3]  # Focus on top 3 topics
                metacognitive_result["adjusted_values"]["focused_topics"] = primary_topics
                metacognitive_result["applied_filters"].append("contextual_coherence")
                metacognitive_result["reasoning"].append(
                    "Applied topic focusing due to potential topic drift in a deep contextual conversation"
                )
        
        # 3. Cognitive dissonance detection
        if "dialectical_tension" in analysis and analysis["dialectical_tension"] > 0.6:
            # High dialectical tension might indicate cognitive dissonance
            metacognitive_result["adjusted_values"]["cognitive_dissonance"] = True
            metacognitive_result["applied_filters"].append("cognitive_dissonance_detection")
            metacognitive_result["reasoning"].append(
                "Detected potential cognitive dissonance due to high dialectical tension in message"
            )
        
        # 4. Temporal relevance adjustment
        if "temporal_relevance" in analysis and "contextual_depth" in analysis:
            # Adjust temporal relevance based on conversation depth
            if analysis["contextual_depth"] > 0.8 and analysis["temporal_relevance"] < 0.4:
                # Deep conversations might have implicit temporal relevance
                metacognitive_result["adjusted_values"]["temporal_relevance"] = analysis["temporal_relevance"] + 0.2
                metacognitive_result["applied_filters"].append("temporal_relevance_adjustment")
                metacognitive_result["reasoning"].append(
                    "Increased temporal relevance due to deep contextual conversation suggesting ongoing importance"
                )
        
        # 5. Epistemic curiosity modulation
        if "epistemic_curiosity" in analysis and "complexity_score" in analysis:
            if analysis["epistemic_curiosity"] > 0.8 and analysis["complexity_score"] < 0.3:
                # High curiosity about simple topics might be overestimated
                metacognitive_result["adjusted_values"]["epistemic_curiosity"] = analysis["epistemic_curiosity"] * 0.8
                metacognitive_result["applied_filters"].append("curiosity_modulation")
                metacognitive_result["reasoning"].append(
                    "Moderated high epistemic curiosity for relatively simple content"
                )
        
        # 6. Initiative threshold dynamic adjustment
        if "requires_initiative" in analysis and analysis["requires_initiative"]:
            # Check if we have sufficient context to justify initiative
            if "contextual_depth" in analysis and analysis["contextual_depth"] < 0.4:
                # Low context depth might make initiative premature
                metacognitive_result["adjusted_values"]["requires_initiative"] = False
                metacognitive_result["applied_filters"].append("initiative_threshold_adjustment")
                metacognitive_result["reasoning"].append(
                    "Suppressed initiative due to insufficient contextual depth for confident action"
                )
        
        # 7. Conceptual integration filter
        if "conceptual_density" in analysis and analysis["conceptual_density"] > 0.7:
            # High concept density might need integration with existing knowledge
            metacognitive_result["adjusted_values"]["requires_knowledge_integration"] = True
            metacognitive_result["applied_filters"].append("conceptual_integration")
            metacognitive_result["reasoning"].append(
                "Flagged need for knowledge integration due to high conceptual density"
            )
            
        return metacognitive_result
        
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
        # Gelişmiş duygu analizi için çok katmanlı yaklaşım
        emotion_vectors = {
            'joy': {'keywords': ['mutlu', 'heyecan', 'müthiş', 'harika', 'sevindim'], 'decay_rate': 0.95},
            'sadness': {'keywords': ['üzgün', 'hayal kırıklığı', 'kayıp', 'yıkılmış'], 'decay_rate': 0.92},
            'anger': {'keywords': ['sinir', 'öfke', 'kızgın', 'hiddet'], 'decay_rate': 0.88},
            'surprise': {'keywords': ['şaşırdım', 'vay canına', 'inanılmaz'], 'decay_rate': 0.96},
            'fear': {'keywords': ['korku', 'endişe', 'panik', 'tedirgin'], 'decay_rate': 0.9},
            'trust': {'keywords': ['güven', 'inanç', 'emin'], 'decay_rate': 0.93},
            'anticipation': {'keywords': ['beklenti', 'merak', 'umut'], 'decay_rate': 0.94},
            'disgust': {'keywords': ['tiksinme', 'iğrenme', 'nefret'], 'decay_rate': 0.85}
        }

        # Konuşma bağlamını dikkate alan dinamik ağırlıklandırma
        context_impact = 1 + (self.core_values['context_awareness'] * 0.5)
        temporal_boost = 1 + (self.core_values['temporal_awareness'] * 0.3)

        # Çok boyutlu duygu vektörü hesaplama
        emotion_profile = defaultdict(float)
        for emotion, data in emotion_vectors.items():
            for keyword in data['keywords']:
                if keyword in text.lower():
                    emotion_profile[emotion] += \
                        self.core_values['emotional_depth'] * \
                        context_impact * \
                        temporal_boost * \
                        (1 / (1 + np.exp(-text.count(keyword))))

        # Zamansal bozulma uygula
        for emotion in emotion_profile:
            emotion_profile[emotion] *= emotion_vectors[emotion]['decay_rate'] ** \
                self._get_time_since_last_emotion(emotion)

        # Meta-bilişsel filtreleme
        if self.core_values['metacognition'] > 0.7:
            max_emo = max(emotion_profile.values(), default=0)
            for emo in emotion_profile:
                emotion_profile[emo] = (emotion_profile[emo] / max_emo) if max_emo > 0 else 0

        # Nöroplastisite etkisi
        plasticity_effect = 1 + (self.core_values['neuroplasticity_factor'] * 0.5)
        return {k: v * plasticity_effect for k, v in emotion_profile.items()}
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