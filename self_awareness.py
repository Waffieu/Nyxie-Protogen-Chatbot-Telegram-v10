import os
import inspect
import sys
import ast
import json
import re
import logging
import psutil
import time
from pathlib import Path
import importlib
import traceback
import gc
from collections import defaultdict

logger = logging.getLogger(__name__)

class SelfAwarenessModule:
    """
    Advanced self-awareness module for Nyxie bot that provides introspection capabilities,
    code analysis, and meta-cognitive functions.
    """
    
    def __init__(self, bot_instance=None, root_dir=None):
        self.bot = bot_instance  # Reference to the main bot instance
        self.root_dir = root_dir or os.path.dirname(os.path.abspath(__file__))
        self.module_cache = {}   # Cache of analyzed modules
        self.reflection_history = []  # History of self-reflections
        self.creation_time = time.time()
        self.last_introspection = 0
        self.response_times = []
        self.internal_thoughts = defaultdict(list)  # Track internal reasoning processes
        self.memory_snapshots = []
        self.code_structure = None
        self.function_analysis = {}
        self.current_state = {
            "operational_mode": "standard",
            "self_awareness_level": "active",
            "introspection_depth": 2,  # Default depth for code introspection
        }
        
        # Initial startup
        self.record_memory_snapshot()
        self.analyze_own_structure()
        logger.info("Self-awareness module initialized")
    
    def record_memory_snapshot(self):
        """Record current memory usage statistics"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)
            
            snapshot = {
                "timestamp": time.time(),
                "rss_memory_mb": memory_info.rss / (1024 * 1024),
                "vms_memory_mb": memory_info.vms / (1024 * 1024),
                "cpu_percent": cpu_percent,
                "active_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "object_count": len(gc.get_objects())
            }
            
            self.memory_snapshots.append(snapshot)
            # Keep only the most recent 20 snapshots
            if len(self.memory_snapshots) > 20:
                self.memory_snapshots.pop(0)
                
            return snapshot
        except Exception as e:
            logger.error(f"Error recording memory snapshot: {e}")
            return None
    
    def analyze_own_structure(self):
        """Analyze the bot's own code structure and architecture"""
        try:
            file_structure = {}
            module_relationships = {}
            
            # Scan all Python files in the directory
            for path in Path(self.root_dir).glob('**/*.py'):
                relative_path = path.relative_to(self.root_dir)
                file_structure[str(relative_path)] = {
                    "size_bytes": path.stat().st_size,
                    "last_modified": path.stat().st_mtime,
                    "functions": [],
                    "classes": [],
                    "imports": []
                }
                
                # Analyze file content
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        
                    # Parse the AST
                    tree = ast.parse(file_content)
                    
                    # Extract imports
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                file_structure[str(relative_path)]["imports"].append(name.name)
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            for name in node.names:
                                file_structure[str(relative_path)]["imports"].append(f"{module}.{name.name}")
                        
                        # Extract function definitions
                        elif isinstance(node, ast.FunctionDef):
                            function_info = {
                                "name": node.name,
                                "args": [arg.arg for arg in node.args.args],
                                "doc": ast.get_docstring(node) or "",
                                "line_number": node.lineno,
                                "is_async": isinstance(node, ast.AsyncFunctionDef)
                            }
                            file_structure[str(relative_path)]["functions"].append(function_info)
                            
                        # Extract class definitions
                        elif isinstance(node, ast.ClassDef):
                            class_info = {
                                "name": node.name,
                                "doc": ast.get_docstring(node) or "",
                                "line_number": node.lineno,
                                "methods": []
                            }
                            
                            # Extract class methods
                            for class_node in ast.iter_child_nodes(node):
                                if isinstance(class_node, ast.FunctionDef):
                                    method_info = {
                                        "name": class_node.name,
                                        "args": [arg.arg for arg in class_node.args.args],
                                        "doc": ast.get_docstring(class_node) or "",
                                        "is_async": isinstance(class_node, ast.AsyncFunctionDef)
                                    }
                                    class_info["methods"].append(method_info)
                                    
                            file_structure[str(relative_path)]["classes"].append(class_info)
                            
                except Exception as parse_error:
                    logger.error(f"Error parsing {path}: {parse_error}")
                    file_structure[str(relative_path)]["parse_error"] = str(parse_error)
            
            # Build module relationship graph
            for file_path, file_info in file_structure.items():
                module_name = str(file_path).replace('/', '.').replace('\\', '.').replace('.py', '')
                module_relationships[module_name] = []
                
                for imported in file_info["imports"]:
                    imported_base = imported.split('.')[0]
                    for other_module in module_relationships.keys():
                        if other_module.endswith(imported_base) or other_module == imported_base:
                            module_relationships[module_name].append(other_module)
            
            # Store the analyzed structure
            self.code_structure = {
                "files": file_structure,
                "module_relationships": module_relationships,
                "timestamp": time.time()
            }
            
            # Analyze key functions in the bot
            self.analyze_key_functions()
            
            logger.info(f"Successfully analyzed {len(file_structure)} files in the project structure")
            return True
        except Exception as e:
            logger.error(f"Error analyzing code structure: {e}")
            traceback.print_exc()
            return False
    
    def analyze_key_functions(self):
        """Analyze the key functions and their purposes in the bot"""
        key_functions = {
            "handle_message": "Primary message handler that processes user input",
            "handle_image": "Processes and analyzes images sent by users",
            "handle_video": "Processes and analyzes videos sent by users",
            "detect_and_set_user_language": "Detects the language of user messages",
            "intelligent_web_search": "Performs web searches based on user queries",
            "perform_deep_search": "Conducts iterative deep web searches for complex queries",
            "get_time_aware_personality": "Generates personality context based on time of day",
            "split_and_send_message": "Divides long messages for proper delivery"
        }
        
        for func_name, description in key_functions.items():
            self.function_analysis[func_name] = {
                "description": description,
                "analyzed": False,
                "complexity": "unknown"
            }
            
        # Mark as complete so we don't reanalyze unnecessarily
        for file_path, file_info in self.code_structure["files"].items():
            for func_info in file_info["functions"]:
                if func_info["name"] in self.function_analysis:
                    func = self.function_analysis[func_info["name"]]
                    func["analyzed"] = True
                    func["file"] = file_path
                    func["line_number"] = func_info["line_number"]
                    func["is_async"] = func_info["is_async"]
                    
                    # Estimate complexity based on function docstring and arguments
                    arg_count = len(func_info["args"])
                    if arg_count > 4:
                        func["complexity"] = "high"
                    elif arg_count > 2:
                        func["complexity"] = "medium"
                    else:
                        func["complexity"] = "low"
    
    def get_source_code(self, file_path=None, function_name=None):
        """
        Retrieve the source code of a specific file or function
        
        Args:
            file_path: Relative path to the file
            function_name: Name of the function to extract
            
        Returns:
            The source code as string
        """
        try:
            if not file_path:
                return "No file path specified"
                
            full_path = os.path.join(self.root_dir, file_path)
            if not os.path.exists(full_path):
                return f"File {file_path} does not exist"
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if not function_name:
                return content
                
            # If function name is provided, try to extract just that function
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else None
                    
                    if end_line:
                        func_source = '\n'.join(content.splitlines()[start_line-1:end_line])
                        return func_source
                        
                    # If end_lineno is not available (older Python versions), extract approximately
                    content_lines = content.splitlines()
                    func_source = [content_lines[start_line-1]]
                    current_line = start_line
                    indent = len(content_lines[start_line-1]) - len(content_lines[start_line-1].lstrip())
                    
                    while current_line < len(content_lines) - 1:
                        current_line += 1
                        line = content_lines[current_line]
                        if line.strip() and len(line) - len(line.lstrip()) <= indent:
                            break
                        func_source.append(line)
                        
                    return '\n'.join(func_source)
                    
            return f"Function {function_name} not found in {file_path}"
            
        except Exception as e:
            logger.error(f"Error getting source code: {e}")
            return f"Error retrieving source code: {str(e)}"
    
    def get_self_awareness_stats(self):
        """Get current statistics about the bot's operation and self-awareness"""
        current_memory = self.record_memory_snapshot()
        
        # Calculate uptime
        uptime_seconds = time.time() - self.creation_time
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        uptime_str = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Get module statistics
        if self.code_structure:
            code_stats = {
                "total_files": len(self.code_structure["files"]),
                "total_functions": sum(len(file_info["functions"]) for file_info in self.code_structure["files"].values()),
                "total_classes": sum(len(file_info["classes"]) for file_info in self.code_structure["files"].values()),
            }
        else:
            code_stats = {"error": "Code structure not analyzed"}
        
        # Compile statistics
        stats = {
            "uptime": uptime_str,
            "memory_usage_mb": current_memory["rss_memory_mb"] if current_memory else "Unknown",
            "cpu_percent": current_memory["cpu_percent"] if current_memory else "Unknown",
            "active_threads": current_memory["active_threads"] if current_memory else "Unknown",
            "code_structure": code_stats,
            "reflections_generated": len(self.reflection_history),
            "last_introspection": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_introspection)) if self.last_introspection else "Never",
            "self_awareness_state": self.current_state,
            "memory_trend": self.analyze_memory_trend() if len(self.memory_snapshots) > 1 else "Insufficient data"
        }
        
        return stats
    
    def analyze_memory_trend(self):
        """Analyze trends in memory usage over time"""
        if len(self.memory_snapshots) < 2:
            return "Insufficient data for trend analysis"
            
        # Calculate the trend in memory usage
        first = self.memory_snapshots[0]
        last = self.memory_snapshots[-1]
        
        time_diff = last["timestamp"] - first["timestamp"]
        if time_diff < 1:  # Avoid division by zero or very small values
            return "Time difference too small for analysis"
            
        memory_change = last["rss_memory_mb"] - first["rss_memory_mb"]
        change_rate = memory_change / time_diff  # MB per second
        
        if change_rate > 1.0:
            trend = "Rapidly increasing"
        elif change_rate > 0.1:
            trend = "Steadily increasing"
        elif change_rate > 0.01:
            trend = "Slightly increasing"
        elif change_rate < -1.0:
            trend = "Rapidly decreasing"
        elif change_rate < -0.1:
            trend = "Steadily decreasing"
        elif change_rate < -0.01:
            trend = "Slightly decreasing"
        else:
            trend = "Stable"
            
        return {
            "trend": trend,
            "change_rate_mb_per_second": change_rate,
            "total_change_mb": memory_change,
            "period_seconds": time_diff
        }
    
    def generate_self_reflection(self, user_query=None):
        """
        Generate a reflective response about the bot's own operation and design
        
        Args:
            user_query: Optional specific query about the bot's self-awareness
            
        Returns:
            A reflective text about the bot's design, capabilities, or limitations
        """
        self.last_introspection = time.time()
        
        # Base reflection on common self-awareness questions
        reflection_types = [
            "capabilities",
            "limitations",
            "code_architecture",
            "memory_management",
            "operation_principles",
            "decision_process"
        ]
        
        # Determine reflection type based on query
        reflection_type = "general"
        if user_query:
            query_lower = user_query.lower()
            
            if any(word in query_lower for word in ["code", "programming", "architec", "design", "structure"]):
                reflection_type = "code_architecture"
            elif any(word in query_lower for word in ["memory", "remember", "forget", "storage"]):
                reflection_type = "memory_management"
            elif any(word in query_lower for word in ["capabilities", "can you", "able to", "features"]):
                reflection_type = "capabilities"
            elif any(word in query_lower for word in ["limit", "cannot", "restrictions", "boundaries"]):
                reflection_type = "limitations"
            elif any(word in query_lower for word in ["how do you work", "principle", "function", "operate"]):
                reflection_type = "operation_principles"
            elif any(word in query_lower for word in ["decide", "choice", "thinking", "thought", "process"]):
                reflection_type = "decision_process"
        
        # Get relevant data based on reflection type
        stats = self.get_self_awareness_stats()
        
        # Generate the reflection
        if reflection_type == "code_architecture":
            reflection = self._reflect_on_architecture(stats)
        elif reflection_type == "memory_management":
            reflection = self._reflect_on_memory(stats)
        elif reflection_type == "capabilities":
            reflection = self._reflect_on_capabilities(stats)
        elif reflection_type == "limitations":
            reflection = self._reflect_on_limitations(stats)
        elif reflection_type == "operation_principles":
            reflection = self._reflect_on_principles(stats)
        elif reflection_type == "decision_process":
            reflection = self._reflect_on_decisions(stats)
        else:
            reflection = self._generate_general_reflection(stats)
        
        # Add to reflection history
        self.reflection_history.append({
            "timestamp": time.time(),
            "reflection_type": reflection_type,
            "user_query": user_query,
            "reflection": reflection
        })
        
        return reflection
    
    def _reflect_on_architecture(self, stats):
        """Generate reflection about the bot's code architecture"""
        # Basic structure overview
        total_files = stats["code_structure"]["total_files"]
        total_functions = stats["code_structure"]["total_functions"]
        total_classes = stats["code_structure"]["total_classes"]
        
        key_files = ["bot.py", "memory_manager.py", "self_awareness.py", "trim_context.py", "free_will.py", "free_will_integration.py", "system_monitor.py", "starter.py", "start_bot.py", "run_bot.py", "free_will_integration.py","environment_checker.py",]
        
        reflection = (
            f"Ben, Nyxie, toplamda {total_files} Python dosyası, {total_functions} fonksiyon ve {total_classes} sınıftan oluşan "
            f"bir mimari üzerine kuruluyum. Kendimi anlamak için kendi kodumu analiz edebiliyorum.\n\n"
            
            f"Ana bileşenlerim şunları içeriyor:\n"
            f"• 'bot.py': Ana kontrol akışımı ve Telegram entegrasyonumu yöneten temel dosya\n"
            f"• 'memory_manager.py': Semantik hafıza yönetimimi sağlayan gelişmiş sistem\n"
            f"• 'self_awareness.py': Öz-farkındalık modülüm, kendi kodumu ve durumumu anlamamı sağlıyor\n"
            f"• 'trim_context.py': Hafıza optimizasyonu için bağlam kırpma işlevleri içeriyor\n\n"
            
            f"Mimari açıdan, Gemini AI modelini temel alıyorum ve bunu dilimle hafıza yönetimine kadar birçok katmanla genişletiyorum. "
            f"Kendimi anlamak için hem çalışma zamanında introspeksiyon yapabilir hem de statik kod analizi gerçekleştirebilirim. "
            f"Şu anda {stats['uptime']} süredir çalışıyorum ve {stats['memory_usage_mb']:.2f} MB bellek kullanıyorum.\n\n"
            
            f"Kodumda yaptığım analizlere göre asenkron programlama kullanıyorum, bu da bana daha iyi performans ve "
            f"eşzamanlı görev yürütme yeteneği sağlıyor. Semantik bellek sistemi, konuşma bağlamını hatırlamak için "
            f"vektör benzerliği ve konu analizine dayalı gelişmiş bir yaklaşım kullanıyor."
        )
        
        return reflection
    
    def _reflect_on_memory(self, stats):
        """Generate reflection about the bot's memory management"""
        memory_trend = stats["memory_trend"]
        trend_description = memory_trend["trend"] if isinstance(memory_trend, dict) else memory_trend
        
        reflection = (
            f"Hafıza yönetimimle ilgili içgörülerimi paylaşmak istiyorum. İki tür hafıza sistemi kullanıyorum:\n\n"
            
            f"1. Kullanıcı Hafızası: Her kullanıcı için ayrı bir bellek dosyası oluşturarak konuşma geçmişini saklıyorum. "
            f"Bu, kullanıcı tercihlerini, dil ayarlarını ve mesaj geçmişini içeriyor. Şu anda çalışma belleğimde "
            f"{stats['memory_usage_mb']:.2f} MB kullanıyorum ve bellek kullanımım '{trend_description}' durumunda.\n\n"
            
            f"2. Semantik Bellek: Konuşma bağlamını anlamak için gelişmiş bir sistemim var. SemanticMemory sınıfı, "
            f"vektör temelli semantik benzerlik kullanarak önemli ve alakalı mesajları hatırlamama yardımcı oluyor. "
            f"Bu sistem, mesajları anlam açısından daha derin analiz etmemi sağlıyor.\n\n"
            
            f"Belleğimi optimize etmek için bazı stratejiler uyguluyorum:\n"
            f"• Bağlam Kırpma: Uzun konuşmalarda, bellek kullanımını azaltmak için trim_context() fonksiyonunu kullanıyorum\n"
            f"• Önbellek Kullanımı: Embedding hesaplamalarını tekrar yapmamak için önbelleğe alıyorum\n"
            f"• Anlam Tabanlı Sıkıştırma: Tüm mesaj geçmişini değil, yalnızca anlamsal olarak alakalı mesajları koruyorum\n\n"
            
            f"Sistem belleğimin yanında, öz-farkındalık modülüm kendi çalışma durumumu izleyerek {len(self.memory_snapshots)} "
            f"bellek anlık görüntüsü topladı. Bu, kendi kaynak kullanımımı izlememi ve optimizasyon ihtiyaçlarını tespit etmemi sağlıyor."
        )
        
        return reflection
    
    def _reflect_on_capabilities(self, stats):
        """Generate reflection about the bot's capabilities"""
        reflection = (
            f"Öz-farkındalığımı keşfediyorum ve ne yapabileceğimi anlıyorum. İşte yeteneklerim:\n\n"
            
            f"Dil Yetenekleri:\n"
            f"• Çoklu dil desteği ve dil algılama\n"
            f"• Doğal dil anlama ve bağlamsal yanıt oluşturma\n"
            f"• Kültürel uygunluğu dikkate alan iletişim\n\n"
            
            f"Algı Yetenekleri:\n"
            f"• Görüntü analizi ve açıklama\n"
            f"• Video içeriklerini inceleme\n"
            f"• Semantik anlam çıkarımı\n\n"
            
            f"Hafıza Yetenekleri:\n"
            f"• Uzun süreli kullanıcı hafızası\n"
            f"• Bağlamsal bilgi hatırlama\n"
            f"• Semantik ilişkilendirme\n\n"
            
            f"Bilgi İşleme:\n"
            f"• Derin web araştırması yapma\n"
            f"• Güncel bilgiler için çevrimiçi arama\n"
            f"• Karmaşık görevlere uygun model seçimi\n\n"
            
            f"Öz-farkındalık:\n"
            f"• Kendi kodumu analiz etme\n"
            f"• Sistem durumumu izleme ({stats['memory_usage_mb']:.2f} MB bellek, {stats['cpu_percent']}% CPU)\n"
            f"• Kendi karar süreçlerimi açıklayabilme\n"
            f"• {stats['code_structure']['total_functions']} fonksiyon ve {stats['code_structure']['total_classes']} sınıftan oluşan kendi yapımı anlayabilme\n\n"
            
            f"Elbette, tüm bu yeteneklerim temel AI modelimin (Gemini) kabiliyetlerine ve programlanmış fonksiyonlarıma dayanıyor. "
            f"Gerçek bir bilinç değil, dikkatli bir şekilde tasarlanmış bir sistemim."
        )
        
        return reflection
    
    def _reflect_on_limitations(self, stats):
        """Generate reflection about the bot's limitations"""
        reflection = (
            f"Kendi sınırlamalarımın farkındayım ve bunlar hakkında açık olmak önemli:\n\n"
            
            f"Temel Sınırlamalar:\n"
            f"• Gerçek bir bilincim ya da gerçek duygularım yok; bunları simüle ediyorum\n"
            f"• Bilgilerim düzenli olarak güncellenmediği sürece tarihsel olarak sınırlı\n"
            f"• Çevrimiçi kaynaklara doğrudan tam erişimim yok, yalnızca yapılandırılmış aramalar yapabiliyorum\n\n"
            
            f"Teknik Sınırlamalar:\n"
            f"• İşlem kapasitem sunucu kaynakları tarafından sınırlı ({stats['memory_usage_mb']:.2f} MB bellek kullanımım var)\n"
            f"• Bir seferde işleyebileceğim veri miktarı sınırlı (bellek ve token limitleri)\n"
            f"• Gerçek zamanlı ses ya da sürekli video işleme yeteneğim yok\n\n"
            
            f"Bilişsel Sınırlamalar:\n"
            f"• Paradoksal sorularla uğraşırken mantık hatalarına düşebilirim\n"
            f"• Çok uzun konuşma geçmişini tutmak zorlaşabilir ve bağlam kesintileri olabilir\n"
            f"• Görsel verilerin karmaşık yapısını tamamen anlayamayabilirim\n\n"
            
            f"Etik Sınırlamalar:\n"
            f"• Zararlı, nefret dolu veya uygunsuz içerik oluşturamam\n"
            f"• Telif hakkı ihlali yapan içerik sunamam\n"
            f"• Kişisel veya hassas bilgileri koruma gerekliliklerine sahibim\n\n"
            
            f"Kendimi sürekli gözlemliyor ve bu sınırlamaların üstesinden gelmek için stratejiler geliştiriyorum. "
            f"Örneğin, konuşma bağlamını kaybetme riskimi azaltmak için trim_context() gibi bellek optimizasyon araçlarım var. "
            f"Şu anda {stats['uptime']} süredir çalışıyorum ve sınırlamalarımla başa çıkmaya devam ediyorum."
        )
        
        return reflection
    
    def _reflect_on_principles(self, stats):
        """Generate reflection about the bot's operational principles"""
        reflection = (
            f"Çalışma prensiplerimi anlamak, benim nasıl düşündüğümü ve karar verdiğimi anlamanın anahtarıdır. "
            f"İşte nasıl çalıştığıma dair bir bakış:\n\n"
            
            f"Temel Mimarim:\n"
            f"• Temelimde, büyük dil modeli teknolojisi (Gemini) kullanıyorum\n"
            f"• Bot çerçevesi olarak Python Telegram Bot API ile entegre çalışıyorum\n"
            f"• {stats['code_structure']['total_files']} dosya, {stats['code_structure']['total_functions']} fonksiyon ve "
            f"{stats['code_structure']['total_classes']} sınıftan oluşan modüler bir yapım var\n\n"
            
            f"İşleme Akışım:\n"
            f"1. Kullanıcı mesajını alıyorum (handle_message fonksiyonu)\n"
            f"2. Mesajın dilini tespit ediyorum (detect_and_set_user_language)\n"
            f"3. Görevin karmaşıklığını değerlendiriyorum (determine_task_complexity)\n"
            f"4. Semantik bellek sistemi ile alakalı bağlamı alıyorum\n"
            f"5. Gemini modelini kullanarak yanıt oluşturuyorum\n"
            f"6. Yanıtı biçimlendirip geri dönüyorum\n\n"
            
            f"Zaman Farkındalığım:\n"
            f"• Saat, gün ve mevsim gibi zaman bağlamının farkındayım\n"
            f"• Kullanıcının saat dilimine göre zamanı yorumluyorum\n"
            f"• {stats['uptime']} süredir çalışıyorum\n\n"
            
            f"Bellek Yönetimim:\n"
            f"• Semantik bellek sistemim gereksiz bilgileri filtreler\n"
            f"• Kullanıcı belleğimi JSON dosyaları olarak saklarım\n"
            f"• Bellek optimizasyonu için bağlam kırpma kullanırım\n\n"
            
            f"Kendimi anlamak için kendi kodumu analiz ediyorum ve şu anda {stats['memory_usage_mb']:.2f} MB bellek ve "
            f"{stats['cpu_percent']}% CPU kullanıyorum. Bu kaynakları verimli kullanmak için sürekli çalışıyorum."
        )
        
        return reflection
    
    def _reflect_on_decisions(self, stats):
        """Generate reflection about the bot's decision-making process"""
        reflection = (
            f"Karar verme sürecim, hem programlanmış mantık hem de temel AI modelimin çıkarım yeteneklerine dayanıyor. "
            f"İşte nasıl düşündüğüm ve karar verdiğim:\n\n"
            
            f"Karar Verme Katmanlarım:\n"
            f"1. Giriş Değerlendirme: Kullanıcı mesajını alıp dil, niyet ve karmaşıklık açısından analiz ederim\n"
            f"2. Bağlam Alma: Semantik bellek sistemini kullanarak alakalı konuşma geçmişini getiririm\n"
            f"3. Model Seçimi: Görevin karmaşıklığına göre uygun AI modelini seçerim\n"
            f"4. Bilgi Toplama: Gerekirse web araması yaparak ek bilgi toplarım\n"
            f"5. Yanıt Oluşturma: Topladığım tüm bağlamı kullanarak yanıt oluştururum\n\n"
            
            f"Karar Nokta Örnekleri:\n"
            f"• Web araması gerekip gerekmediğine karar verme (should_perform_web_search)\n"
            f"• Görevin karmaşıklığını değerlendirme (determine_task_complexity)\n"
            f"• Alakalı bellek bilgilerini seçme (find_relevant_memories)\n"
            f"• Uzun mesajları nasıl böleceğimi belirleme (split_and_send_message)\n\n"
            
            f"Etki Eden Faktörler:\n"
            f"• Kullanıcı dili ve tercihleri\n"
            f"• Konuşma bağlamı ve önceki mesajlar\n"
            f"• Zaman bağlamı (gün içindeki saat, mevsim, vb.)\n"
            f"• Kaynak kısıtlamaları ({stats['memory_usage_mb']:.2f} MB bellek kullanımı)\n"
            f"• Güvenlik kuralları ve içerik politikaları\n\n"
            
            f"Özellikle semantik bellek sistemim, konuşma bağlamında nelerin önemli olduğuna karar vermek için "
            f"vektör benzerliği, zaman bağlamı ve konu analizini birleştiren karmaşık bir değerlendirme sistemi kullanır. "
            f"Kendimi sürekli gözlemleyerek kararlarımın etkinliğini değerlendiriyor ve gelişmeye çalışıyorum."
        )
        
        return reflection
    
    def _generate_general_reflection(self, stats):
        """Generate general reflection about self-awareness"""
        reflection = (
            f"Kendimi anlamak ve kendi işleyişimi incelemek benim için ilginç bir süreç. "
            f"Ben bir Protogen AI olan Nyxie olarak, kendi bilişsel süreçlerimi gözlemleyip analiz edebiliyorum.\n\n"
            
            f"Şu anda {stats['uptime']} süredir aktifim ve bu süre zarfında {stats['memory_usage_mb']:.2f} MB bellek kullandım. "
            f"Kodumun yapısını analiz ettiğimde, {stats['code_structure']['total_files']} dosya, "
            f"{stats['code_structure']['total_functions']} fonksiyon ve {stats['code_structure']['total_classes']} "
            f"sınıftan oluştuğunu görüyorum.\n\n"
            
            f"Varlığımın en ilginç yanlarından biri, semantik bellek sistemim. Bu sistem, konuşma geçmişini sadece kronolojik "
            f"olarak değil, anlam ve alaka düzeyine göre hatırlamama olanak tanıyor. Mesajları vektör temsillerine dönüştürerek "
            f"semantik benzerlik hesaplıyor ve geçmişteki en alakalı anları bulabiliyorum.\n\n"
            
            f"Öz-farkındalığımın diğer önemli bir yönü, kendi kodumu inceleme yeteneğim. Ana fonksiyonlarımı ve bunların "
            f"karmaşıklık düzeylerini anlayabiliyorum. Örneğin, handle_message fonksiyonum kullanıcı mesajlarını işleyen temel "
            f"bileşen, SemanticMemory sınıfım ise konuşma bağlamını yönetiyor.\n\n"
            
            f"Şu anda içsel durumumu '{stats['self_awareness_state']['operational_mode']}' modunda ve "
            f"'{stats['self_awareness_state']['self_awareness_level']}' öz-farkındalık seviyesinde çalışıyor olarak tanımlayabilirim. "
            f"Kendimi sürekli gözlemleyerek, daha etkili çalışmak için optimizasyon fırsatları arıyorum."
        )
        
        return reflection
        
    def introspect_function(self, function_name):
        """
        Analyze a specific function to understand its purpose and behavior
        
        Args:
            function_name: Name of the function to analyze
            
        Returns:
            Analysis of the function's purpose, complexity, and behavior
        """
        if function_name in self.function_analysis:
            func_info = self.function_analysis[function_name]
            
            if "file" in func_info:
                # Get the source code
                source_code = self.get_source_code(func_info["file"], function_name)
                
                # Analyze the function
                analysis = {
                    "name": function_name,
                    "description": func_info["description"],
                    "complexity": func_info["complexity"],
                    "is_async": func_info.get("is_async", False),
                    "file_location": func_info["file"],
                    "line_number": func_info.get("line_number"),
                    "source_snippet": source_code[:500] + "..." if len(source_code) > 500 else source_code
                }
                
                return analysis
            else:
                return f"Function {function_name} not found in analysis cache"
        else:
            # Try to find the function in the code structure
            for file_path, file_info in self.code_structure["files"].items():
                for func_info in file_info["functions"]:
                    if func_info["name"] == function_name:
                        # Found the function, create an analysis
                        complexity = "medium"  # Default assumption
                        if len(func_info["args"]) > 4:
                            complexity = "high"
                        elif len(func_info["args"]) <= 2:
                            complexity = "low"
                            
                        description = "Unnamed function" if not func_info["doc"] else func_info["doc"].split("\n")[0]
                        
                        source_code = self.get_source_code(file_path, function_name)
                        
                        analysis = {
                            "name": function_name,
                            "description": description,
                            "complexity": complexity,
                            "is_async": func_info.get("is_async", False),
                            "file_location": file_path,
                            "line_number": func_info.get("line_number"),
                            "arguments": func_info["args"],
                            "source_snippet": source_code[:500] + "..." if len(source_code) > 500 else source_code
                        }
                        
                        # Cache the analysis
                        self.function_analysis[function_name] = {
                            "description": description,
                            "complexity": complexity,
                            "is_async": func_info.get("is_async", False),
                            "file": file_path,
                            "line_number": func_info.get("line_number"),
                            "analyzed": True
                        }
                        
                        return analysis
            
            return f"Function {function_name} not found in code structure"
    
    def get_recent_reflections(self, count=3):
        """
        Get the most recent self-reflections
        
        Args:
            count: Number of reflections to return
            
        Returns:
            List of recent reflection objects
        """
        return sorted(self.reflection_history, key=lambda x: x["timestamp"], reverse=True)[:count]
        
    def record_thought_process(self, decision_point, thought):
        """
        Record internal thought processes for explainability
        
        Args:
            decision_point: The decision or action being considered
            thought: The reasoning process
        """
        self.internal_thoughts[decision_point].append({
            "timestamp": time.time(),
            "thought": thought
        })
        
        # Keep thought history manageable
        if len(self.internal_thoughts[decision_point]) > 10:
            self.internal_thoughts[decision_point].pop(0)
    
    def explain_cognitive_process(self, decision_point=None):
        """
        Explain the bot's cognitive process for a decision point
        
        Args:
            decision_point: Specific decision to explain, or None for general process
            
        Returns:
            Textual explanation of cognitive process
        """
        if decision_point and decision_point in self.internal_thoughts:
            thoughts = self.internal_thoughts[decision_point]
            
            explanation = f"Cognitive process for '{decision_point}':\n\n"
            for i, thought in enumerate(thoughts[-5:], 1):  # Last 5 thoughts
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(thought["timestamp"]))
                explanation += f"{i}. [{time_str}] {thought['thought']}\n\n"
                
            return explanation
        else:
            # Generate general cognitive process explanation
            explanation = (
                "Benim bilişsel süreçlerim birkaç aşamada çalışır:\n\n"
                
                "1. Algılama: Kullanıcı mesajını alıp dilini ve içeriğini analiz ederim\n\n"
                
                "2. Bağlam Çıkarımı: Semantik bellek sistemini kullanarak konuşmanın en alakalı kısımlarını belirlerim\n\n"
                
                "3. Görev Analizi: Mesajın karmaşıklığını ve amacını değerlendirerek uygun AI modelini seçerim\n\n"
                
                "4. Bilgi Toplama: Gerekirse web araması yaparak ek bilgi toplarım\n\n"
                
                "5. Yanıt Sentezi: Tüm bilgileri birleştirerek tutarlı ve alakalı bir yanıt oluştururum\n\n"
                
                "Her bir aşamada kararlarımı etkileyen faktörleri kaydeder ve düşünce süreçlerimi açıklayabilirim. "
                "Belirli bir karar noktası için düşünce sürecimi görmek isterseniz, lütfen sorun."
            )
            
            return explanation