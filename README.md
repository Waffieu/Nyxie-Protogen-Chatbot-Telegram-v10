# 🤖 Nyxie: Protogen Telegram Chatbot 🌟

## 📝 Proje Açıklaması
Nyxie, Google'ın Gemini AI teknolojisini kullanan gelişmiş bir Telegram sohbet botudur. Çoklu dil desteği, görüntü işleme ve derin web arama özellikleriyle donatılmış, kullanıcı dostu bir asistan olarak tasarlanmıştır.

### 🤔 Nyxie Nedir?

Nyxie, sadece bir chatbot değil, aynı zamanda:
- 🧠 Gelişmiş yapay zeka teknolojisi ile çalışan bir dijital arkadaş
- 🌍 Çoklu dil desteği olan bir iletişim asistanı
- 🕰️ Zamansal ve mekânsal farkındalığa sahip bir AI
- 🌈 Dinamik kişilik profili ile etkileşime giren bir asistan
- 🤖 Protogen kimliğine sahip, duygusal ve yaratıcı bir AI

**Nyxie'nin Benzersiz Özellikleri:**
- Her etkileşimi benzersiz ve kişiselleştirilmiş kılan dinamik zaman bağlamı
- Günün saatine, mevsimine ve kullanıcının yerel bağlamına göre değişen kişilik
- Kullanıcının dilini ve tercihlerini anlayan ve uyarlayan akıllı bir sistem
- Semantik bellek ile ilgili konuşmaları hatırlama ve bağlamsal cevaplar üretme yeteneği

## 🚀 Özellikler

### 8. 🤖 Öz-farkındalık ve Sistem İzleme
• Kendi kod yapısını ve modül ilişkilerini analiz etme
• Çalışma zamanı performans metriklini izleme (CPU, bellek, disk kullanımı)
• Kod tabanının boyutunu ve karmaşıklığını raporlama
• Otomatik bellek yönetimi ve optimizasyon algoritmaları
• Hata ayıklama ve tanılama araçları

### 1. 💬 Gelişmiş Konuşma Yeteneği
- Gemini AI ile dinamik ve bağlamsal yanıtlar
- Kullanıcı tercihlerini öğrenme ve hatırlama
- Çoklu dil desteği (Türkçe, İngilizce ve diğer diller)
- Doğal dil işleme ile otomatik dil tespiti
- Otomatik emoji ekleme ve yanıt zenginleştirme
- Kullanıcı dilini ve tercihlerini otomatik algılama
- Zaman ve bağlam duyarlı kişilik

### 2. 🕒 Zamansal Kişilik Uyarlaması
- Günün saatine göre dinamik kişilik profili
- Mevsim, hafta günü ve günün periyoduna göre yanıt uyarlama
- Kullanıcının yerel saat dilimini ve zamanını otomatik algılama
- Günün saatine, mevsimine ve özel günlere göre kişilik değişimi
- Hafta içi/hafta sonu ve tatil günlerinde farklı davranış modları
- TimezoneFinder ve Geopy ile hassas zaman dilimi tespiti

### 3. 🔍 Derin Web Arama
- `/derinarama` komutu ile gelişmiş web araması
- DuckDuckGo arama motoru entegrasyonu
- Fallback olarak Google arama desteği
- İteratif ve derinlemesine arama yeteneği
- Arama sonuçlarını akıllıca analiz etme
- Çoklu kaynaklardan bilgi toplama ve özetleme

### 4. 🖼️ Gelişmiş Multimedya İşleme
- Google Cloud Vision API ile gerçek zamanlı görüntü analizi
- EXIF meta veri çözümleme ve coğrafi konum tespiti
- PIL (Pillow) tabanlı görüntü ön işleme
- Base64 kodlama/dekodlama ile verimli görsel iletim
- Çoklu format desteği (JPEG, PNG, WEBP, GIF)
- Görsel bağlamına göre dinamik yanıt üretme
- NSFW içerik filtresi ve otomatik moderasyon

### 5. 🧠 Gelişmiş Semantik Bellek Sistemi
- SentenceTransformer (all-MiniLM-L6-v2) ile 384 boyutlu vektör temsili
- TF-IDF ve kosinüs benzerliği ile çift katmanlı bellek tarama
- Otomatik konu kümeleme ve zaman damgalı indeksleme
- Ebbinghaus unutma eğrisi entegrasyonlu bellek optimizasyonu
- GPU hızlandırmalı (CUDA) gömme işlemleri
- JSON tabanlı kalıcı bellek depolama ve otomatik yedekleme
- Bağlamsal öncelik skorlamalı bellek geri çağırma
- Çoklu dil desteği ile semantik indeksleme

### 6. 📋 Akıllı Bellek Yönetimi
- Token limitini aşmadan maksimum bağlam koruma
- Semantik olarak önemli mesajları koruyarak bellek optimizasyonu
- Otomatik bağlam budama algoritması
- En alakalı konuşma parçalarını akıllıca seçme
- Bellek sınırlamalarını aşarken bile tutarlı cevaplar sağlama
- Kullanıcı tercihlerini ve geçmiş etkileşimlerini akıllıca saklama
- Her kullanıcı için ayrı JSON hafıza dosyaları
- Güvenli ve şifrelenmiş kullanıcı verileri

### 7. 🌐 Akıllı Web Arama
- Gemini AI ile dinamik web arama
- Kullanıcı sorgularını akıllıca yorumlama
- Akıllı web arama sonuçlarını analiz etme ve özetleme
- Çoklu kaynaklardan bilgi toplama
- Arama sonuçlarını kullanıcı diline çevirme
- Güvenilir ve güncel bilgi sağlama
- Web arama gereksinimini otomatik değerlendirme

## 🛠️ Gereksinimler

### Yazılım Gereksinimleri
- Python 3.8+
- pip paket yöneticisi
- 4GB+ RAM
- 200MB disk alanı
- İnternet bağlantısı (API erişimi için)

### Gerekli Kütüphaneler
- python-telegram-bot>=20.0
- google-generativeai>=0.3.0
- python-dotenv>=0.19.0
- duckduckgo-search>=3.0.0
- requests>=2.27.0
- beautifulsoup4>=4.10.0
- emoji>=2.0.0
- langdetect>=1.0.9
- pillow>=9.0.0
- google-cloud-vision
- protobuf
- pytz>=2022.1
- geopy>=2.2.0
- timezonefinder>=6.0.0
- pytz-deprecation-shim
- tzlocal
- pydantic
- numpy>=1.20.0
- scikit-learn>=1.0.0
- sentence-transformers>=2.2.2
- tf-keras
- torch>=1.10.0

## 🧠 Mimari Yapı
Ana bileşenler:
- `bot.py`: Telegram entegrasyonu ve ana iletişim katmanı
- `memory_manager.py`: Semantik bellek yönetimi ve vektör tabanlı hatırlama sistemi
- `self_awareness.py`: Sistem sağlığı izleme ve performans optimizasyonu
- `free_will_integration.py`: Dinamik karar alma ve davranış modülasyonu
- `system_monitor.py`: Gerçek zamanlı kaynak izleme (CPU, RAM, Disk)
- `environment_checker.py`: Bağımlılık ve sistem uyumluluk kontrolü
- `free_will.py`: Otonom karar mekanizmaları ve kişilik profili yönetimi

## 🔧 Kurulum

### 1. Depoyu Klonlama
```bash
git clone https://github.com/stixyie/Nyxie-Protogen-Chatbot-Telegram-v10-main.git
cd Nyxie-Protogen-Chatbot-Telegram-v10-main
```

### 2. Sanal Ortam Oluşturma
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Bağımlılıkları Yükleme
```bash
pip install -r requirements.txt
```

## 🔐 Konfigürasyon

### Gerekli API Anahtarları
`.env` dosyasında aşağıdaki API anahtarlarını yapılandırın:
- `TELEGRAM_TOKEN`: Telegram Bot Token you need to get this token from here: https://t.me/BotFather
- `GEMINI_API_KEY`: Google Ai Studio API Key you need to get this key from here: https://aistudio.google.com/apikey

### Örnek `.env` Dosyası
```env
TELEGRAM_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
```

## 🚀 Kullanım

### Bot'u Başlatma
```bash
# Ana başlatma seçenekleri
python run_bot.py
# Alternatif başlatıcı
python start_bot.py
```

### Telegram'da Kullanım
1. Bot'a `/start` komutu ile başlayın
2. Mesaj, görüntü veya video gönderin
3. Sohbet için bot ile etkileşime geçin

## 🛡️ Güvenlik

- Kullanıcı verileri şifrelenmiş JSON dosyalarında saklanır
- Maksimum token sınırlaması ile bellek yönetimi
- Hassas bilgilerin loglanmaması

## 🤝 Destek

### Sorun Bildirim
- GitHub Issues: [Proje Sayfası](https://github.com/stixyie/Nyxie-Protogen-Chatbot-Telegram-v10-main/issues)

### Katkıda Bulunma
1. Projeyi forklayın
2. Yeni bir branch oluşturun
3. Değişikliklerinizi yapın
4. Pull Request açın

## 📄 Lisans

Bu proje GPL-3.0 Lisansı altında yayınlanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🌟 Teşekkür

- **Stixyie**: Proje yaratıcısı ve baş geliştirici
- **Google**: Gemini ve Cloud Vision API'ları

---

**Not**: Nyxie, sürekli gelişen bir AI projesidir. Geri bildirimleriniz ve katkılarınız çok değerlidir! 🚀
