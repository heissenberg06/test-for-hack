"""
Uygulama genelindeki ayarları .env dosyasından okuyan tek merkez.

Neden ayrı bir dosya?
  - Her modülün kendi başına os.getenv() çağırması yerine, tüm ayarlar tek yerden
    okunur ve tip kontrolü yapılır (örn. yanlışlıkla PORT="abc" yazılırsa burada patlar,
    uygulamanın ortasında değil).
  - Kloudeks/MIA'ya geçiş günü sadece .env değişecek, bu dosyaya dokunulmayacak.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Hangi LLM sağlayıcısı kullanılacak: "openai" veya "kloudeks"
    llm_provider: str = "openai"

    # OpenAI ayarları
    openai_api_key: str = ""
    openai_model: str = "gpt-5-nano"

    # Kloudeks/MIA ayarları (yarışma günü doldurulacak)
    kloudeks_api_key: str = ""
    kloudeks_base_url: str = "https://mia.csp.kloudeks.com/v1"
    kloudeks_model: str = "gpt-oss-120b"

    # Veri / depolama yolları
    data_dir: str = "data/synthetic"
    chroma_dir: str = "chroma_db"


@lru_cache
def get_settings() -> Settings:
    """Ayarları bir kere okuyup önbelleğe alır (her çağrıda .env'i tekrar okumaz)."""
    return Settings()
