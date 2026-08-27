"""
Provider-agnostic LLM client factory.

Amaç: Kod tabanının geri kalanı (agent, RAG, vs.) hiçbir zaman "OpenAI" veya
"Kloudeks" ismini görmesin — sadece get_chat_model() çağırsın. Hangi sağlayıcının
kullanılacağına .env'deki LLM_PROVIDER değişkeni karar verir.

Yarışma günü Kloudeks/MIA key'i geldiğinde:
  1) .env içinde LLM_PROVIDER=kloudeks yapılacak
  2) KLOUDEKS_API_KEY doldurulacak
  3) Bu dosyada TEK BİR SATIR bile değişmeyecek.

Neden bu çalışıyor: Kloudeks/MIA, OpenAI ile uyumlu bir API sunuyor
(base_url + api_key ile OpenAI SDK'sı üzerinden erişilebiliyor), bu yüzden
tek bir istemci sınıfı (ChatOpenAI) her iki sağlayıcı için de yeterli.
"""
from langchain_openai import ChatOpenAI

from app.config import get_settings


def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Ayarlara göre doğru sağlayıcıya bağlı bir ChatOpenAI örneği döndürür."""
    settings = get_settings()

    if settings.llm_provider == "kloudeks":
        if not settings.kloudeks_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=kloudeks ama KLOUDEKS_API_KEY boş. "
                ".env dosyasını kontrol et."
            )
        return ChatOpenAI(
            model=settings.kloudeks_model,
            api_key=settings.kloudeks_api_key,
            base_url=settings.kloudeks_base_url,
            temperature=temperature,
        )

    # Varsayılan: openai
    if not settings.openai_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=openai ama OPENAI_API_KEY boş. .env dosyasını kontrol et."
        )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )
