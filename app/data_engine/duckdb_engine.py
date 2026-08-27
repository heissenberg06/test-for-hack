"""
CSV veri setlerini DuckDB'ye yükleyip SQL ile sorgulanabilir hale getiren katman.

Neden bu dosya var:
  - Agent'ın "sql_query" tool'u bu modülü kullanacak.
  - Agent SQL yazabilmesi için hangi tabloların/kolonların var olduğunu bilmeli;
    bu yüzden şema bilgisini otomatik çıkarıp okunabilir bir metne çeviriyoruz
    (get_schema_description) — bu metin, agent'ın prompt'una eklenecek.
  - Yanlış/bozuk SQL (LLM'in hatası) uygulamayı çökertmemeli; run_query hatayı
    yakalayıp okunabilir bir mesaj döndürüyor, agent bunu görüp SQL'ini düzeltebilir.
"""
from pathlib import Path

import duckdb

from app.config import get_settings

# Tablo adı -> CSV dosya adı eşlemesi. Yeni bir veri kaynağı eklediğimizde
# (örn. gerçek yarışma verisi) sadece bu sözlüğe bir satır eklemek yeterli olacak.
TABLES = {
    "customers": "customers.csv",
    "companies": "companies.csv",
    "loans": "loans.csv",
    "payments": "payments.csv",
    "company_financials": "company_financials.csv",
    "sanctions": "sanctions.csv",
    "news_events": "news_events.csv",
}

# Agent'ın SQL yazarken kolonların ne anlama geldiğini bilmesi için kısa açıklamalar.
# Bu bilgi CSV'nin kendisinde yok, biz elle giriyoruz -> agent'a "iş bağlamı" veriyor.
COLUMN_NOTES = {
    "customers": "Bireysel müşteriler. customer_id -> loans.owner_id ile eşleşir (owner_type='customer' iken).",
    "companies": "Firmalar. company_id -> loans.owner_id (owner_type='company'), sanctions.company_id, news_events.company_id, company_financials.company_id ile eşleşir.",
    "loans": "Krediler. owner_type ('customer' ya da 'company') hangi tabloya bağlı olduğunu belirtir. temerrut_flag: kredi temerrüde düştü mü (True/False).",
    "payments": "Her kredi için aylık ödeme kayıtları. loan_id -> loans.loan_id. odeme_durumu: 'Zamanında Ödendi' / 'Gecikmeli Ödendi' / 'Ödenmedi'.",
    "company_financials": "Firmaların 2021-2025 yıllık mali özetleri (ciro, net kâr, borç, özkaynak).",
    "sanctions": "İhale yasağı kayıtları. Sadece bazı firmalar için mevcut (aktif_mi: yasak hâlâ geçerli mi).",
    "news_events": "Firmalarla ilgili sentetik haber başlıkları + sentiment ('olumlu'/'olumsuz'/'notr').",
}


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Bellek-içi bir DuckDB bağlantısı açar ve tüm CSV'leri gerçek tablolar olarak yükler.

    Neden bellek-içi (":memory:")? Veri setimiz küçük (~15 MB), her istekte diskten
    okumaya gerek yok; uygulama başlarken bir kere yüklenip RAM'de kalması yeterli
    ve daha hızlı. Kalıcı bir .duckdb dosyasına ihtiyaç duyarsak (örn. çok büyük veri
    gelirse) bu fonksiyonu tek satır değiştirerek diske yazan hale getirebiliriz.
    """
    settings = get_settings()
    data_dir = Path(settings.data_dir)

    con = duckdb.connect(database=":memory:")

    for table_name, csv_file in TABLES.items():
        csv_path = data_dir / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Beklenen veri dosyası bulunamadı: {csv_path}")
        # read_csv_auto: DuckDB kolon tiplerini (int, float, date, bool...) otomatik
        # algılar. CREATE TABLE AS SELECT ile CSV'nin tamamını gerçek bir tabloya
        # dönüştürüyoruz (sonraki sorgular disk I/O yapmadan RAM üzerinden çalışır).
        con.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path.as_posix()}')"
        )

    return con


def get_schema_description(con: duckdb.DuckDBPyConnection) -> str:
    """
    Agent'ın prompt'una eklenecek, tüm tabloların şemasını + iş bağlamı notlarını
    içeren okunabilir bir metin üretir. Örnek çıktı için testte örneğine bakabilirsin.
    """
    lines = []
    for table_name in TABLES:
        columns = con.execute(f"DESCRIBE {table_name}").fetchall()
        col_desc = ", ".join(f"{col[0]} ({col[1]})" for col in columns)
        lines.append(f"### Tablo: {table_name}")
        lines.append(f"Açıklama: {COLUMN_NOTES.get(table_name, '')}")
        lines.append(f"Kolonlar: {col_desc}")
        lines.append("")
    return "\n".join(lines)


def run_query(con: duckdb.DuckDBPyConnection, sql: str) -> dict:
    """
    Verilen SQL'i çalıştırır. Başarılıysa satırları + kolon adlarını, başarısızsa
    hata mesajını döndürür (exception fırlatmaz) — agent hatayı görüp SQL'ini
    düzeltip tekrar deneyebilsin diye.
    """
    try:
        result = con.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        # Çok büyük sonuçlarda agent'ın context'ini şişirmemek için üst sınır koyuyoruz.
        truncated = len(rows) > 200
        rows = rows[:200]
        return {
            "success": True,
            "columns": columns,
            "rows": [dict(zip(columns, row)) for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }
    except Exception as exc:  # DuckDB'nin fırlattığı hata tiplerini tek tek elemek yerine
        return {"success": False, "error": str(exc)}
