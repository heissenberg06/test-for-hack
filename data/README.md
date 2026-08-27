# Veri Katmanı

## data/real/ — Gerçek veri
- **german.data / german.doc**: UCI Statlog German Credit Data (1000 kayıt, 20 öznitelik).
  Gerçek, herkese açık bir kredi risk veri setidir. Kaynak: UCI Machine Learning Repository.
  Kategorik kodları (`A11`, `A34` vb.) için `german.doc` içindeki sözlüğe bakın.
  Kullanım amacı: agent'ın "bilinen/harici" bir veri setiyle de çalışabildiğini göstermek
  ve kendi sentetik verimizin dağılımlarını gerçekçilik açısından karşılaştırmak.

## data/synthetic/ — Sentetik veri (generate_synthetic.py ile üretildi)
Tamamen kurgusal, gerçek kişi/firma içermez. `python data/generate_synthetic.py` ile
deterministik (seed=42) olarak yeniden üretilebilir.

| Dosya | Satır | Açıklama |
|---|---|---|
| customers.csv | ~4.000 | Bireysel müşteriler: demografi, gelir, KKB skoru |
| companies.csv | ~1.200 | Firmalar: MERSİS no, sektör, sermaye, KKB firma skoru |
| loans.csv | ~7.200 | Krediler (bireysel + firma), tutar/vade/faiz/durum/temerrüt bayrağı |
| payments.csv | ~215.000 | Her kredi için aylık ödeme planı ve gecikme geçmişi |
| company_financials.csv | 6.000 | Firma başına 2021-2025 yıllık ciro/kâr/borç/özkaynak |
| sanctions.csv | ~70 | İhale yasağı tarzı kayıtlar (bazı firmalar için, %6'sı) |
| news_events.csv | ~2.000 | Firmalarla ilgili sentetik haber başlıkları + sentiment etiketi (RAG/metin analizi denemeleri için) |
| eval_questions.json | 7 | Veriden hesaplanmış **ground-truth cevaplı** soru seti — agent'ın ürettiği cevabı otomatik doğrulamak (self-check/judge katmanı) için başlangıç seti |

### Önemli tasarım notları
- Firma isimleri **tamamen kurgusaldır** — Faker'ın tr_TR `company()` sağlayıcısı gerçek marka
  isimleri döndürdüğü için kullanılmadı, kendi kurgusal isim üretecimiz yazıldı.
  Bkz. `COMPANY_WORDS_1` / `COMPANY_WORDS_2` listeleri.
- Risk sinyalleri (düşük gelir, genç yaş, işsizlik, yeni kurulmuş firma, riskli sektör)
  temerrüt olasılığına kasıtlı olarak enjekte edildi — böylece agent'ın "neden yüksek riskli"
  sorusuna anlamlı, tutarlı bir cevap üretip üretemediği test edilebilir.
- `eval_questions.json` yalnızca başlangıç seti; agent geliştirdikçe soru sayısını ve
  karmaşıklığını (çok adımlı, çok tablolu sorular) artıracağız.

### Yeniden üretme
```bash
source .venv/bin/activate
python data/generate_synthetic.py
```
