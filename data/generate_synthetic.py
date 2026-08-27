"""
KKB Hackathon simülasyonu için sentetik veri üretici.

Üretilenler:
  - customers.csv        : bireysel müşteriler (kredi başvurusu sahipleri)
  - companies.csv         : firmalar (ticaret sicili tarzı bilgiler)
  - loans.csv              : krediler (bireysel + firma), customers/companies'e bağlı
  - payments.csv           : her kredi için aylık ödeme planı / gecikme geçmişi
  - company_financials.csv : firmaların yıllık mali özet verileri
  - sanctions.csv          : bazı firmalar için "ihale yasağı" tarzı resmi kayıtlar
  - news_events.csv        : firmalarla ilgili sentetik haber/duyuru metinleri (RAG için)
  - eval_questions.json    : veriden hesaplanmış "ground truth" cevaplı soru seti (kendi judge/self-check katmanımız için)

Not: Tamamı sentetiktir, gerçek kişi/firma verisi içermez. Gerçekçilik için
UCI German Credit Data (data/real/german.data) dağılımlarından esinlenilmiştir.
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("tr_TR")
Faker.seed(SEED)

OUT = Path(__file__).parent / "synthetic"
OUT.mkdir(exist_ok=True)

N_CUSTOMERS = 4000
N_COMPANIES = 1200
SECTORS = [
    "Perakende Ticaret", "İnşaat", "Tekstil", "Gıda ve İçecek", "Lojistik",
    "Bilişim ve Yazılım", "Otomotiv", "Enerji", "Turizm ve Konaklama",
    "Sağlık", "Tarım", "Metal ve Makine", "Kimya", "Eğitim", "Finans ve Sigorta",
]
CITIES = [
    "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya",
    "Gaziantep", "Kayseri", "Mersin", "Eskişehir", "Samsun", "Denizli", "Trabzon",
]
LOAN_TYPES = ["İhtiyaç Kredisi", "Taşıt Kredisi", "Konut Kredisi", "KOBİ Kredisi", "İşletme Kredisi", "Kredi Kartı"]
EDU_LEVELS = ["İlkokul", "Lise", "Ön Lisans", "Lisans", "Yüksek Lisans", "Doktora"]
EMPLOYMENT = ["Ücretli Çalışan", "Serbest Meslek", "İşveren", "Emekli", "İşsiz"]


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 0)))


# ---------------------------------------------------------------------------
# 1) CUSTOMERS (bireysel)
# ---------------------------------------------------------------------------
customers = []
for i in range(1, N_CUSTOMERS + 1):
    age = int(np.clip(np.random.normal(40, 12), 18, 80))
    income = max(8000, np.random.lognormal(mean=9.9, sigma=0.45))  # TRY/ay
    risk_base = 0.0
    risk_base += (income < 15000) * 0.15
    risk_base += (age < 25) * 0.1
    employment = random.choices(EMPLOYMENT, weights=[55, 20, 10, 10, 5])[0]
    risk_base += (employment == "İşsiz") * 0.35
    risk_base += (employment == "Emekli") * 0.05
    customers.append({
        "customer_id": f"C{i:05d}",
        "ad_soyad": fake.name(),
        "yas": age,
        "cinsiyet": random.choice(["Kadın", "Erkek"]),
        "sehir": random.choices(CITIES, weights=[30,15,12,8,6,5,4,4,4,3,3,3,2,1])[0],
        "egitim": random.choices(EDU_LEVELS, weights=[5,30,15,35,12,3])[0],
        "meslek_durumu": employment,
        "aylik_gelir_tl": round(income, 2),
        "medeni_durum": random.choice(["Bekar", "Evli", "Boşanmış", "Dul"]),
        "musteri_kkb_skoru": int(np.clip(np.random.normal(1500, 250) - risk_base * 800, 300, 1900)),
        "kayit_tarihi": random_date(date(2015, 1, 1), date(2025, 6, 1)).isoformat(),
        "_risk_base": risk_base,
    })
customers_df = pd.DataFrame(customers)

# ---------------------------------------------------------------------------
# 2) COMPANIES (firma)
# ---------------------------------------------------------------------------
def mersis_no():
    return "".join(random.choices("0123456789", k=16))

# Faker'ın tr_TR company() sağlayıcısı gerçek marka isimlerini (Arçelik, Migros vb.)
# temel alıyor. Sentetik veride (özellikle olumsuz haber/yasaklama kayıtlarında)
# gerçek bir firma adının geçmesini önlemek için tamamen kurgusal isimler üretiyoruz.
COMPANY_WORDS_1 = [
    "Yıldız", "Toprak", "Anka", "Deniz", "Bereket", "Doruk", "Umut", "Zirve",
    "Pusula", "Kıvılcım", "Bumerang", "Nokta", "Yörünge", "Mavera", "Sahil",
    "Kaptan", "Meridyen", "Rüzgar", "Vadi", "Kervan", "Bahar", "Çınar", "Damla",
    "Ege", "Toros", "Aydın", "Berrak", "Cevher", "Doğuş", "Erguvan",
]
COMPANY_WORDS_2 = [
    "Endüstri", "Teknoloji", "Yapı", "Tekstil", "Gıda", "Lojistik", "Enerji",
    "Makine", "Kimya", "Otomotiv", "Yazılım", "Turizm", "Tarım", "Metal",
    "İnşaat", "Danışmanlık", "Ticaret", "Grup", "Holding", "Sistem",
]

def fake_company_name() -> str:
    return f"{random.choice(COMPANY_WORDS_1)} {random.choice(COMPANY_WORDS_2)}"

companies = []
for i in range(1, N_COMPANIES + 1):
    kurulus = random_date(date(1995, 1, 1), date(2024, 1, 1))
    sektor = random.choice(SECTORS)
    sermaye = round(np.random.lognormal(mean=12.5, sigma=1.1), 2)
    calisan = max(1, int(np.random.lognormal(mean=2.6, sigma=1.0)))
    yas_yil = (date(2026, 8, 27) - kurulus).days / 365.25
    risk_base = 0.0
    risk_base += (yas_yil < 2) * 0.2
    risk_base += (calisan < 5) * 0.1
    risk_base += (sektor in ["İnşaat", "Turizm ve Konaklama"]) * 0.08
    companies.append({
        "company_id": f"F{i:05d}",
        "firma_unvani": fake_company_name() + " " + random.choice(["A.Ş.", "Ltd. Şti."]),
        "mersis_no": mersis_no(),
        "sektor": sektor,
        "sehir": random.choices(CITIES, weights=[28,14,12,8,6,5,4,4,4,4,3,3,3,2])[0],
        "kurulus_tarihi": kurulus.isoformat(),
        "sermaye_tl": sermaye,
        "calisan_sayisi": calisan,
        "kkb_firma_skoru": int(np.clip(np.random.normal(1400, 300) - risk_base * 900, 200, 1900)),
        "_risk_base": risk_base,
    })
companies_df = pd.DataFrame(companies)

# ---------------------------------------------------------------------------
# 3) LOANS (bireysel + firma karışık)
# ---------------------------------------------------------------------------
loans = []
loan_id_counter = 1
today = date(2026, 8, 27)

def make_loan(owner_id, owner_type, risk_base, is_company):
    global loan_id_counter
    loan_type = random.choice(LOAN_TYPES[3:] if is_company else LOAN_TYPES[:5] + ["Kredi Kartı"])
    amount = round(np.random.lognormal(mean=11.5 if is_company else 10.2, sigma=1.0), 2)
    term = random.choice([6, 12, 24, 36, 48, 60, 84, 120])
    start = random_date(date(2019, 1, 1), today - timedelta(days=30))
    rate = round(np.random.uniform(2.5, 4.9), 2)  # aylık %
    status = "Kapalı" if (today - start).days / 30 > term and random.random() < 0.85 else "Aktif"
    default_prob = np.clip(0.03 + risk_base * 0.5, 0.01, 0.6)
    is_default = random.random() < default_prob
    loans.append({
        "loan_id": f"L{loan_id_counter:06d}",
        "owner_id": owner_id,
        "owner_type": owner_type,
        "kredi_turu": loan_type,
        "tutar_tl": amount,
        "vade_ay": term,
        "aylik_faiz_yuzde": rate,
        "baslangic_tarihi": start.isoformat(),
        "durum": status,
        "temerrut_flag": is_default,
        "_default_prob": default_prob,
    })
    loan_id_counter += 1

for _, c in customers_df.iterrows():
    for _ in range(np.random.choice([0, 1, 1, 2, 2, 3], p=[0.1, 0.3, 0.25, 0.2, 0.1, 0.05])):
        make_loan(c["customer_id"], "customer", c["_risk_base"], is_company=False)

for _, f in companies_df.iterrows():
    for _ in range(np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.35, 0.25, 0.15, 0.1])):
        make_loan(f["company_id"], "company", f["_risk_base"], is_company=True)

loans_df = pd.DataFrame(loans)

# ---------------------------------------------------------------------------
# 4) PAYMENTS (ödeme planı / gecikme geçmişi) - her kredi için aylık kayıt
# ---------------------------------------------------------------------------
payments = []
pay_id = 1
for _, ln in loans_df.iterrows():
    start = date.fromisoformat(ln["baslangic_tarihi"])
    installment = round(ln["tutar_tl"] * (ln["aylik_faiz_yuzde"] / 100 + 1 / ln["vade_ay"]), 2)
    months_elapsed = min(ln["vade_ay"], max(1, int((today - start).days / 30)))
    for m in range(1, months_elapsed + 1):
        due = start + timedelta(days=30 * m)
        if due > today:
            break
        is_late = random.random() < ln["_default_prob"] * 0.8
        days_late = int(np.random.exponential(15)) if is_late else 0
        missed = ln["temerrut_flag"] and m >= months_elapsed - 1 and random.random() < 0.6
        payments.append({
            "payment_id": f"P{pay_id:07d}",
            "loan_id": ln["loan_id"],
            "taksit_no": m,
            "vade_tarihi": due.isoformat(),
            "tutar_tl": installment,
            "odeme_durumu": "Ödenmedi" if missed else ("Gecikmeli Ödendi" if is_late else "Zamanında Ödendi"),
            "gecikme_gun": 0 if missed else days_late,
        })
        pay_id += 1
payments_df = pd.DataFrame(payments)

# ---------------------------------------------------------------------------
# 5) COMPANY FINANCIALS (yıllık mali özet, 2021-2025)
# ---------------------------------------------------------------------------
financials = []
for _, f in companies_df.iterrows():
    base_revenue = f["sermaye_tl"] * np.random.uniform(0.8, 3.5)
    trend = np.random.uniform(-0.08, 0.18)  # yıllık büyüme eğilimi
    for year in range(2021, 2026):
        yoy_noise = np.random.uniform(-0.15, 0.15)
        revenue = base_revenue * ((1 + trend + yoy_noise) ** (year - 2021))
        expense_ratio = np.random.uniform(0.7, 0.98) + f["_risk_base"] * 0.1
        net_profit = revenue * (1 - expense_ratio)
        debt = revenue * np.random.uniform(0.1, 0.9) * (1 + f["_risk_base"])
        financials.append({
            "company_id": f["company_id"],
            "yil": year,
            "ciro_tl": round(revenue, 2),
            "net_kar_tl": round(net_profit, 2),
            "toplam_borc_tl": round(debt, 2),
            "ozkaynak_tl": round(f["sermaye_tl"] + net_profit * np.random.uniform(0.3, 1.0), 2),
        })
financials_df = pd.DataFrame(financials)

# ---------------------------------------------------------------------------
# 6) SANCTIONS (ihale yasağı tarzı, düşük olasılıkla bazı firmalarda)
# ---------------------------------------------------------------------------
sanction_reasons = [
    "4735 Sayılı Kanun'un 25. maddesi uyarınca ihalelere fesat karıştırma",
    "Sözleşme şartlarına aykırı davranış nedeniyle yasaklama",
    "Sahte belge düzenleme nedeniyle ihalelerden yasaklama",
    "Taahhüdünü yerine getirmeme nedeniyle geçici yasaklama",
]
sanctions = []
sanc_id = 1
sanctioned_companies = companies_df.sample(frac=0.06, random_state=SEED)
for _, f in sanctioned_companies.iterrows():
    karar_tarihi = random_date(date(2022, 1, 1), date(2026, 6, 1))
    sure_yil = random.choice([1, 1, 2, 3])
    bitis = date(karar_tarihi.year + sure_yil, karar_tarihi.month, karar_tarihi.day)
    sanctions.append({
        "sanction_id": f"S{sanc_id:04d}",
        "company_id": f["company_id"],
        "karar_tarihi": karar_tarihi.isoformat(),
        "yasak_suresi_yil": sure_yil,
        "yasak_bitis_tarihi": bitis.isoformat(),
        "gerekce": random.choice(sanction_reasons),
        "aktif_mi": bitis > today,
        "resmi_gazete_sayi": random.randint(31900, 32900),
    })
    sanc_id += 1
sanctions_df = pd.DataFrame(sanctions)

# ---------------------------------------------------------------------------
# 7) NEWS EVENTS (sentetik haber metinleri - RAG demo için)
# ---------------------------------------------------------------------------
NEWS_TEMPLATES_POS = [
    "{firma}, {yil} yılında {sektor} sektöründe {tutar} milyon TL'lik yeni yatırım kararı aldığını açıkladı.",
    "{firma}, {sehir}'deki üretim kapasitesini artırmak için genişleme planlarını duyurdu.",
    "{firma} yönetimi, {yil} yılı ilk yarısında cironun beklentilerin üzerinde arttığını bildirdi.",
    "{firma}, yeni ihracat pazarlarına açılarak uluslararası büyüme hedeflerini yükseltti.",
]
NEWS_TEMPLATES_NEG = [
    "{firma} hakkında {sehir} Ticaret Mahkemesi'nde alacaklılarla ilgili dava açıldığı öğrenildi.",
    "{firma}, {yil} yılında nakit akışı sorunları nedeniyle bazı ödemelerini erteledi.",
    "{firma} yönetimi, sektördeki daralma nedeniyle küçülme kararı aldığını duyurdu.",
    "{firma} hakkında konkordato talebiyle ilgili haberler gündeme geldi.",
]
NEWS_TEMPLATES_NEU = [
    "{firma}, {yil} yılı genel kurul toplantısını {sehir}'de gerçekleştirdi.",
    "{firma}, yönetim kurulu üyeliklerinde değişikliğe gitti.",
    "{firma}, sektör raporlarında {sektor} alanındaki oyunculardan biri olarak yer aldı.",
]
news = []
news_id = 1
sample_companies_for_news = companies_df.sample(n=min(1500, len(companies_df)), random_state=SEED)
for _, f in sample_companies_for_news.iterrows():
    n_news = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
    for _ in range(n_news):
        sentiment = random.choices(["olumlu", "olumsuz", "notr"], weights=[0.45, 0.25, 0.3])[0]
        template = random.choice({"olumlu": NEWS_TEMPLATES_POS, "olumsuz": NEWS_TEMPLATES_NEG, "notr": NEWS_TEMPLATES_NEU}[sentiment])
        yil = random.randint(2022, 2026)
        text = template.format(
            firma=f["firma_unvani"], yil=yil, sektor=f["sektor"], sehir=f["sehir"],
            tutar=random.randint(5, 500),
        )
        news.append({
            "news_id": f"N{news_id:05d}",
            "company_id": f["company_id"],
            "tarih": random_date(date(yil, 1, 1), date(min(yil, 2026), 12, 31) if yil < 2026 else today).isoformat(),
            "kaynak": random.choice(["Dünya Gazetesi", "Ekonomim", "Sözcü", "Hürriyet Ekonomi", "Bloomberg HT"]),
            "baslik": text,
            "sentiment": sentiment,
        })
        news_id += 1
news_df = pd.DataFrame(news)

# ---------------------------------------------------------------------------
# Temizlik: iç kullanım kolonlarını (_risk_base, _default_prob) ayrı tutup
# müşteriye görünecek "temiz" versiyonları da diskte tutalım (ikisi de faydalı).
# ---------------------------------------------------------------------------
customers_clean = customers_df.drop(columns=["_risk_base"])
companies_clean = companies_df.drop(columns=["_risk_base"])
loans_clean = loans_df.drop(columns=["_default_prob"])

customers_clean.to_csv(OUT / "customers.csv", index=False)
companies_clean.to_csv(OUT / "companies.csv", index=False)
loans_clean.to_csv(OUT / "loans.csv", index=False)
payments_df.to_csv(OUT / "payments.csv", index=False)
financials_df.to_csv(OUT / "company_financials.csv", index=False)
sanctions_df.to_csv(OUT / "sanctions.csv", index=False)
news_df.to_csv(OUT / "news_events.csv", index=False)

# ---------------------------------------------------------------------------
# 8) EVAL QUESTIONS - veriden hesaplanmış ground-truth cevaplı soru seti
#    (kendi "judge/self-check" katmanımızı test etmek için)
# ---------------------------------------------------------------------------
questions = []

# Basit toplam/oran soruları
total_customers = len(customers_clean)
total_companies = len(companies_clean)
active_loans = int((loans_clean["durum"] == "Aktif").sum())
default_rate_customers = round(
    loans_clean[loans_clean.owner_type == "customer"]["temerrut_flag"].mean() * 100, 2
)
top_sector = companies_clean["sektor"].value_counts().idxmax()
active_sanctions = int(sanctions_df["aktif_mi"].sum())
highest_default_sector = (
    loans_df[loans_df.owner_type == "company"]
    .merge(companies_df[["company_id", "sektor"]], left_on="owner_id", right_on="company_id")
    .groupby("sektor")["temerrut_flag"].mean().idxmax()
)

questions = [
    {
        "id": "Q1",
        "soru": "Veri setinde toplam kaç bireysel müşteri kaydı bulunuyor?",
        "beklenen_cevap": total_customers,
        "tip": "sayisal",
        "kaynak_tablo": "customers.csv",
    },
    {
        "id": "Q2",
        "soru": "Veri setinde toplam kaç firma kaydı bulunuyor?",
        "beklenen_cevap": total_companies,
        "tip": "sayisal",
        "kaynak_tablo": "companies.csv",
    },
    {
        "id": "Q3",
        "soru": "Şu anda kaç kredi 'Aktif' durumda?",
        "beklenen_cevap": active_loans,
        "tip": "sayisal",
        "kaynak_tablo": "loans.csv",
    },
    {
        "id": "Q4",
        "soru": "Bireysel müşterilerin kredilerinde temerrüt oranı yüzde kaçtır (temerrut_flag oranı)?",
        "beklenen_cevap": default_rate_customers,
        "tip": "sayisal_yuzde",
        "kaynak_tablo": "loans.csv",
    },
    {
        "id": "Q5",
        "soru": "Firmalar arasında en çok temsil edilen sektör hangisidir?",
        "beklenen_cevap": top_sector,
        "tip": "kategorik",
        "kaynak_tablo": "companies.csv",
    },
    {
        "id": "Q6",
        "soru": "Şu anda kaç firmanın aktif ihale yasağı bulunmaktadır?",
        "beklenen_cevap": active_sanctions,
        "tip": "sayisal",
        "kaynak_tablo": "sanctions.csv",
    },
    {
        "id": "Q7",
        "soru": "Firma kredilerinde temerrüt oranı en yüksek olan sektör hangisidir?",
        "beklenen_cevap": highest_default_sector,
        "tip": "kategorik",
        "kaynak_tablo": "loans.csv + companies.csv",
    },
]

with open(OUT / "eval_questions.json", "w", encoding="utf-8") as fh:
    json.dump(questions, fh, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Özet
# ---------------------------------------------------------------------------
print("Üretim tamamlandı ->", OUT)
for name, df in [
    ("customers", customers_clean), ("companies", companies_clean),
    ("loans", loans_clean), ("payments", payments_df),
    ("company_financials", financials_df), ("sanctions", sanctions_df),
    ("news_events", news_df),
]:
    print(f"  {name:20s} {len(df):6d} satır")
print(f"  {'eval_questions':20s} {len(questions):6d} soru")
