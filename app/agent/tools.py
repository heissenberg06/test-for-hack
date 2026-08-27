"""
Agent'ın kullanabileceği tool'lar (LLM'in çağırabileceği fonksiyonlar).

Önemli tasarım kararı: Her tool @tool ile işaretlenen sıradan bir Python
fonksiyonu. LangChain, fonksiyonun:
  - adını         -> tool'un adı olarak
  - docstring'ini  -> LLM'e "bu ne işe yarar" açıklaması olarak
  - tip ipuçlarını -> LLM'in doldurması gereken parametre şeması olarak
kullanır. Yani docstring burada süs değil, LLM'in okuyacağı gerçek bir talimat.

Tool'lar LLM'e METİN döndürmeli (Python objesi değil) çünkü LLM'in gördüğü
her şey sonuçta metin. Bu yüzden sözlük/liste sonuçlarını json.dumps ile
string'e çeviriyoruz.
"""
import ast
import json
import operator
from functools import lru_cache

from langchain_core.tools import tool

from app.data_engine.duckdb_engine import get_connection, get_schema_description, run_query


@lru_cache
def _get_cached_connection():
    """
    DuckDB bağlantısını (ve CSV yüklemesini) bir kere yapıp önbelleğe alır.
    Neden: get_connection() her çağrıldığında 7 CSV'yi yeniden diskten okuyup
    yeniden tablo oluşturur — bu hem yavaş hem gereksiz. Agent bir konuşma
    boyunca onlarca kez sql_query çağırabilir; hepsi aynı, tek seferde
    kurulmuş bağlantıyı paylaşmalı.
    """
    return get_connection()


def get_schema_for_prompt() -> str:
    """
    Agent'ın sistem promptuna gömülecek şema açıklamasını döndürür.
    Bunu bilinçli olarak bir TOOL yapmadık — çünkü şema sabit ve küçük;
    agent'ın her konuşmada "önce şemaya bakayım" diye bir tool-call
    harcaması yerine, bunu baştan sistem promptuna gömüp bir adım
    tasarruf ediyoruz.
    """
    return get_schema_description(_get_cached_connection())


@tool
def sql_query(query: str) -> str:
    """
    Kredi/firma veritabanında SQL (DuckDB lehçesi) sorgusu çalıştırır ve sonucu döner.

    Kullanılabilir tablolar: customers, companies, loans, payments,
    company_financials, sanctions, news_events. Tabloların kolonları ve
    aralarındaki ilişkiler sistem promptunda verilmiştir.

    Sadece SELECT sorguları kullan. Sonuç en fazla 200 satırla sınırlıdır.
    Sorgu hatalıysa (yanlış kolon/tablo adı vb.) hata mesajı dönecektir;
    hatayı oku ve sorguyu düzeltip tekrar dene.
    """
    con = _get_cached_connection()
    result = run_query(con, query)
    return json.dumps(result, ensure_ascii=False, default=str)


# --- Güvenli hesap makinesi -------------------------------------------------
# NEDEN eval() KULLANMIYORUZ: eval("__import__('os').system('rm -rf /')") gibi
# bir string, Python'ın eval()'ına verilirse gerçekten çalışır. LLM çıktısı
# bizim doğrudan kontrolümüzde olmayan bir girdi olduğundan (prompt injection
# riski dahil), asla eval()/exec() ile çalıştırılmamalı. Bunun yerine ifadeyi
# ast (Abstract Syntax Tree) ile ayrıştırıp SADECE aritmetik işlemlere izin
# veriyoruz; başka her şey (fonksiyon çağrısı, değişken, import) reddedilir.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"İzin verilmeyen ifade: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """
    Basit aritmetik ifadeleri güvenli şekilde hesaplar (örn. "1250000 * 0.03 / 12").

    Sadece +, -, *, /, **, % operatörleri ve sayılar desteklenir. SQL sorgusunun
    dönmediği ek hesaplamalar (oran, yüzde, taksit hesabı vb.) için kullan.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return json.dumps({"success": True, "result": result}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


TOOLS = [sql_query, calculator]
