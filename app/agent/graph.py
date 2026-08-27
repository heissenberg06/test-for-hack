"""
Agent'ın "düşün -> tool çağır -> sonucu değerlendir -> gerekirse tekrarla -> bitir"
döngüsünü LangGraph ile kuran modül.

Önceki elle yazdığımız (sabit, döngüsüz) versiyondan farkı:
  - Model istediği kadar (0, 1, 5...) tool çağırabilir; graf bunu otomatik yönetir.
  - Hangi tool'un çağrılacağına dair eşleştirmeyi biz elle yazmıyoruz, LangGraph'ın
    hazır ToolNode'u, tool_call'daki "name" alanına bakıp doğru fonksiyonu buluyor.
  - "messages" state'i (konuşma geçmişi) otomatik biriktiriliyor, elle
    messages.append(...) yazmamıza gerek kalmıyor.

Graf şeması:

    START -> agent -> (tool çağrısı var mı?) -> tools -> agent -> ... -> END
                              |
                              +-- yoksa direkt --------------------------> END
"""
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.llm.client import get_chat_model
from app.agent.tools import TOOLS, get_schema_for_prompt

SYSTEM_PROMPT_TEMPLATE = """Sen KKB'nin kredi/firma veri analitiği asistanısın.

KURALLAR:
- Sorulara SADECE elindeki tool'ları (sql_query, calculator) kullanarak, gerçek
  veriye dayanarak cevap ver. Asla veri uydurma veya tahmin yürütme.
- Bir sayı/istatistik iddia ediyorsan, mutlaka önce sql_query ile hesapla.
- SQL hatası alırsan, hatayı oku ve sorguyu düzeltip tekrar dene.
- Cevabında hangi veriye/sorguya dayandığını kısaca belirt.

VERİTABANI ŞEMASI:
{schema}
"""


def _build_agent_node():
    """
    'agent' düğümünü üretir: mevcut mesaj geçmişini LLM'e verir, LLM'in
    cevabını (metin ya da tool_calls içeren AIMessage) state'e ekler.
    """
    llm = get_chat_model()
    llm_with_tools = llm.bind_tools(TOOLS)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(schema=get_schema_for_prompt())

    def agent_node(state: MessagesState):
        # Sistem promptunu HER seferinde mesajların başına ekliyoruz (state'e
        # kalıcı olarak eklemiyoruz) -- böylece state sadece gerçek konuşmayı
        # taşır, sistem promptu tekrar tekrar biriktirilmez.
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node


def build_graph():
    """Derlenmiş (çalıştırılabilir) LangGraph agent'ını döndürür."""
    builder = StateGraph(MessagesState)

    builder.add_node("agent", _build_agent_node())
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    # tools_condition: son mesajda tool_calls varsa "tools" düğümüne,
    # yoksa END'e gider. Bu, elle yazdığımız "for call in ai_msg.tool_calls"
    # kontrolünün yerini alıyor -- ama döngüsüz değil, tekrar tekrar çalışabiliyor.
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    # tools -> agent: araç çalıştıktan sonra kontrol tekrar LLM'e dönüyor.
    # Bu satır, elle yazdığımız kodda EKSİK olan asıl döngüyü kuruyor.
    builder.add_edge("tools", "agent")

    return builder.compile()
