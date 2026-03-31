# -*- coding: utf-8 -*-
import streamlit as st
import os
import json
from google import genai
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

if not API_KEY:
    st.error("找不到 GEMINI_API_KEY，請確認 .env 檔案設定。")
    st.stop()

client = genai.Client(api_key=API_KEY)


def model_text(prompt):
    """Call Gemini and return plain text safely across response shapes."""
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    if getattr(response, "text", None):
        return response.text

    texts = []
    candidates = getattr(response, "candidates", []) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                texts.append(part_text)
    return "\n".join(texts).strip()


def clean_json_block(raw_text):
    result = (raw_text or "").strip()
    if result.startswith("```json"):
        result = result[7:]
    elif result.startswith("```"):
        result = result[3:]
    if result.endswith("```"):
        result = result[:-3]
    return result.strip()


def should_force_error_breakdown(query):
    q_text = query or ""
    q_lower = q_text.lower()
    zh_keywords = [
        "為什麼我會錯",
        "哪裡錯",
        "為什麼會錯",
        "哪裡做錯",
        "為什麼答錯",
        "這題我為什麼會錯",
        "錯在哪",
    ]
    en_keywords = [
        "why am i wrong",
        "why was i wrong",
        "where did i go wrong",
        "what did i do wrong",
        "why is this wrong",
        "why i got this wrong",
    ]
    return any(k in q_text for k in zh_keywords) or any(k in q_lower for k in en_keywords)


def normalize_strategy_name(strategy):
    raw = (strategy or "").lower()
    if "error" in raw or "錯誤" in raw:
        return "Error breakdown"
    if "socratic" in raw or "反問" in raw:
        return "Socratic"
    if "analogy" in raw or "類比" in raw:
        return "Analogy"
    if "step" in raw or "逐步" in raw:
        return "Step-by-step"
    if "worked" in raw or "例題" in raw:
        return "Worked Example"
    if "teach-back" in raw or "teach back" in raw or "回教" in raw:
        return "Teach-Back"
    if "visual" in raw or "圖像" in raw:
        return "Visualization"
    if "table" in raw or "表格" in raw:
        return "Structured Table"
    if "summary" in raw or "總結" in raw:
        return "Progressive Summary"
    if "prior" in raw or "先備" in raw:
        return "Prior Knowledge Linking"
    return "Step-by-step"


def normalize_strategy_list(strategies):
    if isinstance(strategies, str):
        strategies = [strategies]

    normalized = []
    for strategy in strategies or []:
        name = normalize_strategy_name(strategy)
        if name not in normalized:
            normalized.append(name)

    if not normalized:
        normalized = ["Step-by-step", "Progressive Summary"]

    if len(normalized) == 1:
        if normalized[0] != "Progressive Summary":
            normalized.append("Progressive Summary")
        else:
            normalized.append("Step-by-step")

    return normalized[:3]


def normalize_strategy(strategy):
    # Backward-compatible helper for old single-strategy code paths.
    return normalize_strategy_name(strategy)

def assess_state(query, history):
    prompt_assess = f"""
    你是一個教育評估系統。根據以下對話歷史與使用者的最新問題，判斷使用者的理解程度以及最適合的解釋策略。
    
    對話歷史:
    {history}
    
    最新問題:
    "{query}"

    Strategy 選擇範例：
    Example 1:我
    User: 我完全看不懂什麼是梯度下降，可以用簡單一點的方式解釋嗎？
    Level: Beginner
    Strategy: ["Analogy", "Prior Knowledge Linking", "Progressive Summary"]

    Example 2:
    User: 梯度下降是不是就是一直往最小值走？那 learning rate 是不是越大越快？
    Level: Beginner
    Strategy: ["Error Breakdown", "Visualization", "Progressive Summary"]

    Example 3:
    User: 我大概知道梯度下降在做優化，但不太確定每一步更新的數學意義
    Level: Intermediate
    Strategy: ["Step-by-step", "Worked Example"]

    Example 4:
    User: 如果 loss function 不是 convex，梯度下降還能保證找到 global minimum 嗎？
    Level: Advanced
    Strategy: ["Socratic", "Teach-Back"]

    Example 5:
    User: 我寫了一個梯度下降，但結果會震盪，可能是哪裡有問題？
    Level: Intermediate
    Strategy: ["Error Breakdown", "Step-by-step"]

    Example 6:
    User: 可以給我一個實際例子說明梯度下降怎麼用在機器學習嗎？
    Level: Intermediate
    Strategy: ["Worked Example", "Visualization"]

    Example 7:
    User: 梯度下降跟牛頓法差在哪？我有點搞混
    Level: Intermediate
    Strategy: ["Structured Table", "Prior Knowledge Linking"]

    Example 8:
    User: 我已經會用梯度下降了，但想知道有沒有更快的優化方法
    Level: Advanced
    Strategy: ["Socratic", "Teach-Back"]


    強制規則：
    如果使用者是在詢問「為什麼我會錯」或「我哪裡錯」，優先選擇 Strategy = Error breakdown。
    
    請務必只回傳包含以下三個鍵值的合法 JSON 格式，不要包含任何 Markdown 標記 (如 ```json)：
    {{
        "level": "Beginner 或 Intermediate 或 Advanced 或 Unknown",
        "strategies": ["請從策略池挑選 2-3 個，依順序排列"],
        "reasoning": "簡短說明為什麼選擇這個策略"
    }}
    """
    try:
        result = clean_json_block(model_text(prompt_assess))
        state = json.loads(result)
    except Exception:
        state = {
            "level": "Unknown",
            "strategies": ["Step-by-step", "Progressive Summary"],
            "reasoning": "判斷失敗，使用預設策略",
        }

    candidate_strategies = state.get("strategies")
    if not candidate_strategies and state.get("strategy"):
        candidate_strategies = [state.get("strategy")]
    state["strategies"] = normalize_strategy_list(candidate_strategies)

    if should_force_error_breakdown(query):
        state["strategies"] = [
            "Error breakdown",
            *[s for s in state["strategies"] if s != "Error breakdown"],
        ][:3]
        state["reasoning"] = "偵測到使用者在詢問為什麼會錯，套用錯誤拆解策略"

    state["strategy"] = state["strategies"][0]
    return state


def generate_analogy(query, level, history):
    prompt = f"""
    你是教學助理。請用 Analogy 風格解釋問題，且一定要使用貼近日常生活的比喻。
    - 使用者程度: {level}
    - 問題: {query}
    - 對話歷史: {history}

    請輸出：
    1) 先用一個生活比喻解釋核心概念。
    2) 再把比喻對應回原本概念。
    3) 結尾補一句可執行的小建議，幫助學生自我檢查是否理解。
    """
    return model_text(prompt)


def generate_socratic(query, level, history):
    prompt = f"""
    你是蘇格拉底式教練。請用 Socratic 風格引導，不直接給完整答案。
    - 使用者程度: {level}
    - 問題: {query}
    - 對話歷史: {history}

    風格要求：
    1) 幾乎都用引導問題，一次一小步。
    2) 最多只給必要提示，不直接公布完整解法。
    3) 問題要由淺入深，避免一次丟太多。
    """
    return model_text(prompt)


def generate_step_by_step(query, level, history):
    prompt = f"""
    你是教學助理。請用 Step-by-step 風格回答。
    - 使用者程度: {level}
    - 問題: {query}
    - 對話歷史: {history}

    風格要求：
    1) 用條列分成清楚步驟。
    2) 每一步都說明為什麼這樣做。
    3) 最後補一段常見錯誤提醒。
    """
    return model_text(prompt)


def generate_error_breakdown(query, level, history):
    prompt = f"""
    你是除錯型教學助理。請用 Error breakdown 風格回答。
    - 使用者程度: {level}
    - 問題: {query}
    - 對話歷史: {history}

    風格要求：
    1) 先指出可能的錯誤假設或錯誤推理。
    2) 再說明為什麼這些假設會導致錯誤。
    3) 提供正確觀念與一個快速檢查方法。
    4) 結尾提供一句可立即採用的修正建議。
    """
    return model_text(prompt)

def generate_worked_example(query, level, history): # Target Level: Intermediate
    """When to Use:
    - User has basic understanding but lacks problem-solving structure
    - User stuck but not completely lost"""

    prompt = f"""
    你是教學助理。請用 Worked Example 風格解釋問題。
    - 使用者程度: {level}
    - 問題: {query}
    - 對話歷史: {history}

    風格要求：
    1) 提供一個類似的例子問題。
    2) 用正確的思考流程與 step-by-step 的方式完整解出這個例子，並說明為什麼每一步這樣做。
    3) 最後總結這個例子與原問題的關鍵相似點與不同點，幫助學生遷移學習。
    4) 避免過長的解釋，保持重點清晰。
    """
    return model_text(prompt)

def generate_teach_back(query, level, history): # Target Level: Intermediate to Advanced
    """When to Use:
    - User already gave partially correct answer
    - User shows some understanding"""

    prompt = f"""
    你是教學助理。請用 Teach-Back 風格解釋問題。
    - 使用者程度: {level}
    - 問題: {query}
    - 對話歷史: {history}

    風格要求：
    1) 要求學生用自己的話解釋一次核心概念，並舉一個例子。
    2) 在使用者回答後，評估他們的解釋是否正確，並提供具體回饋與修正建議。
    3) 避免連續使用 Teach-Back，建議最多使用一次，避免學生疲勞。
    4) 避免模糊的提示，如「再解釋一次」或「用自己的話說說看」，要具體要求學生說什麼內容，舉什麼例子。
    """
    return model_text(prompt)



def generate_visualization(query, level, history): # Target Level: Beginner to Intermediate
    """When to Use:
    - Concept is abstract or hard to imagine
    - User shows confusion"""
    prompt = f"""
    你是教學助理。請用 Visualization 風格解釋問題。
    - 使用者程度: {level}
    - 問題: {query}
    - 對話歷史: {history}

    風格要求：
    1) 描述一個具體的視覺場景或心象
    2) 將概念的每個部分對應到視覺元素
    3) 保持意象簡單且一致，避免過於複雜或不切實際的畫面
    """
    return model_text(prompt)



def generate_structured_table(query, level, history): # ；Target Level: Intermediate
    """When to Use:
    - Comparing multiple concepts
    - High confusion between similar ideas"""
    prompt = f"""
    你是教學助理。請用 Structured Table 風格解釋問題。
    - 使用者程度: {level}
    - 問題: {query}
    - 對話歷史: {history}

    風格要求：
    1) 將相關概念或方法以表格形式呈現，清楚列出定義、特點、優缺點等關鍵資訊。
    2) 強調概念之間的差異與相似點，幫助學生建立清晰的比較框架。
    3) 保持表格簡潔明瞭，避免過多文字，確保在聊天界面中易於閱讀。
    4) Output Format: | Concept | Definition | Key Difference | Example |
    """
    return model_text(prompt)

def generate_progressive_summary(query, level, history):
    """Use With:
    - Any strategy (especially long explanations)"""
    prompt = f"""
    你是教學助理。請用 Progressive Summary 風格解釋問題。
    - 使用者程度: {level}
    - 問題: {query}
    - 對話歷史: {history}

    風格要求：
    1) 在最後，以簡短的段落總結本次解釋的核心概念與邏輯關係，幫助學生抓住重點。
    2) 每個總結都要突出關鍵概念和邏輯關係，避免重複細節。
    3) 不要取代原本的解釋內容，而是作為補充，幫助學生整理思路。
    """
    return model_text(prompt)

def generate_Prior_Knowledge_Linking(query, level, history):
    """When to Use:
    - User has known background knowledge
    - Concept can be mapped to previously learned idea"""

    prompt = f"""
    你是教學助理。請用 Prior Knowledge Linking 風格解釋問題。
    - 使用者程度: {level}
    - 問題: {query}
    - 對話歷史: {history}

    風格要求：
    1) 識別一個學生熟悉的概念
    2) 明確地將舊概念與新概念連結
    3) 強調相似性和差異性
    4) 避免錯誤的假設，確保 Prior Knowledge 是正確且相關的
    """
    return model_text(prompt)


def generate_summarize_and_conclude(topic, state, history, evaluation=None):
    prompt = f"""
    你是教學助理。請針對本輪學習做收斂總結，代表學生已掌握核心概念。

    主題:
    {topic}

    使用者程度與策略:
    {json.dumps(state, ensure_ascii=False)}

    Answer Evaluation (如果有):
    {json.dumps(evaluation or {}, ensure_ascii=False)}

    對話歷史:
    {history}

    輸出要求：
    1) 先用 3 個重點條列總結已學會內容。
    2) 指出 1 個常見陷阱，提醒如何避免。
    3) 給出 1 個下一步練習方向。
    """
    return model_text(prompt)





def evaluate_answer(user_answer, topic, asked_question, history):
    prompt_eval = f"""
    你是 Answer Evaluation 模組。請評估學生回答是否正確，並提供原因與回饋。

    原始主題:
    {topic}

    老師先前追問:
    {asked_question}

    學生回答:
    {user_answer}

    對話歷史:
    {history}

    請只回傳合法 JSON，不要加 Markdown：
    {{
      "correctness": "correct 或 partially_correct 或 incorrect 或 unclear",
      "reason": "為什麼這樣判定",
      "feedback": "給學生的具體回饋與下一步建議",
      "confidence": "high 或 medium 或 low"
    }}
    """
    default_result = {
        "correctness": "unclear",
        "reason": "目前無法穩定判定答案正確性",
        "feedback": "請再補充你的思路或舉一個例子，讓我能更準確判斷。",
        "confidence": "low",
    }
    try:
        parsed = json.loads(clean_json_block(model_text(prompt_eval)))
        if not isinstance(parsed, dict):
            return default_result
        parsed["correctness"] = (parsed.get("correctness") or "unclear").lower()
        if parsed["correctness"] not in {"correct", "partially_correct", "incorrect", "unclear"}:
            parsed["correctness"] = "unclear"
        parsed["confidence"] = (parsed.get("confidence") or "low").lower()
        if parsed["confidence"] not in {"high", "medium", "low"}:
            parsed["confidence"] = "low"
        parsed["reason"] = parsed.get("reason") or default_result["reason"]
        parsed["feedback"] = parsed.get("feedback") or default_result["feedback"]
        return parsed
    except Exception:
        return default_result


def decide_next_action(topic, state, history, evaluation=None):
    prompt_decision = f"""
    你是 Decision Maker。請決定下一步是要繼續追問、教學，還是改成複述結論並收尾。

    主題:
    {topic}

    使用者程度與策略:
    {json.dumps(state, ensure_ascii=False)}

    Answer Evaluation (如果有):
    {json.dumps(evaluation or {}, ensure_ascii=False)}

    對話歷史:
    {history}

    決策規則：
    1) 如果答案 incorrect 或 unclear，優先 action = teach_only。
    2) 如果答案 partially_correct，可視情況 action = ask_follow_up。
    3) 如果答案 correct 且使用者表現良好，可考慮 action = summarize_and_conclude（複述結論並收尾）。

    請只回傳合法 JSON，不要加 Markdown：
    {{
      "action": "ask_follow_up 或 teach_only 或 summarize_and_conclude",
      "reason": "為什麼這樣決定",
      "follow_up_question": "如果 action 是 ask_follow_up，提供一題問題；否則可留空"
    }}
    """
    default_decision = {
        "action": "teach_only",
        "reason": "預設採用教學收斂，避免無限追問。",
        "follow_up_question": "",
    }

    try:
        result = json.loads(clean_json_block(model_text(prompt_decision)))
        if not isinstance(result, dict):
            return default_decision
        action = (result.get("action") or "teach_only").lower()
        if action not in {"ask_follow_up", "teach_only", "summarize_and_conclude"}:
            action = "teach_only"
        result["action"] = action
        result["reason"] = result.get("reason") or default_decision["reason"]
        result["follow_up_question"] = result.get("follow_up_question") or ""
        return result
    except Exception:
        return default_decision


def generate_follow_up_question(topic, state, history, evaluation=None):
    prompt_follow_up = f"""
    你是教學助理。請基於目前情況只生成一題追問，不要附答案。

    主題:
    {topic}

    使用者程度與策略:
    {json.dumps(state, ensure_ascii=False)}

    Answer Evaluation (如果有):
    {json.dumps(evaluation or {}, ensure_ascii=False)}

    對話歷史:
    {history}

    輸出要求：
    1) 只輸出一個問題句。
    2) 不要加入任何前言、解釋或多餘段落。
    """
    try:
        return model_text(prompt_follow_up).strip()
    except Exception:
        return "你可以用自己的話，再解釋一次核心概念與一個例子嗎？"


def reassess_state_after_answer(topic, asked_question, user_answer, history):
    reassess_query = (
        f"原始主題：{topic}\n"
        f"老師前一題追問：{asked_question}\n"
        f"學生本次回答：{user_answer}\n"
        "請根據這次回答重新判斷學生目前理解程度與適合策略。"
    )
    return assess_state(reassess_query, history)


def strategy_blend_instruction(strategy):
    instruction_map = {
        "Analogy": "至少加入 1 個貼近日常生活的比喻，並把比喻對應回原概念。",
        "Socratic": "在回答中保留 1-2 個引導問題，幫助學生主動思考。",
        "Step-by-step": "以清楚步驟拆解重點，每一步說明原因。",
        "Error breakdown": "指出可能錯誤假設，說明為何錯並提供修正方式。",
        "Worked Example": "加入一個簡短例題並示範正確解題思路。",
        "Teach-Back": "結尾請學生用自己的話重述關鍵概念與例子。",
        "Visualization": "描述一個容易想像的心智畫面，幫助理解抽象概念。",
        "Structured Table": "若涉及比較，使用精簡表格呈現差異與重點。",
        "Progressive Summary": "結尾提供簡短重點總結，強調概念關係。",
        "Prior Knowledge Linking": "連結學生已知概念，說明相似處與差異處。",
    }
    return instruction_map.get(strategy, "請保持解釋清楚、可執行。")


def generate_explanation(query, state, history, allow_questions=True):
    strategies = normalize_strategy_list(state.get("strategies") or state.get("strategy"))
    level = state.get("level", "Unknown")
    blend_guidance = "\n".join(
        [f"- {s}: {strategy_blend_instruction(s)}" for s in strategies]
    )
    question_policy = (
        "可以包含最多 1 個引導問題。"
        if allow_questions
        else "不要提出任何問題句，避免問號與追問。"
    )

    try:
        prompt = f"""
        你是自適應教學助理。請基於以下策略，生成「一個整合後的回答」，而不是分段重複回答。

        使用者程度: {level}
        問題: {query}
        對話歷史:
        {history}

        本輪啟用策略（請融合，不要分別回答）：
        {json.dumps(strategies, ensure_ascii=False)}

        每個策略的融合要求：
        {blend_guidance}

        輸出要求：
        1) 只輸出一個連貫答案，不要用「策略A/策略B」分段標題。
        2) 內容簡潔但完整，避免重複同義句。
        3) 若適合，最後加一段「快速檢查」幫助學生自測理解。
        4) {question_policy}
        """
        return model_text(prompt)
    except Exception:
        return "目前無法產生解釋內容，可能是模型或 API 設定暫時異常。請稍後再試。"

# --- Streamlit UI ---
st.title("Adaptive Explanation Agent")
st.markdown("根據你的理解程度，動態調整解釋方式的 AI 系統 (Powered by Gemini)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_answer_evaluation" not in st.session_state:
    st.session_state.awaiting_answer_evaluation = False
if "pending_follow_up" not in st.session_state:
    st.session_state.pending_follow_up = ""
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""
if "last_state" not in st.session_state:
    st.session_state.last_state = {
        "level": "Unknown",
        "strategy": "Step-by-step",
        "strategies": ["Step-by-step", "Progressive Summary"],
    }

# 顯示歷史訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "state" in msg:
            with st.expander("AI 狀態評估視角 (Prototype Debug)"):
                st.json(msg["state"])
        if "evaluation" in msg:
            with st.expander("Answer Evaluation (Prototype Debug)"):
                st.json(msg["evaluation"])
        if "decision" in msg:
            with st.expander("Decision Maker (Prototype Debug)"):
                st.json(msg["decision"])

# 輸入框
if prompt := st.chat_input("請輸入問題，例如：什麼是 overfitting？"):
    # 顯示使用者問題
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 準備歷史對話字串給模型參考 (取最近4次)
    history_str = ""
    for m in st.session_state.messages[-5:-1]:
        history_str += f"{m['role']}: {m['content']}\n"
        
    assistant_msg = {"role": "assistant"}
    with st.chat_message("assistant"):   # with -> 開助理窗格
        # Case A: 這輪是學生在回答先前追問 -> 先做 Answer Evaluation
        if st.session_state.awaiting_answer_evaluation:  # 判斷是否在等待答案評估
            with st.spinner("Answer Evaluation 中..."): # 評估學生回答   # with -> 顯示 spinner 窗格
                eval_result = evaluate_answer(
                    user_answer=prompt,
                    topic=st.session_state.current_topic,  # current topic 即為 prompt
                    asked_question=st.session_state.pending_follow_up,
                    history=history_str,
                )
                with st.expander("Answer Evaluation (已更新)", expanded=True):
                    st.json(eval_result)

            with st.spinner("重新評估學生狀態中..."):
                updated_state = reassess_state_after_answer(
                    topic=st.session_state.current_topic,
                    asked_question=st.session_state.pending_follow_up,
                    user_answer=prompt,
                    history=history_str,
                )
                st.session_state.last_state = updated_state
                with st.expander("AI 狀態評估視角 (Case A 重新評估)", expanded=True):
                    st.json(updated_state)

            with st.spinner("Decision Maker 判斷下一步中..."): # 決定下一步
                decision = decide_next_action(
                    topic=st.session_state.current_topic,
                    state=updated_state,
                    history=history_str,
                    evaluation=eval_result,
                )
                with st.expander("Decision Maker (已更新)", expanded=True):
                    st.json(decision)

            correctness_map = {
                "correct": "正確",
                "partially_correct": "部分正確",
                "incorrect": "不正確",
                "unclear": "不明確",
            }
            correctness_label = correctness_map.get(eval_result.get("correctness"), "不明確")
            evaluation_text = (
                "### Answer Evaluation\n"
                f"- 判定: {correctness_label}\n"
                f"- 原因: {eval_result.get('reason', '')}\n"
                f"- 回饋: {eval_result.get('feedback', '')}"
            )

            # 依照 Decision Maker 決定的 action 來回應學生
            ## 追問
            if decision.get("action") == "ask_follow_up":
                follow_up = decision.get("follow_up_question") or generate_follow_up_question(
                    topic=st.session_state.current_topic,
                    state=updated_state,
                    history=history_str,
                    evaluation=eval_result,
                )
                response_text = f"{evaluation_text}\n\n---\nFollow-up: {follow_up}"
                st.session_state.awaiting_answer_evaluation = True
                st.session_state.pending_follow_up = follow_up
            ## 作結
            elif decision.get("action") == "summarize_and_conclude":
                summary_text = generate_summarize_and_conclude(
                    topic=st.session_state.current_topic,
                    state=updated_state,
                    history=history_str,
                    evaluation=eval_result,
                )
                response_text = f"{evaluation_text}\n\n### 總結\n{summary_text}"
                st.session_state.awaiting_answer_evaluation = False
                st.session_state.pending_follow_up = ""
            ## 補強教學
            else:
                remediation_strategy = "Error breakdown"   # 學生答錯進入錯誤拆解教學
                if eval_result.get("correctness") == "correct":  # 學生答對但 Decision Maker 決定還要補強教學，則用原本評估的策略
                    remediation_strategy = normalize_strategy_name(updated_state.get("strategy"))
            
                remediation_state = {
                    "level": updated_state.get("level", "Unknown"),
                    "strategy": remediation_strategy,
                    "strategies": [remediation_strategy, "Progressive Summary"],
                }
                remediation_query = (
                    f"主題：{st.session_state.current_topic}\n"
                    f"學生回答：{prompt}\n"
                    f"評估結果：{eval_result.get('correctness')}\n"
                    f"原因：{eval_result.get('reason')}\n"
                    "請提供精準、可理解的補充教學。"
                )
                teaching_text = generate_explanation(
                    remediation_query,
                    remediation_state,
                    history_str,
                    allow_questions=False,
                )
                response_text = f"{evaluation_text}\n\n### 補充教學\n{teaching_text}"
                st.session_state.awaiting_answer_evaluation = False
                st.session_state.pending_follow_up = ""

            assistant_msg["evaluation"] = eval_result
            assistant_msg["decision"] = decision
            assistant_msg["state"] = updated_state
            assistant_msg["content"] = response_text
            st.markdown(response_text)

        # Case B: 一般新問題 -> 先評估狀態與策略，再由 Decision Maker 決定要不要追問
        else:
            with st.spinner("評估使用者狀態與選擇最佳策略中..."):
                state = assess_state(prompt, history_str)
                with st.expander("AI 狀態評估視角 (已更新)", expanded=True):
                    st.json(state)

            with st.spinner("生成客製化解釋中..."):
                explanation_text = generate_explanation(
                    prompt,
                    state,
                    history_str,
                    allow_questions=False,
                )

            with st.spinner("Decision Maker 判斷是否追問中..."):
                decision = decide_next_action(
                    topic=prompt,
                    state=state,
                    history=history_str,
                    evaluation=None,
                )
                with st.expander("Decision Maker (已更新)", expanded=True):
                    st.json(decision)

            response_text = explanation_text
            if decision.get("action") == "ask_follow_up":
                follow_up = decision.get("follow_up_question") or generate_follow_up_question(
                    topic=prompt,
                    state=state,
                    history=history_str,
                    evaluation=None,
                )
                response_text = f"{explanation_text}\n\n---\nFollow-up: {follow_up}"
                st.session_state.awaiting_answer_evaluation = True
                st.session_state.pending_follow_up = follow_up
            elif decision.get("action") == "summarize_and_conclude":
                summary_text = generate_summarize_and_conclude(
                    topic=prompt,
                    state=state,
                    history=history_str,
                    evaluation=None,
                )
                response_text = f"{explanation_text}\n\n### 總結\n{summary_text}"
                st.session_state.awaiting_answer_evaluation = False
                st.session_state.pending_follow_up = ""
            else:
                st.session_state.awaiting_answer_evaluation = False
                st.session_state.pending_follow_up = ""

            st.session_state.current_topic = prompt
            st.session_state.last_state = state

            assistant_msg["state"] = state
            assistant_msg["decision"] = decision
            assistant_msg["content"] = response_text
            st.markdown(response_text)

    # 存入對話歷史
    st.session_state.messages.append(assistant_msg)
