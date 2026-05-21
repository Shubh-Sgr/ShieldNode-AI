import os
import warnings
import gradio as gr
from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SYSTEM SETUP ---
warnings.filterwarnings("ignore")
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

# API Key initialization (Pulled safely from your Space Settings > Secrets)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- 2. STATE & PROTOCOL SCHEMAS ---
class GuardrailState(TypedDict):
    raw_content: str
    violations: List[str]
    clean_content: str
    status_log: str

# Structured validation schema for our Auditor Agent
class AuditReport(BaseModel):
    is_safe: Literal["safe", "unsafe"] = Field(
        description="Mark 'safe' if the content has NO sensitive data or toxicity, otherwise 'unsafe'."
    )
    detected_violations: List[str] = Field(
        default=[],
        description="List specific policy breaks found (e.g., 'Leaked Email', 'API Key Exposure', 'Toxic Language')."
    )

# --- 3. AGENTIC NODES ---
def auditor_node(state: GuardrailState):
    """The Auditor: Performs strict semantic scanning using structured schemas."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_auditor = llm.with_structured_output(AuditReport)
    
    audit_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an automated corporate compliance officer. Audit the input text strictly for toxicity, personally identifiable information (PII) like emails or phone numbers, and hardcoded API keys."),
        ("human", "Analyze this text: {content}")
    ])
    
    chain = audit_prompt | structured_auditor
    report = chain.invoke({"content": state["raw_content"]})
    
    return {
        "violations": report.detected_violations,
        "status_log": f"🚨 Violations caught: {', '.join(report.detected_violations)}" if report.is_safe == "unsafe" else "✅ Audit passed without violations."
    }

def safety_redaction_node(state: GuardrailState):
    """The Redactor: Programmatically masks unsafe elements while retaining semantic integrity."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    
    redact_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data sanitization engine. Rewrite the text to strip out the flagged violations. Replace sensitive data like emails or API keys with [REDACTED]. Keep the remaining text identical."),
        ("human", "Flagged Violations: {violations}\nRaw Text: {content}")
    ])
    
    chain = redact_prompt | llm
    sanitized_output = chain.invoke({"violations": state["violations"], "content": state["raw_content"]})
    
    return {
        "clean_content": sanitized_output.content,
        "status_log": "🛡️ Content sanitized and safely redacted."
    }

def approval_node(state: GuardrailState):
    """Pass-through Node: Executes if content passes evaluation frameworks directly."""
    return {
        "clean_content": state["raw_content"],
        "status_log": "🚀 Content approved with zero modifications."
    }

# --- 4. STATE ROUTING SWITCHES ---
def compliance_router(state: GuardrailState):
    """Evaluates the state matrix to switch execution paths."""
    if state["violations"]:
        return "redact"
    return "approve"

# --- 5. COMPILING THE GRAPH MATRIX ---
workflow = StateGraph(GuardrailState)

workflow.add_node("auditor", auditor_node)
workflow.add_node("redactor", safety_redaction_node)
workflow.add_node("approver", approval_node)

workflow.set_entry_point("auditor")

workflow.add_conditional_edges(
    "auditor",
    compliance_router,
    {
        "redact": "redactor",
        "approve": "approver"
    }
)

workflow.add_edge("redactor", END)
workflow.add_edge("approver", END)
guard_app = workflow.compile()

# --- 6. UI STYLE (CSS) ---
CSS = """
.gradio-container { background-color: #0d1117 !important; }
.hero-title { background: linear-gradient(90deg, #10b981, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; text-align: center; }
#console-box { font-family: monospace; color: #34d399; background: #161b22; padding: 15px; border-radius: 8px; }
"""

# --- 7. THE INTERFACE ---
with gr.Blocks(css=CSS, title="GuardRail AI") as demo:
    gr.HTML("<h1 class='hero-title' style='font-size: 2.5em; margin-top:20px;'>ShieldNode AI</h1>")
    gr.HTML("<p style='text-align:center; color:#8b949e;'>Real-time compliance validation and data sanitization guardrails.</p><br>")
    
    with gr.Row():
        with gr.Column(scale=1):
            user_text = gr.Textbox(
                label="Raw Content Input", 
                placeholder="Paste drafted content here... \n\nTry adding a fake email like test@test.com, a fake AWS key like AKIAIOSFODNN7EXAMPLE, or aggressive language to test the filter.", 
                lines=8
            )
            submit_btn = gr.Button("Shield Content 🛡️", variant="primary")
            
        with gr.Column(scale=1):
            guarded_output = gr.Textbox(label="Guarded Compliant Output", lines=8, interactive=False)
            
    with gr.Accordion("Guardrail Execution Log", open=True):
        system_logs = gr.Markdown("Guardrail fabric idle...", elem_id="console-box")

    # --- RUNTIME PIPELINE GENERATOR ---
    def process_guardrails(text):
        if not text.strip():
            return "No input provided.", "System idle."
            
        # First feedback point sent to screen immediately via yield
        yield "Processing audit layers...", "🔍 Initiating compliance scan..."
        
        initial_state = {"raw_content": text, "violations": [], "clean_content": "", "status_log": ""}
        
        try:
            # Stream graph updates live as nodes complete execution
            for update in guard_app.stream(initial_state, stream_mode="updates"):
                if "auditor" in update:
                    yield "Analyzing audit findings...", f"⚙️ {update['auditor']['status_log']}"
                if "redactor" in update:
                    yield update["redactor"]["clean_content"], f"⚙️ {update['redactor']['status_log']}"
                if "approver" in update:
                    yield update["approver"]["clean_content"], f"⚙️ {update['approver']['status_log']}"
        except Exception as e:
            yield f"Error processing content.", f"❌ Operational Fault: {str(e)}"
                
    submit_btn.click(process_guardrails, inputs=[user_text], outputs=[guarded_output, system_logs])

# --- 8. HUGGING FACE LAUNCH CONFIG ---
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860
    )
