# FashionVN Assistant - End-to-End Production Guide
**Realistic Vietnamese Fashion E-commerce AI Chatbot (2026)**

**Project**: StyleVN - Vietnamese online fashion store customer support chatbot  
**Goal**: Automate 70-80% of customer queries (order status, sizing, returns, stock, promotions)  
**Base model**: Qwen2.5-7B-Instruct  
**Deployment**: AWS cloud-first (g6.xlarge recommended)  
**Language**: Primarily Vietnamese  
**Estimated total initial cost**: $4,500 - $9,000 USD (3-6 months)

## Table of Contents
1. Project Overview & Requirements
2. Prerequisites & Environment Setup
3. AWS Cloud Setup (g6.xlarge)
4. Data Collection & Preparation
5. Fine-Tuning with QLoRA
6. RAG Pipeline Implementation
7. Guardrails & Faithfulness Checks
8. Model Quantization
9. vLLM Deployment on AWS
10. Frontend & Integration (Streamlit + Zalo OA)
11. Evaluation & Monitoring
12. Security, Compliance & Logging
13. Cost Breakdown & Scaling Plan
14. Maintenance & Iteration Loop

## 1. Project Overview & Requirements

**Key Features**
- Answer Vietnamese customer questions naturally
- Real-time order status lookup
- Product catalog + stock + size chart retrieval
- Return/exchange policy enforcement
- Promotion & discount handling
- Refusal for out-of-scope or hallucinated answers
- Logging for continuous improvement

**Success Metrics**
- Hallucination rate < 5%
- Faithfulness > 92%
- User satisfaction > 85%
- Handle 1,000+ chats/day with < 1.5s TTFT (p95)

## 2. Prerequisites & Environment Setup

**Local machine requirements**:
- Ubuntu 22.04 / 24.04 or macOS
- Python 3.11+
- NVIDIA GPU with ≥24GB VRAM (for fine-tuning) or use cloud

**Install core tools**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes peft trl datasets
pip install axolotl[flash-attn]  # or use unsloth
pip install langchain llama-index chromadb sentence-transformers
pip install vllm openai  # openai for client compatibility
pip install streamlit fastapi uvicorn
```

## 3. AWS Cloud Setup (g6.xlarge - Recommended)

**Launch instance**:
1. Go to EC2 → Launch Instance
2. AMI: Deep Learning AMI GPU PyTorch 2.4 (Ubuntu 22.04)
3. Instance type: **g6.xlarge** (1× L4 24GB)
4. Storage: 100 GB gp3
5. Security group: Allow inbound 8000 (vLLM), 8501 (Streamlit), 22 (SSH)

**SSH & Setup**:
```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
sudo apt update && sudo apt upgrade -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Create persistent volume** (optional but recommended):
- Create EBS volume (100-200GB)
- Attach to instance
- Mount: `/mnt/data`

## 4. Data Collection & Preparation

**Required datasets** (target 800-2,500 high-quality examples):

1. **FAQs & Policies** (500 examples)
   - Return policy, size chart, shipping, promotion rules

2. **Product Catalog Q&A** (800 examples)
   - Generated from product descriptions + images (use GPT-4o/Claude for initial synthesis)

3. **Order Status Simulation** (300 examples)
   - Mock order database queries

4. **Real Chat Logs** (anonymized from Zalo/FB)

**Data format** (Alpaca / ChatML style with Qwen template):
```json
{
  "messages": [
    {"role": "system", "content": "Bạn là nhân viên hỗ trợ khách hàng của StyleVN..."},
    {"role": "user", "content": "Đơn hàng 12345 của em ship chưa ạ?"},
    {"role": "assistant", "content": "Dạ đơn hàng #12345 hiện đang ở trạng thái 'Đang chuẩn bị'. Dự kiến giao trong 2-3 ngày tới ạ."}
  ]
}
```

**Use this script to apply Qwen chat template**:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def apply_chat_template(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}
```

## 5. Fine-Tuning with QLoRA

**Recommended config (Axolotl yaml)**:
```yaml
# qwen2.5-7b-fashionvn.yml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
chat_template: qwen
datasets:
  - path: fashionvn_dataset.jsonl
    type: chatml
    split: train
val_set_size: 0.1
output_dir: ./qwen2.5-7b-fashionvn-lora

adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true

learning_rate: 2e-4
num_epochs: 1
micro_batch_size: 2
gradient_accumulation_steps: 8   # effective batch size = 16
optimizer: adamw_8bit
lr_scheduler: cosine
warmup_steps: 100
max_seq_length: 4096
```

**Run training**:
```bash
accelerate launch -m axolotl.cli.train qwen2.5-7b-fashionvn.yml
```

## 6. RAG Pipeline Implementation

**Embedding model**: BGE-M3 (multilingual, excellent for Vietnamese)

**Vector DB**: Chroma (simple) or PGVector on AWS RDS

**Example RAG code (LangChain)**:
```python
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

vectorstore = Chroma(persist_directory="./fashion_vector_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

prompt_template = """
Bạn là nhân viên hỗ trợ của StyleVN.
Chỉ sử dụng thông tin từ context bên dưới để trả lời.
Nếu không có thông tin, hãy trả lời: "Xin lỗi, em chưa có thông tin này ạ."

Context: {context}
Câu hỏi: {question}

Trả lời ngắn gọn, lịch sự, bằng tiếng Việt:
"""
```

## 7. Guardrails & Faithfulness Check

**Simple post-generation verifier**:
```python
def check_faithfulness(answer, context):
    verification_prompt = f"""
    Context: {context}
    Answer: {answer}
    Mọi thông tin trong Answer có được hỗ trợ trực tiếp bởi Context không? 
    Trả lời chỉ JSON: {{"pass": true/false, "reason": "..." }}
    """
    # Call small judge model (Phi-3-mini or Llama-3.1-8B)
    response = judge_llm.invoke(verification_prompt)
    return json.loads(response)
```

## 8. Model Quantization

**Use AWQ or GPTQ for best performance with vLLM**:
```bash
# After merging LoRA
python -m awq.entry --model_path ./merged_model --quant_path ./qwen2.5-7b-awq --quantize
```

## 9. vLLM Deployment on AWS

**Launch vLLM server**:
```bash
vllm serve ./qwen2.5-7b-fashionvn-merged \
  --quantization awq \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 \
  --port 8000
```

**Test**:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
```

## 10. Frontend & Zalo Integration

**Simple Streamlit demo**:
```python
import streamlit as st
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

st.title("StyleVN Assistant")
if prompt := st.chat_input("Hỏi về sản phẩm, đơn hàng..."):
    response = client.chat.completions.create(
        model="qwen2.5-7b-fashionvn",
        messages=[{"role": "user", "content": prompt}]
    )
    st.write(response.choices[0].message.content)
```

**Zalo OA webhook** (FastAPI example):
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/zalo-webhook")
async def zalo_webhook(request: dict):
    user_message = request["message"]["text"]
    # Call vLLM → send back via Zalo API
    return {"response": ai_response}
```

## 11. Evaluation & Monitoring

- Use **LangSmith** or **Phoenix** for tracing
- Run weekly eval on 200 held-out Vietnamese questions
- Track: faithfulness, latency, thumbs-up rate, escalation rate

## 12. Security & Compliance

- Never expose raw order data in prompt
- Mask sensitive info (phone, address)
- Rate limiting on vLLM
- Log all conversations (anonymized)

## 13. Cost Breakdown (Monthly, g6.xlarge)

- EC2 g6.xlarge: $580–790 (On-Demand) → $300–450 with Savings Plan
- Storage + data transfer: $30–60
- Total monthly: **$350–550** (after optimization)
- Break-even: ~6–9 months at 1,500 chats/day

## 14. Maintenance & Iteration

- Weekly: collect new good/bad examples → re-fine-tune LoRA adapter (2-4 hours)
- Monthly: update product catalog embeddings
- Quarterly: evaluate base model upgrade (Qwen3, Llama-4, etc.)

**Next Action**: Start with Step 3 (launch AWS instance) and collect your first 200 FAQ examples.

