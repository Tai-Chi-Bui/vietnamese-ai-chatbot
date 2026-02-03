# vietnamese-ai-chatbot
### Project name: StyleVN Assistant
### Business goal: Automate 70%-80% of customer support chats on Zalo, facebook, website
### Target: Vietnamese-speaking customers (slang, abbreviations, casual tones)
### Base model: Qwen2.5-7B-Instruct
### Fine-tuning methods: QLoRA
### Deployment strategy: Cloud-first -> AWS
### Infrastructure budget goal: Realistic mid-size company (initial investment ~$4,000–$10,000 USD, monthly running cost target $150–400)

## Step 1: High-level architecture

Here's your architecture diagram converted into clean Markdown format:

```markdown
User (Zalo / Web / FB Messenger)
          ↓
[Frontend / Webhook]
  └─ simple web interface + Zalo/FB integration
          ↓
[API Gateway / Load Balancer]
  └─ AWS API Gateway or Application Load Balancer
          ↓
vLLM inference server
  └─ OpenAI-compatible endpoint
          ↓
Qwen2.5-7B-Instruct + LoRA adapter
  └─ fine-tuned on fashion data
          ↓
RAG pipeline:
  ├─ Vector DB
  │    └─ product catalog + FAQs + policies
  ├─ Real-time order lookup
  │    └─ via database connector
  ├─ Query rewriting + re-ranking
  └─ Strict faithfulness guardrail
          ↓
Post-processing & safety checks
          ↓
Response back to user
```


```
