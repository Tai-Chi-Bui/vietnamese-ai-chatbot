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

## Step 2: AWS deployment strategy
We'll start with a single-node setup (one GPU instance) for MVP/early production.
Later, we can scale to multi-gpu or auto-scaling when traffic grows.

### Recommendation for MVP:
**g6.xlarge (1x NVIDIA L4 24GB)**:
- Very good price/performance for 7B models in 4-bit
- Can run Qwen2.5-7B comfortably with vLLM + continuous batching
- Expected throughput: 100-250 tokens (depending on the settings)
- Should handle 1000-3000 concurrent chats/day with good latency
If g6.xlarge is not available in your region, use g5.xlarge or g5.2xlarge.

### Cost-saving plan:
- Use Saving Plan (1-year or 3-year) -> 30%-60% off on demand
- Use spot instances for non-critical testing/training -> 60%-90% cheaper
- Run inference only during business hours + scale-to-zero for off-hours (if using SageMaker)
  

