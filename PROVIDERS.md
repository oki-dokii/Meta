# 🚀 Multi-Provider API Support

ContentModerationEnv now supports **three LLM providers**:

1. **Groq** (Default) - Fast & free tier available
2. **OpenAI** - Industry standard
3. **Grok** (X.AI) - Latest from X/Twitter

---

## Quick Start

### Groq (Recommended - Fast & Free!)

```bash
# Get API key: https://console.groq.com
export LLM_PROVIDER=groq          # or omit (groq is default)
export GROQ_API_KEY=gsk_...
python3 inference.py
```

**Available Models:**
- `llama-3.3-70b-versatile` (default) - Best quality
- `llama-3.1-8b-instant` - Fastest
- `mixtral-8x7b-32768` - Long context
- `gemma2-9b-it` - Efficient

### OpenAI

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini     # optional
python3 inference.py
```

### Grok (X.AI)

```bash
export LLM_PROVIDER=grok
export XAI_API_KEY=xai-...
export MODEL_NAME=grok-beta       # optional
python3 inference.py
```

---

## Hugging Face Space Configuration

### Set Secrets in HF Space

Go to: `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME/settings`

**For Groq (Recommended):**
```
LLM_PROVIDER = groq
GROQ_API_KEY = gsk_your_actual_key_here
MODEL_NAME = llama-3.3-70b-versatile
```

**For OpenAI:**
```
LLM_PROVIDER = openai
OPENAI_API_KEY = sk_your_actual_key_here
MODEL_NAME = gpt-4o-mini
```

**For Grok:**
```
LLM_PROVIDER = grok
XAI_API_KEY = xai_your_actual_key_here
MODEL_NAME = grok-beta
```

---

## Getting API Keys

### Groq
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Go to API Keys
4. Create new key
5. **Free tier includes generous limits!**

### OpenAI
1. Visit [platform.openai.com](https://platform.openai.com)
2. Sign up
3. Add payment method
4. Create API key

### Grok (X.AI)
1. Visit [x.ai/api](https://x.ai/api)
2. Sign up
3. Generate API key

---

## Provider Comparison

| Provider | Speed | Cost | Free Tier | Best For |
|----------|-------|------|-----------|----------|
| **Groq** | ⚡⚡⚡ Very Fast | $ Low | ✅ Yes | Development, Testing |
| **OpenAI** | ⚡⚡ Fast | $$ Medium | ❌ No | Production |
| **Grok** | ⚡⚡ Fast | $$ Medium | Limited | Latest features |

---

## Advanced Configuration

### Custom Model

```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=gsk_...
export MODEL_NAME=llama-3.1-8b-instant  # Override default
python3 inference.py
```

### Custom API Endpoint

```bash
export LLM_PROVIDER=openai
export API_BASE_URL=https://your-custom-endpoint.com/v1
export OPENAI_API_KEY=your-key
python3 inference.py
```

---

## Testing

### Test Provider Configuration

```bash
python3 test_providers.py
```

### Test with Actual API

```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=your-key

# Run on a single scenario
python3 -c "
from content_moderation_env import ContentModerationEnv
from inference import get_llm_action, client

env = ContentModerationEnv('moderation_benchmark.json')
state = env.reset('scen_easy_1')
action, error = get_llm_action(client, state, 'easy')
print(f'Action: {action}')
print(f'Error: {error}')
"
```

---

## Troubleshooting

### "No API key found"

**Groq:**
```bash
export GROQ_API_KEY=gsk_...
```

**OpenAI:**
```bash
export OPENAI_API_KEY=sk_...
# OR
export HF_TOKEN=hf_...
```

**Grok:**
```bash
export XAI_API_KEY=xai_...
# OR
export GROK_API_KEY=xai_...
```

### "Rate limit exceeded"

- **Groq**: Very generous free tier, but wait if hit
- **OpenAI**: Add credits or reduce rate
- **Grok**: Check your usage limits

### Wrong provider being used

```bash
# Explicitly set provider
export LLM_PROVIDER=groq  # or openai, or grok

# Verify
python3 -c "from inference import PROVIDER; print(f'Using: {PROVIDER}')"
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | Provider: `groq`, `openai`, or `grok` |
| `GROQ_API_KEY` | — | Groq API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `XAI_API_KEY` | — | Grok/X.AI API key |
| `MODEL_NAME` | Provider-specific | Override default model |
| `API_BASE_URL` | Provider-specific | Custom API endpoint |

---

## Why Groq is Default

✅ **Fast** - 10x faster inference than OpenAI  
✅ **Free tier** - Generous limits for development  
✅ **Open models** - Llama, Mixtral, Gemma  
✅ **OpenAI compatible** - Same API format  
✅ **No credit card** - Start immediately  

Perfect for development, testing, and demos!

---

## Production Recommendations

- **Development**: Groq (free, fast)
- **Production**: OpenAI or Groq (depending on volume)
- **Experimentation**: Grok (latest features)

---

## Need Help?

- **Groq Docs**: [console.groq.com/docs](https://console.groq.com/docs)
- **OpenAI Docs**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **Grok Docs**: [x.ai/api](https://x.ai/api)

---

✨ **All providers work identically - just change the environment variables!**
