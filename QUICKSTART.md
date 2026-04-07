# 🚀 Quick Start: Deploy to Hugging Face with Grok API

Follow these steps to deploy ContentModerationEnv to Hugging Face Spaces with Grok support:

## 1. Get Your API Key

Get your Grok API key from [x.ai](https://x.ai/api) or use OpenAI key from [platform.openai.com](https://platform.openai.com/api-keys)

## 2. Create HF Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space)
- Choose **Docker** SDK
- Name it (e.g., `content-moderation-env`)
- Select **Public** or **Private**

## 3. Deploy (Three Options)

### Option A: From GitHub (Easiest! ⭐)

1. Click **"Import from GitHub"** when creating Space
2. Enter your GitHub repo URL: `https://github.com/YOUR_USERNAME/YOUR_REPO`
3. Click Create Space
4. Done! HF auto-deploys from GitHub

**→ See [DEPLOY_FROM_GITHUB.md](DEPLOY_FROM_GITHUB.md) for details**

### Option B: Automatic Script

```bash
# Install HF CLI if not already installed
pip install huggingface_hub
huggingface-cli login

# Run deployment script
./deploy.sh YOUR_USERNAME SPACE_NAME
```

### Option C: Manual

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
cd SPACE_NAME

# Copy all files
cp -r /path/to/Meta/* .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

## 4. Configure Secrets

Go to your Space Settings → Repository secrets and add:

**For Grok:**
```
XAI_API_KEY=your-xai-api-key-here
LLM_PROVIDER=grok
MODEL_NAME=grok-beta
```

**For OpenAI:**
```
OPENAI_API_KEY=your-openai-key-here
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
```

## 5. Wait for Build

Monitor the **Logs** tab. Build takes ~5-10 minutes.

## 6. Test Your Space

Visit: `https://YOUR_USERNAME-SPACE_NAME.hf.space`

---

## Test Locally First

```bash
# Test configuration
python3 test_deployment.py

# Or run with Docker
docker build -t content-mod .
docker run -p 7860:7860 \
  -e LLM_PROVIDER=grok \
  -e XAI_API_KEY=your-key \
  content-mod
```

---

## Troubleshooting

**Build fails?**
- Check Logs tab for errors
- Verify Dockerfile and requirements.txt exist

**API not working?**
- Confirm secret names are exact (case-sensitive)
- No extra spaces in secret values
- Try regenerating API key

**Need help?**
See full guide: [DEPLOY.md](DEPLOY.md)
