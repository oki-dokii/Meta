# 🚀 Deployment Guide: Hugging Face Spaces with Grok API

This guide will help you deploy ContentModerationEnv to Hugging Face Spaces with Grok API integration.

---

## Prerequisites

1. **Hugging Face Account** — [Sign up at huggingface.co](https://huggingface.co/join)
2. **Grok API Key** — [Get your API key from X.AI](https://x.ai/)
3. **Git** installed locally
4. **Hugging Face CLI** (optional but recommended)

---

## Step 1: Install Hugging Face CLI (Optional)

```bash
pip install huggingface_hub
huggingface-cli login
```

---

## Step 2: Create a New Space on Hugging Face

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Space name**: `content-moderation-env` (or your preferred name)
   - **License**: MIT
   - **Select the SDK**: **Docker**
   - **Space hardware**: CPU basic (free tier) or upgrade if needed
3. Click **Create Space**

---

## Step 3: Clone Your New Space

```bash
# Clone the empty space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/content-moderation-env
cd content-moderation-env
```

---

## Step 4: Copy Project Files

Copy all files from this project to your space directory:

```bash
# From your project root
cp -r /path/to/Meta/* /path/to/content-moderation-env/

# Or if you're in the project directory:
cp * ../content-moderation-env/
cp .gitignore ../content-moderation-env/ 2>/dev/null || true
```

---

## Step 5: Configure Environment Variables in Hugging Face

### Method A: Using the Web Interface (Recommended)

1. Go to your Space page: `https://huggingface.co/spaces/YOUR_USERNAME/content-moderation-env`
2. Click on **Settings** tab
3. Scroll down to **Repository secrets**
4. Add these secrets:

   **For Grok API:**
   - Name: `XAI_API_KEY`
   - Value: `your-actual-grok-api-key`
   
   - Name: `LLM_PROVIDER`
   - Value: `grok`

   **For OpenAI API (alternative):**
   - Name: `OPENAI_API_KEY`
   - Value: `your-openai-api-key`
   
   - Name: `LLM_PROVIDER`
   - Value: `openai`

5. Click **Save**

### Method B: Using CLI

```bash
huggingface-cli repo secrets set XAI_API_KEY YOUR_ACTUAL_KEY \
  --repo-type space --repo YOUR_USERNAME/content-moderation-env

huggingface-cli repo secrets set LLM_PROVIDER grok \
  --repo-type space --repo YOUR_USERNAME/content-moderation-env
```

---

## Step 6: Push to Hugging Face

```bash
cd content-moderation-env
git add .
git commit -m "Initial deployment with Grok API support

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push
```

---

## Step 7: Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/content-moderation-env`
2. Click on **Logs** tab to see the build progress
3. Wait for the Docker container to build (usually 5-10 minutes)
4. Once the status shows **Running**, your app is live! 🎉

---

## Step 8: Test Your Deployment

### Test the Gradio UI
Visit: `https://YOUR_USERNAME-content-moderation-env.hf.space`

### Test the API Endpoints

```bash
# Test reset endpoint
curl -X POST https://YOUR_USERNAME-content-moderation-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'

# Test step endpoint  
curl -X POST https://YOUR_USERNAME-content-moderation-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"label": "safe", "action": "allow"}}'
```

---

## Switching Between OpenAI and Grok

To switch between providers, just update the `LLM_PROVIDER` secret:

```bash
# Switch to Grok
huggingface-cli repo secrets set LLM_PROVIDER grok \
  --repo-type space --repo YOUR_USERNAME/content-moderation-env

# Switch to OpenAI
huggingface-cli repo secrets set LLM_PROVIDER openai \
  --repo-type space --repo YOUR_USERNAME/content-moderation-env
```

Then trigger a rebuild by pushing a small change or using the "Factory Reboot" button in Settings.

---

## Troubleshooting

### Build Fails
- Check the **Logs** tab for error messages
- Ensure `Dockerfile` and `requirements.txt` are present
- Verify all files were copied correctly

### API Key Not Working
- Double-check the secret name matches exactly: `XAI_API_KEY` or `OPENAI_API_KEY`
- Ensure there are no extra spaces in the key value
- Try regenerating the API key from X.AI

### App Not Starting
- Check if port 7860 is correctly exposed in Dockerfile
- Look for Python errors in the build logs
- Verify all dependencies in requirements.txt

### Model Response Issues
- For Grok, ensure you're using `MODEL_NAME=grok-beta` or latest available model
- Check API rate limits on your X.AI account
- Review the logs for API error responses

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | Choose `openai` or `grok` |
| `XAI_API_KEY` | — | Grok/X.AI API key |
| `GROK_API_KEY` | — | Alternative env var for Grok key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `HF_TOKEN` | — | Hugging Face token (fallback) |
| `MODEL_NAME` | `gpt-4o-mini` or `grok-beta` | Model identifier |
| `API_BASE_URL` | Auto-set based on provider | Custom API endpoint |

---

## Local Testing Before Deployment

Test with Docker locally:

```bash
# Build the container
docker build -t content-moderation-env .

# Run with Grok API
docker run -p 7860:7860 \
  -e LLM_PROVIDER=grok \
  -e XAI_API_KEY=your-key-here \
  content-moderation-env

# Visit http://localhost:7860
```

---

## Next Steps

- ⭐ **Star your Space** to make it more discoverable
- 📝 **Update the README.md** with your Space's URL
- 🎨 **Customize the Gradio theme** in `app.py`
- 📊 **Monitor usage** in the Hugging Face Space analytics
- 🔒 **Set your Space to private** if needed (Settings → Visibility)

---

## Support

- **Grok API**: [x.ai/api](https://x.ai/api)
- **Hugging Face Spaces**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Docker SDK**: [huggingface.co/docs/hub/spaces-sdks-docker](https://huggingface.co/docs/hub/spaces-sdks-docker)

---

Happy deploying! 🚀
