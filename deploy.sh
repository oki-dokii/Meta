#!/bin/bash
# Quick deployment script for Hugging Face Spaces
# Usage: ./deploy.sh YOUR_USERNAME SPACE_NAME

set -e

if [ $# -ne 2 ]; then
    echo "Usage: ./deploy.sh YOUR_USERNAME SPACE_NAME"
    echo "Example: ./deploy.sh myusername content-moderation-env"
    exit 1
fi

USERNAME=$1
SPACE_NAME=$2
REPO_URL="https://huggingface.co/spaces/${USERNAME}/${SPACE_NAME}"

echo "🚀 Deploying to Hugging Face Spaces"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Username: ${USERNAME}"
echo "Space: ${SPACE_NAME}"
echo "URL: ${REPO_URL}"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚠️  huggingface-cli not found. Installing..."
    pip install huggingface_hub
fi

# Check if user is logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ Not logged in to Hugging Face"
    echo "Run: huggingface-cli login"
    exit 1
fi

# Clone or update the space
TEMP_DIR="/tmp/hf_space_${SPACE_NAME}"
if [ -d "${TEMP_DIR}" ]; then
    echo "📁 Updating existing clone..."
    cd "${TEMP_DIR}"
    git pull
else
    echo "📥 Cloning space repository..."
    git clone "${REPO_URL}" "${TEMP_DIR}"
    cd "${TEMP_DIR}"
fi

# Copy files
echo "📋 Copying project files..."
cp -r "${OLDPWD}"/* . 2>/dev/null || true
cp "${OLDPWD}"/.gitignore . 2>/dev/null || true

# Remove .env files (secrets go in HF settings)
rm -f .env .env.local

# Commit and push
echo "📤 Pushing to Hugging Face..."
git add .
git commit -m "Deploy ContentModerationEnv with Grok/OpenAI support

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" || echo "No changes to commit"
git push

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Set API keys in Space Settings → Repository secrets:"
echo "   • XAI_API_KEY (for Grok)"
echo "   • OPENAI_API_KEY (for OpenAI)"
echo "   • LLM_PROVIDER=grok or openai"
echo ""
echo "2. Visit your Space:"
echo "   ${REPO_URL}"
echo ""
echo "3. Monitor build logs in the Logs tab"
echo ""
echo "📖 Full guide: DEPLOY.md"
