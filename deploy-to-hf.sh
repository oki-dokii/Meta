#!/bin/bash
# Simplified deployment script - Push to HF Space directly from GitHub repo
# Supports both SSH and HTTPS

set -e

echo "🚀 Deploy to Hugging Face Space"
echo "================================"
echo ""

# Check arguments
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: ./deploy-to-hf.sh YOUR_HF_USERNAME SPACE_NAME [ssh|https]"
    echo ""
    echo "Example (SSH - recommended):"
    echo "  ./deploy-to-hf.sh myusername content-moderation-env ssh"
    echo ""
    echo "Example (HTTPS):"
    echo "  ./deploy-to-hf.sh myusername content-moderation-env https"
    echo ""
    echo "Default: SSH (if not specified)"
    echo ""
    echo "This will:"
    echo "  1. Add HF Space as a git remote"
    echo "  2. Push your current branch to HF"
    echo "  3. HF will automatically build and deploy"
    exit 1
fi

HF_USERNAME=$1
SPACE_NAME=$2
AUTH_METHOD=${3:-ssh}  # Default to SSH

SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

# Set remote URL based on auth method
if [ "$AUTH_METHOD" = "ssh" ]; then
    REMOTE_URL="git@hf.co:spaces/${HF_USERNAME}/${SPACE_NAME}"
    echo "🔐 Using SSH authentication"
else
    REMOTE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
    echo "🔑 Using HTTPS authentication"
fi

echo "Target Space: ${SPACE_URL}"
echo "Git Remote: ${REMOTE_URL}"
echo ""

# Check if we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  You have uncommitted changes:"
    git status --short
    echo ""
    read -p "Commit them now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "Deploy to HF Spaces

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
    else
        echo "Please commit changes before deploying."
        exit 1
    fi
fi

# Check if remote already exists
if git remote | grep -q "^space$"; then
    echo "✓ Remote 'space' already configured"
    echo "  Updating URL to: ${REMOTE_URL}"
    git remote set-url space "${REMOTE_URL}"
else
    echo "➕ Adding HF Space as git remote 'space'..."
    git remote add space "${REMOTE_URL}"
fi

# Test SSH connection if using SSH
if [ "$AUTH_METHOD" = "ssh" ]; then
    echo ""
    echo "🔍 Testing SSH connection..."
    if ssh -T git@hf.co 2>&1 | grep -q "welcome to Hugging Face"; then
        echo "✅ SSH connection successful"
    else
        echo "⚠️  SSH connection test inconclusive"
        echo "   If push fails, check:"
        echo "   1. SSH key added to HF: https://huggingface.co/settings/keys"
        echo "   2. SSH agent running: ssh-add -l"
    fi
fi

echo ""
echo "📤 Pushing to Hugging Face Space..."
echo ""

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Pushing branch: ${CURRENT_BRANCH} → main"

# Push to HF Space (they use 'main' branch)
if git push space "${CURRENT_BRANCH}:main" 2>&1; then
    echo ""
    echo "✅ Successfully deployed!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎉 Your app is deploying!"
    echo ""
    echo "📍 Space URL: ${SPACE_URL}"
    echo ""
    echo "Next steps:"
    echo "  1. Set API secrets in Space Settings:"
    echo "     • XAI_API_KEY (for Grok)"
    echo "     • LLM_PROVIDER=grok"
    echo ""
    echo "  2. Monitor build: ${SPACE_URL}?logs=container"
    echo ""
    echo "  3. Visit when ready: ${SPACE_URL}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "💡 Future updates:"
    echo "   git push           # Push to GitHub"
    echo "   git push space main  # Deploy to HF"
    echo ""
else
    echo ""
    echo "❌ Push failed!"
    echo ""
    if [ "$AUTH_METHOD" = "ssh" ]; then
        echo "SSH troubleshooting:"
        echo "  1. Create the Space first: https://huggingface.co/new-space"
        echo "  2. Add SSH key: https://huggingface.co/settings/keys"
        echo "  3. Test connection: ssh -T git@hf.co"
        echo "  4. Add key to agent: ssh-add ~/.ssh/id_ed25519"
        echo ""
        echo "Or try HTTPS instead:"
        echo "  ./deploy-to-hf.sh ${HF_USERNAME} ${SPACE_NAME} https"
    else
        echo "HTTPS troubleshooting:"
        echo "  1. Create the Space first: https://huggingface.co/new-space"
        echo "  2. Get token: https://huggingface.co/settings/tokens"
        echo "  3. Use token: git remote set-url space https://USERNAME:TOKEN@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
        echo "  4. Or login: huggingface-cli login"
        echo ""
        echo "Or try SSH instead:"
        echo "  ./deploy-to-hf.sh ${HF_USERNAME} ${SPACE_NAME} ssh"
    fi
    echo ""
    exit 1
fi
