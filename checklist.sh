#!/bin/bash
# Pre-Deployment Checklist - Run this before deploying

echo "🔍 Pre-Deployment Checklist for Hugging Face"
echo "============================================="
echo ""

# Track status
all_good=true

# 1. Check if files exist
echo "📁 Checking required files..."
required_files=(
    "Dockerfile"
    "app.py"
    "requirements.txt"
    "content_moderation_env.py"
    "moderation_benchmark.json"
    "inference.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - MISSING!"
        all_good=false
    fi
done
echo ""

# 2. Check environment variables
echo "🔑 Checking environment configuration..."
if [ -z "$LLM_PROVIDER" ]; then
    echo "  ⚠️  LLM_PROVIDER not set (will default to 'openai')"
else
    echo "  ✅ LLM_PROVIDER=$LLM_PROVIDER"
fi

if [ "$LLM_PROVIDER" = "grok" ]; then
    if [ -z "$XAI_API_KEY" ] && [ -z "$GROK_API_KEY" ]; then
        echo "  ❌ XAI_API_KEY or GROK_API_KEY required for Grok"
        all_good=false
    else
        echo "  ✅ Grok API key found"
    fi
else
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$HF_TOKEN" ]; then
        echo "  ⚠️  OPENAI_API_KEY or HF_TOKEN not set"
        echo "     (You can set this in HF Space secrets)"
    else
        echo "  ✅ OpenAI API key found"
    fi
fi
echo ""

# 3. Check Python environment
echo "🐍 Checking Python environment..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    echo "  ✅ $python_version"
else
    echo "  ❌ Python 3 not found"
    all_good=false
fi
echo ""

# 4. Check dependencies
echo "📦 Checking Python dependencies..."
missing_deps=()
for dep in gradio openai anthropic pydantic; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo "  ✅ $dep installed"
    else
        echo "  ⚠️  $dep not installed (will be installed during deployment)"
        missing_deps+=("$dep")
    fi
done
echo ""

# 5. Check git status
echo "📝 Checking git status..."
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "  ✅ Git repository detected"
    
    # Check if there are uncommitted changes
    if [ -n "$(git status --porcelain)" ]; then
        echo "  ⚠️  You have uncommitted changes:"
        git status --short | head -5
        echo "     (These will be committed during deployment)"
    else
        echo "  ✅ Working directory clean"
    fi
else
    echo "  ℹ️  Not a git repository (this is OK)"
fi
echo ""

# 6. Check HF CLI
echo "🤗 Checking Hugging Face CLI..."
if command -v huggingface-cli &> /dev/null; then
    echo "  ✅ huggingface-cli installed"
    
    if huggingface-cli whoami &> /dev/null; then
        hf_user=$(huggingface-cli whoami | grep "username:" | awk '{print $2}')
        echo "  ✅ Logged in as: $hf_user"
    else
        echo "  ❌ Not logged in. Run: huggingface-cli login"
        all_good=false
    fi
else
    echo "  ⚠️  huggingface-cli not installed"
    echo "     Install: pip install huggingface_hub"
fi
echo ""

# 7. Docker check (optional)
echo "🐳 Checking Docker (optional, for local testing)..."
if command -v docker &> /dev/null; then
    echo "  ✅ Docker installed"
    if docker ps &> /dev/null; then
        echo "  ✅ Docker daemon running"
    else
        echo "  ⚠️  Docker daemon not running"
    fi
else
    echo "  ℹ️  Docker not installed (optional for local testing)"
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$all_good" = true ]; then
    echo "✅ ALL CHECKS PASSED! Ready to deploy."
    echo ""
    echo "Next steps:"
    echo "  1. Get API key from https://x.ai/api or OpenAI"
    echo "  2. Create HF Space at https://huggingface.co/new-space"
    echo "  3. Run: ./deploy.sh YOUR_USERNAME SPACE_NAME"
    echo "  4. Set secrets in HF Space Settings"
    exit 0
else
    echo "⚠️  SOME CHECKS FAILED"
    echo ""
    echo "Please fix the issues marked with ❌ above."
    echo "Then run this checklist again."
    exit 1
fi
