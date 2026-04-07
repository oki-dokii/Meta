# 🔗 Deploy from GitHub to Hugging Face Spaces

## Method 1: Use HF Space as Git Remote (Recommended!)

This is the **actual working method** that HF Spaces uses:

### Step 1: Create Empty Space on HF

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name**: `content-moderation-env` (or your choice)
   - **License**: MIT
   - **SDK**: **Docker**
   - **Visibility**: Public or Private
3. Click **Create Space**
4. You'll see an empty space with setup instructions

### Step 2: Push from Your GitHub Repo to HF

**Option A: Using SSH (Recommended - No Passwords!) 🔐**

```bash
# In your local repo directory
cd /path/to/your/Meta

# Add HF Space as a git remote (SSH)
git remote add space git@hf.co:spaces/YOUR_HF_USERNAME/SPACE_NAME

# Push to HF Space
git push space main
```

**Option B: Using HTTPS (Requires Token)**

```bash
# Add HF Space as a git remote (HTTPS)
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/SPACE_NAME

# Push to HF Space (will prompt for username/token)
git push space main
```

✨ **Done!** Your code is now on HF and deploying!

---

## 🔐 SSH Setup (One-Time)

If you haven't added your SSH key to Hugging Face yet:

### 1. Generate SSH Key (if you don't have one)

```bash
# Check if you have an SSH key
ls ~/.ssh/id_*.pub

# If not, generate one
ssh-keygen -t ed25519 -C "your-email@example.com"
# Press Enter to accept defaults
```

### 2. Copy Your Public Key

```bash
# Display your public key
cat ~/.ssh/id_ed25519.pub

# Or copy to clipboard (Linux)
cat ~/.ssh/id_ed25519.pub | xclip -selection clipboard

# Or (Mac)
cat ~/.ssh/id_ed25519.pub | pbcopy
```

### 3. Add to Hugging Face

1. Go to [huggingface.co/settings/keys](https://huggingface.co/settings/keys)
2. Click **"Add SSH key"**
3. Paste your public key
4. Click **"Add key"**

### 4. Test Connection

```bash
# Test SSH connection to HF
ssh -T git@hf.co

# You should see: "Hi YOUR_USERNAME, welcome to Hugging Face!"
```

✅ **Now you can use SSH for all HF git operations!**

### Step 3: Keep GitHub as Primary (Optional)

```bash
# Continue using GitHub as normal
git push origin main  # Push to GitHub

# Also push to HF when you want to deploy
git push space main   # Deploy to HF
```

### Troubleshooting SSH

**"Permission denied (publickey)"**
```bash
# Check which key is being used
ssh -vT git@hf.co

# If needed, specify key explicitly
git remote set-url space git@hf.co:spaces/YOUR_USERNAME/SPACE_NAME
ssh-add ~/.ssh/id_ed25519
```

**Using different SSH key**
```bash
# Add to ~/.ssh/config
cat >> ~/.ssh/config << 'EOF'
Host hf.co
    HostName hf.co
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
EOF
```

---

## Method 2: Clone HF Space, Copy Files, Push

Alternative if you prefer starting fresh:

```bash
# Clone the empty HF Space
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/SPACE_NAME
cd SPACE_NAME

# Copy your project files
cp -r /path/to/your/Meta/* .
cp /path/to/your/Meta/.gitignore .

# Commit and push
git add .
git commit -m "Initial deployment with Grok support"
git push
```

---

## Method 3: GitHub Actions Auto-Deploy (Advanced)

Keep GitHub as source of truth, auto-deploy to HF on every push:

### Setup GitHub Action

Create `.github/workflows/deploy-to-hf.yml` in your GitHub repo:

```yaml
name: Deploy to Hugging Face

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config --global user.email "actions@github.com"
          git config --global user.name "GitHub Actions"
          git remote add space https://YOUR_HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/YOUR_HF_USERNAME/SPACE_NAME
          git push space main --force
```

### Add HF Token to GitHub

1. Get HF token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (write access)
2. Go to your GitHub repo → Settings → Secrets and variables → Actions
3. Add secret: `HF_TOKEN` = your token

Now every push to GitHub auto-deploys to HF! 🚀

---

## Configure Secrets (Required!)

After connecting, set your API keys:

1. Go to **Settings** → **Repository secrets**
2. Add secrets (click **+ New secret**):

**For Grok:**
```
Name: XAI_API_KEY
Value: your-grok-api-key-here

Name: LLM_PROVIDER
Value: grok

Name: MODEL_NAME (optional)
Value: grok-beta
```

**For OpenAI:**
```
Name: OPENAI_API_KEY
Value: your-openai-key-here

Name: LLM_PROVIDER
Value: openai
```

3. Click **Save**

---

## Push Changes and Auto-Deploy

Now every time you push to GitHub:

```bash
git add .
git commit -m "Update content moderation env"
git push origin main
```

→ **Hugging Face automatically rebuilds** and deploys! 🚀

---

## Verify Your README.md Header

Make sure your `README.md` starts with this YAML frontmatter (it already does):

```yaml
---
title: ContentModerationEnv
emoji: 🛡️
colorFrom: indigo
colorTo: violet
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - benchmark
  - content-moderation
---
```

This tells HF Spaces to use Docker SDK.

---

## Monitor Build

1. Go to your Space URL: `https://huggingface.co/spaces/YOU/SPACE`
2. Click **Logs** tab
3. Watch the build progress (~5-10 min first time)
4. Status will change to **Running** when ready

---

## Benefits of GitHub Integration

✅ **Auto-sync** - Push to GitHub → Auto-deploys to HF
✅ **No manual copying** - Works directly from your repo
✅ **Version control** - All changes tracked in GitHub
✅ **Easy rollback** - Use git to revert bad deploys
✅ **CI/CD** - Can add GitHub Actions for tests

---

## Troubleshooting

### Build fails after connecting GitHub

**Check:**
- Dockerfile exists in repo root
- requirements.txt exists
- README.md has correct YAML header
- All files are committed to GitHub

### Space not updating after push

**Try:**
- Force rebuild: Settings → Factory Reboot
- Check build logs for errors
- Verify GitHub webhook is active (Settings)

### Can't link to GitHub

**Make sure:**
- Repository is public (or you authorized private access)
- You have admin rights to the repo
- GitHub app permissions are granted

---

## GitHub Actions (Optional Advanced Setup)

You can add automated tests before deployment:

```yaml
# .github/workflows/test.yml
name: Test before HF Deploy

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python3 test_deployment.py
```

---

## Quick Command Summary

```bash
# 1. Push your code to GitHub (if not already)
git add .
git commit -m "Ready for HF Spaces"
git push origin main

# 2. Create HF Space with GitHub import
# → Use web UI: huggingface.co/new-space
# → Click "Import from GitHub"
# → Enter your repo URL

# 3. Configure secrets in HF Space Settings

# 4. Done! Future updates:
git commit -am "Update"
git push
# → Auto-deploys to HF! ✨
```

---

## Your GitHub Repo URL Format

```
https://github.com/YOUR_USERNAME/YOUR_REPO_NAME
```

For example:
```
https://github.com/yourusername/Meta
```

---

Need help? Check the [HF Spaces docs](https://huggingface.co/docs/hub/spaces-github-actions)
