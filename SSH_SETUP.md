# 🔐 SSH Setup for Hugging Face Spaces

Using SSH is the **easiest way** to push to Hugging Face - no passwords or tokens needed!

---

## Quick Setup (5 Minutes)

### 1. Check if You Have an SSH Key

```bash
ls ~/.ssh/id_*.pub
```

**If you see files** → Skip to Step 3  
**If not** → Continue to Step 2

---

### 2. Generate SSH Key (One-Time)

```bash
# Generate a new SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Press Enter to accept default location
# Press Enter twice for no passphrase (or set one if you prefer)
```

You'll see:
```
Your identification has been saved in /home/username/.ssh/id_ed25519
Your public key has been saved in /home/username/.ssh/id_ed25519.pub
```

---

### 3. Copy Your Public Key

```bash
# Display your public key
cat ~/.ssh/id_ed25519.pub
```

**Linux (with xclip):**
```bash
cat ~/.ssh/id_ed25519.pub | xclip -selection clipboard
```

**Mac:**
```bash
cat ~/.ssh/id_ed25519.pub | pbcopy
```

**Or just copy the output manually** - it looks like:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJl3dIeudNqd0DMRA5AyJ... your-email@example.com
```

---

### 4. Add to Hugging Face

1. Go to **[huggingface.co/settings/keys](https://huggingface.co/settings/keys)**
2. Click **"Add SSH key"**
3. Give it a name (e.g., "My Laptop")
4. Paste your public key
5. Click **"Add key"**

✅ Done!

---

### 5. Test Connection

```bash
# Test SSH to Hugging Face
ssh -T git@hf.co
```

**Expected output:**
```
Hi YOUR_USERNAME, welcome to Hugging Face!
```

If you see this → ✅ **SSH is working!**

---

## Using SSH with HF Spaces

### Deploy Your Project

```bash
# In your project directory
cd /home/jdsb/Desktop/BiULding/Hackathons/Meta

# Add HF Space as remote (using SSH)
git remote add space git@hf.co:spaces/YOUR_USERNAME/SPACE_NAME

# Push to deploy
git push space main
```

### Or Use the Script

```bash
./deploy-to-hf.sh YOUR_USERNAME SPACE_NAME ssh
```

---

## Troubleshooting

### "Permission denied (publickey)"

**Check which key SSH is trying:**
```bash
ssh -vT git@hf.co 2>&1 | grep "Offering public key"
```

**Add your key to SSH agent:**
```bash
# Start SSH agent
eval "$(ssh-agent -s)"

# Add your key
ssh-add ~/.ssh/id_ed25519

# Verify it's added
ssh-add -l
```

**Still not working?** Specify the key explicitly:
```bash
ssh -i ~/.ssh/id_ed25519 -T git@hf.co
```

---

### "Could not resolve hostname hf.co"

Try the full hostname:
```bash
git remote set-url space git@huggingface.co:spaces/YOUR_USERNAME/SPACE_NAME
```

---

### Using Multiple SSH Keys

If you have multiple SSH keys (e.g., one for GitHub, one for HF), configure `~/.ssh/config`:

```bash
# Edit SSH config
nano ~/.ssh/config

# Add this:
Host hf.co huggingface.co
    HostName huggingface.co
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa
    IdentitiesOnly yes
```

Save and test:
```bash
ssh -T git@hf.co
ssh -T git@github.com
```

---

### "ssh-keygen: command not found"

**Linux:**
```bash
sudo apt-get install openssh-client
```

**Mac:**
OpenSSH should be pre-installed. If not:
```bash
brew install openssh
```

---

## SSH vs HTTPS Comparison

| Feature | SSH | HTTPS |
|---------|-----|-------|
| **Setup** | One-time key setup | Token per machine |
| **Push** | No password | Token or password |
| **Security** | More secure | Secure but token-based |
| **Speed** | Faster | Slightly slower |
| **Ease** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Recommendation:** Use SSH! 🔐

---

## Quick Reference

```bash
# Generate key
ssh-keygen -t ed25519 -C "your-email@example.com"

# View public key
cat ~/.ssh/id_ed25519.pub

# Add to HF
# → https://huggingface.co/settings/keys

# Test connection
ssh -T git@hf.co

# Add remote (SSH)
git remote add space git@hf.co:spaces/USERNAME/SPACE_NAME

# Push
git push space main

# Update remote URL to SSH
git remote set-url space git@hf.co:spaces/USERNAME/SPACE_NAME
```

---

## Need Help?

- **HF SSH Docs:** [huggingface.co/docs/hub/security-git-ssh](https://huggingface.co/docs/hub/security-git-ssh)
- **GitHub SSH Guide:** [docs.github.com/en/authentication/connecting-to-github-with-ssh](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
- **Test deployment:** `./deploy-to-hf.sh USERNAME SPACE ssh`

---

✨ **You're all set!** Now you can push to HF Spaces without any password prompts.
