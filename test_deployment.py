#!/usr/bin/env python3
"""
Test script to verify Grok API configuration before deployment
"""
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_configuration():
    """Test environment configuration"""
    print("🧪 Testing Configuration")
    print("=" * 50)
    
    # Check provider
    provider = os.getenv("LLM_PROVIDER", "openai")
    print(f"✓ Provider: {provider}")
    
    # Check API keys
    if provider == "grok":
        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        key_var = "XAI_API_KEY or GROK_API_KEY"
        default_model = "grok-beta"
        default_url = "https://api.x.ai/v1"
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
        key_var = "OPENAI_API_KEY or HF_TOKEN"
        default_model = "gpt-4o-mini"
        default_url = "https://api.openai.com/v1"
    
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✓ API Key ({key_var}): {masked}")
    else:
        print(f"✗ API Key ({key_var}): NOT SET")
        return False
    
    # Check model
    model = os.getenv("MODEL_NAME", default_model)
    print(f"✓ Model: {model}")
    
    # Check API URL
    api_url = os.getenv("API_BASE_URL", default_url)
    print(f"✓ API URL: {api_url}")
    
    print("=" * 50)
    return True

def test_api_connection():
    """Test actual API connection"""
    print("\n🔌 Testing API Connection")
    print("=" * 50)
    
    try:
        from openai import OpenAI
        from inference import API_BASE_URL, MODEL_NAME, API_KEY, PROVIDER
        
        if not API_KEY:
            print("✗ No API key configured")
            return False
        
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        
        print(f"Testing {PROVIDER} API...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello' if you can read this."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content
        print(f"✓ API Response: {result}")
        print("✓ Connection successful!")
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def test_environment_loading():
    """Test environment file loading"""
    print("\n📁 Testing Environment Loading")
    print("=" * 50)
    
    from content_moderation_env import ContentModerationEnv
    
    try:
        env = ContentModerationEnv("moderation_benchmark.json", seed=42)
        print(f"✓ Environment loaded: {env.num_scenarios} scenarios")
        
        state = env.reset()
        print(f"✓ Reset successful: scenario loaded")
        
        result = env.step({"label": "safe", "action": "allow"})
        print(f"✓ Step successful: reward = {result['reward']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "🚀 ContentModerationEnv - Pre-Deployment Tests ".center(50, "="))
    print()
    
    results = []
    
    # Test 1: Configuration
    results.append(("Configuration", test_configuration()))
    
    # Test 2: Environment
    results.append(("Environment", test_environment_loading()))
    
    # Test 3: API Connection (only if config passed)
    if results[0][1]:
        results.append(("API Connection", test_api_connection()))
    else:
        print("\n⚠️  Skipping API test (configuration incomplete)")
    
    # Summary
    print("\n" + "📊 Test Summary ".center(50, "="))
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        all_passed = all_passed and passed
    
    print("=" * 50)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to deploy.")
        print("\nNext steps:")
        print("  1. Run: ./deploy.sh YOUR_USERNAME SPACE_NAME")
        print("  2. Or manually: git push to your HF Space")
        print("  3. Set secrets in HF Space Settings")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before deploying.")
        print("\nTroubleshooting:")
        print("  1. Set environment variables (see .env.example)")
        print("  2. Verify API key is valid")
        print("  3. Check network connectivity")
        return 1

if __name__ == "__main__":
    sys.exit(main())
