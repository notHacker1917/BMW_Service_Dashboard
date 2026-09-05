"""Environment verification and setup helper."""
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("  WARNING: Python 3.11+ recommended")
        return False
    return True


def check_dependencies(): 
    """Check if all dependencies are installed."""
    required_packages = [
        "pandas",
        "numpy",
        "sklearn",  # scikit-learn
        "sentence_transformers",
        "hdbscan",
        "umap",     # umap-learn
        "litellm",
        "langdetect",
        "fastapi",
        "uvicorn",
        "streamlit",
        "plotly",
        "loguru",
        "tenacity",
        "aiohttp",
        "dotenv",   # python-dotenv
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package if package != "sklearn" else "sklearn")
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (MISSING)")
            missing.append(package)
    
    return len(missing) == 0


def check_directories():
    """Check if required directories exist."""
    base_dir = Path(__file__).parent / "yamaha_feedback_ai"
    required_dirs = [
        "app",
        "app/preprocessing",
        "app/extraction",
        "app/embedding",
        "app/clustering",
        "app/labeling",
        "app/dashboard",
        "app/api",
        "app/database",
        "app/utils",
        "data/raw",
        "data/processed",
        "data/outputs",
    ]
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}")
        else:
            print(f"✗ {dir_name} (MISSING)")
            return False
    
    return True


def check_environment_file():
    """Check .env file."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"✓ .env file exists")
        # Check for API key
        with open(env_file, 'r') as f:
            content = f.read()
            if "OPENAI_API_KEY=" in content:
                print("  NOTE: Ensure OPENAI_API_KEY is set to a valid key")
        return True
    else:
        print(f"✗ .env file not found")
        return False


def main():
    print("\n" + "="*60)
    print("YAMAHA FEEDBACK ANALYSIS - ENVIRONMENT CHECK")
    print("="*60 + "\n")
    
    print("[Python]")
    py_ok = check_python_version()
    print()
    
    print("[Dependencies]")
    deps_ok = check_dependencies()
    print()
    
    print("[Directory Structure]")
    dirs_ok = check_directories()
    print()
    
    print("[Configuration]")
    env_ok = check_environment_file()
    print()
    
    print("="*60)
    if py_ok and deps_ok and dirs_ok and env_ok:
        print("✓ Environment is ready!")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Run: python run_pipeline.py")
        print("3. View results in Streamlit dashboard")
    else:
        print("✗ Some checks failed. See above for details.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
