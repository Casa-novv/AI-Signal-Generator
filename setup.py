#!/usr/bin/env python3
"""
Setup script for AI Forex Signal Generator
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        # Install basic requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def install_ta_lib():
    """Install TA-Lib with platform-specific instructions"""
    print("📊 Installing TA-Lib...")
    
    system = platform.system().lower()
    
    if system == "windows":
        print("🪟 Windows detected")
        print("Please install TA-Lib manually:")
        print("1. Download TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        print("2. Install with: pip install TA_Lib-0.4.25-cp3x-cp3x-win_amd64.whl")
        return True
        
    elif system == "darwin":  # macOS
        print("🍎 macOS detected")
        try:
            # Try to install with homebrew
            subprocess.check_call(["brew", "install", "ta-lib"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "TA-Lib"])
            print("✅ TA-Lib installed successfully")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Failed to install TA-Lib automatically")
            print("Please install manually:")
            print("1. brew install ta-lib")
            print("2. pip install TA-Lib")
            return False
            
    elif system == "linux":
        print("🐧 Linux detected")
        try:
            # Try to install dependencies
            subprocess.check_call(["sudo", "apt-get", "update"])
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "build-essential", "wget"])
            
            # Download and compile TA-Lib
            subprocess.check_call([
                "wget", "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
            ])
            subprocess.check_call(["tar", "-xzf", "ta-lib-0.4.0-src.tar.gz"])
            
            os.chdir("ta-lib")
            subprocess.check_call(["./configure", "--prefix=/usr"])
            subprocess.check_call(["make"])
            subprocess.check_call(["sudo", "make", "install"])
            
            os.chdir("..")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "TA-Lib"])
            
            print("✅ TA-Lib installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install TA-Lib: {e}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "data/csv",
        "data/exports", 
        "data/models",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def create_config_files():
    """Create default configuration files"""
    print("⚙️ Creating configuration files...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write("""# AI Forex Signal Generator Configuration
ENVIRONMENT=development
DEBUG=True
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
""")
        print("✅ Created .env file")
    
    # Create .gitignore if it doesn't exist
    gitignore_file = Path(".gitignore")
    if not gitignore_file.exists():
        with open(gitignore_file, "w") as f:
            f.write("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Data files
data/csv/*.csv
data/exports/*
logs/*.log
*.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml
""")
        print("✅ Created .gitignore file")
    
    return True

def check_mt5_installation():
    """Check if MetaTrader 5 is available"""
    print("🔍 Checking MetaTrader 5 installation...")
    
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader 5 Python package is available")
        
        # Try to initialize (this will fail if MT5 terminal is not installed)
        if mt5.initialize():
            print("✅ MetaTrader 5 terminal is accessible")
            mt5.shutdown()
        else:
            print("⚠️ MetaTrader 5 terminal not found or not running")
            print("You can still use the application with CSV data")
        
        return True
    except ImportError:
        print("❌ MetaTrader 5 Python package not found")
        print("Install with: pip install MetaTrader5")
        return False

def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import streamlit
        import pandas
        import numpy
        import plotly
        print("✅ Core packages imported successfully")
        
        # Test data directory access
        test_file = Path("data/test.txt")
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Data directory is writable")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AI Forex Signal Generator Setup")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Create directories
    if not create_directories():
        success = False
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Install TA-Lib (optional)
    install_ta_lib()  # Don't fail setup if this fails
    
    # Create config files
    if not create_config_files():
        success = False
    
    # Check MT5 (optional)
    check_mt5_installation()  # Don't fail setup if this fails
    
    # Run tests
    if not run_tests():
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("✅ Setup completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Configure your MT5 credentials in .env file (optional)")
        print("2. Run the application: streamlit run app.py")
        print("3. Upload CSV data or connect to MT5")
        print("4. Start generating trading signals!")
    else:
        print("❌ Setup completed with some issues")
        print("Please check the error messages above and resolve them manually")
    
    print("\n📚 Documentation: https://github.com/Casa-novv/AI-Signal-Generator")
    print("🐛 Issues: https://github.com/Casa-novv/AI-Signal-Generator/issues")

if __name__ == "__main__":
    main()
