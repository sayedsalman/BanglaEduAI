import subprocess
import sys
import os

def main():
    print("🚀 Starting BanglaEduAI...")
    # Start API in background
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("✅ API running at http://localhost:8000")
    # Start Gradio frontend
    frontend_proc = subprocess.Popen(
        [sys.executable, "frontend/app.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("✅ Frontend running at http://localhost:7860")
    print("Press Ctrl+C to stop.")
    try:
        api_proc.wait()
    except KeyboardInterrupt:
        api_proc.terminate()
        frontend_proc.terminate()
        print("\n🛑 Services stopped.")

if __name__ == "__main__":
    main()
