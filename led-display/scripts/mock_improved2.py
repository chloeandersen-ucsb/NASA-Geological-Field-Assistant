import random
import signal
import sys
import time
from datetime import datetime

# Mock phrases that will be "transcribed" for UI testing
MOCK_PHRASES = [
    "hello world",
    "this is a test",
    "rock classification",
    "sample transcription",
    "voice to text",
    "testing the system",
    "mock transcription data",
    "another phrase here",
    "geological field notes",
    "sample collected",
    "location marked",
]

_running = True

def signal_handler(sig, frame):
    global _running
    _running = False
    print("\nSTREAMING COMPLETE.\n")
    print("FINAL TRANSCRIPT:\n")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    print("Using device: cpu")
    print("RECORDING NOW... (Ctrl+C to stop)\n")
    sys.stdout.flush()
    
    time.sleep(0.5)
    
    phrases_used = []
    
    while _running:
        # Simulate streaming transcription
        phrase = random.choice(MOCK_PHRASES)
        phrases_used.append(phrase)
        
        # Format exactly like the real script: [HH:MM:SS] ['phrase']
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S")
        print(f"[{timestamp}] ['{phrase}']")
        sys.stdout.flush()
        
        # Wait a bit before next phrase (simulate real-time transcription)
        time.sleep(random.uniform(1.0, 3.0))
    
    # This shouldn't be reached due to signal handler, but just in case
    print("\nSTREAMING COMPLETE.\n")
    print("FINAL TRANSCRIPT:\n")
    sys.exit(0)

if __name__ == "__main__":
    main()
