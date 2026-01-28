import random
import signal
import sys
import time
from datetime import datetime

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
        phrase = random.choice(MOCK_PHRASES)
        phrases_used.append(phrase)
        
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S")
        print(f"[{timestamp}] ['{phrase}']")
        sys.stdout.flush()
        
        time.sleep(random.uniform(1.0, 3.0))
    
    print("\nSTREAMING COMPLETE.\n")
    print("FINAL TRANSCRIPT:\n")
    sys.exit(0)

if __name__ == "__main__":
    main()
