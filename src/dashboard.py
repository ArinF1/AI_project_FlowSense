import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

LOG_FILE = Path("src/data/wind_log.csv")

def animate(i):
    if not LOG_FILE.exists():
        return

    data = pd.read_csv(LOG_FILE, names=['time', 'label', 'conf'])
    # Keep only the last 50 readings for the graph
    recent_data = data.tail(100)
    
    counts = recent_data['label'].value_counts()
    
    plt.cla()
    
    # Create the Bar Chart
    colors = ['#2ecc71' if x == 'stationary' else '#3498db' for x in counts.index]
    plt.bar(counts.index, counts.values, color=colors)
    
    # UI Styling
    plt.ylabel("Frequency by frames: 50")
    plt.ylim(0, 50)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Current Status Text
    current_state = recent_data['label'].iloc[-1].upper()
    plt.text(0.5, 1.05, f"CURRENT STATE: {current_state}", 
             transform=plt.gca().transAxes, ha="center", 
             fontsize=12, fontweight='bold', color='red')

fig = plt.figure(figsize=(10, 6))
ani = FuncAnimation(fig, animate, interval=200) # Update every 200ms

plt.tight_layout()
plt.show()