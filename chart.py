import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Table I: Throughput Performance
# -------------------------------
conditions = ["Home WiFi", "Home WiFi + VPN", "Mobile Hotspot"]
throughput = [102, 17, 8.6]
retransmissions = [0, 38, 21]

plt.figure(figsize=(8, 5))
plt.bar(conditions, throughput)
plt.title("Throughput Comparison (Mbps)")
plt.ylabel("Average Throughput (Mbps)")
plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------
# Table II: RTT Measurements
# -------------------------------
min_rtt = [243.6, 194.6, 228.3]
avg_rtt = [296.2, 208.2, 292.3]
max_rtt = [345.3, 307.9, 455.5]

x = np.arange(len(conditions))
width = 0.25
colors=['#1a80bb', '#ea801c', '#b8b8b8']
plt.figure(figsize=(10, 5))
plt.bar(x - width, min_rtt, width, label="Min RTT", color=colors[0])
plt.bar(x, avg_rtt, width, label="Avg RTT", color=colors[1])
plt.bar(x + width, max_rtt, width, label="Max RTT", color=colors[2])
plt.xticks(x, conditions)
plt.ylabel("RTT (ms)")
plt.title("RTT Summary (Ping Measurements)")
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------
# Table III: Hop Count
# -------------------------------
hops = [25, 18, 0]

plt.figure(figsize=(8, 5))
plt.bar(conditions, hops, color='orange')
plt.title("Visible Hop Count (Traceroute)")
plt.ylabel("Number of Hops")
plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
