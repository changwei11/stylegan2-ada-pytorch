import json
import matplotlib.pyplot as plt
import os 

# Replace with your actual directory name!
stats_file = './training-runs/00009-wikiart_256_-auto1-batch32-resumeffhq256/stats.jsonl'

# Check if the file exists before running
if not os.path.exists(stats_file):
    print(f"Error: Could not find {stats_file}")
    print("Please check your ./training-runs folder and make sure the directory name matches exactly.")
    exit()
ticks = []
g_losses = []
d_losses = []
r1_penalties = []

print("Reading stats.jsonl...")

# 2. Parse the JSONL file line-by-line
with open(stats_file, 'r') as f:
    for line_num, line in enumerate(f):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            
            # Pull progress tick (or use the line number as an index backup)
            if 'Progress/tick' in data:
                ticks.append(data['Progress/tick']['mean'])
            else:
                ticks.append(line_num)
                
            # Safely extract Generator Loss
            if 'Loss/G/loss' in data:
                g_losses.append(data['Loss/G/loss']['mean'])
            else:
                g_losses.append(None)
                
            # Safely extract Discriminator Loss
            if 'Loss/D/loss' in data:
                d_losses.append(data['Loss/D/loss']['mean'])
            else:
                d_losses.append(None)
                
            # Safely extract R1 Penalty (Great for checking training stability!)
            if 'Loss/r1_penalty' in data:
                r1_penalties.append(data['Loss/r1_penalty']['mean'])
            else:
                r1_penalties.append(None)
                
        except json.JSONDecodeError:
            print(f"Skipping malformed or incomplete data on line {line_num}")

# 3. Plotting the losses
plt.figure(figsize=(12, 6))

# Subplot 1: GAN Losses
plt.subplot(1, 2, 1)
plt.plot(ticks, g_losses, label='Generator Loss', color='#2ca02c', linewidth=2)
plt.plot(ticks, d_losses, label='Discriminator Loss', color='#d62728', linewidth=2)
plt.xlabel('Progress (Ticks)', fontsize=11)
plt.ylabel('Loss Value', fontsize=11)
plt.title('StyleGAN2-ADA Training Losses', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: R1 Regularization Penalty
plt.subplot(1, 2, 2)
plt.plot(ticks, r1_penalties, label='R1 Gradient Penalty', color='#1f77b4', linewidth=2)
plt.xlabel('Progress (Ticks)', fontsize=11)
plt.ylabel('Penalty Value', fontsize=11)
plt.title('R1 Regularization Stability', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# 4. Save the generated visualization
output_image = '00009_train_loss.png'
plt.savefig(output_image, dpi=300)
print(f"Success! Plot successfully saved as '{output_image}'")