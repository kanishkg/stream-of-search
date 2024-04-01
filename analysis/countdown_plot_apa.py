import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid", {'axes.grid' : False})
sns.despine(left=True, bottom=False)
# sns.despine(left=True, bottom=True)
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["font.size"] = 24 
# Data for each reference, with offsets applied
data = {
    "Step": [
        0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
        2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 
        2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 
        3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200
    ],
    "Accuracy": [
        0.505, 0.485, 0.494, 0.502, 0.503, 0.506, 0.516, 0.512, 0.509, 0.523, 0.511, 0.519, 0.522, 0.527, 0.528, 0.53, 0.536,
        0.531, 0.533, 0.539, 0.538, 0.542, 0.54, 0.536, 0.537, 0.542, 0.547, 0.543, 0.537, 
        0.547, 0.553, 0.557, 0.559, 0.562, 0.564, 0.562, 0.558, 0.564, 
         0.576, 0.573, 0.567, 0.568, 0.567, 0.56, 0.576, 0.566, 0.575, 0.573, 0.563
    ],
    "Ref": [
        "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1",
        "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1", "Ref 1",
        "Ref 1", "Ref 1", "Ref 1", 
        "Ref 2", "Ref 2", "Ref 2", "Ref 2", "Ref 2", "Ref 2", "Ref 2", "Ref 2", 
        "Ref 3", "Ref 3", "Ref 3", "Ref 3", "Ref 3", "Ref 3", "Ref 3", "Ref 3", "Ref 3", "Ref 3", "Ref 3", "Ref 3"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set color palette
sns.set_palette("colorblind")

# Plot
plt.figure(figsize=(12, 10))
sns.lineplot(data=df, x="Step", y="Accuracy", hue="Ref")
plt.title('')
plt.xlabel('Training Step')
plt.ylabel('Accuracy')
plt.grid(True, which='major', axis='y', linestyle='-', color='lightgrey', alpha=0.5)
# remove legend
plt.legend().remove()
# remove border
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.tight_layout()

# plt.show()
plt.savefig("plots/apa_accuracy_over_time.svg")
