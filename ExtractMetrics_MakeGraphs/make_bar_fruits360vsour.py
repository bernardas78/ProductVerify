import matplotlib.pyplot as plt
import numpy as np
import os

plot_title = "Verification ROC AUC ~ Dataset"
legend_title = "ROC AUC"
dest_filename = "bar_fruits360vsour.png"

species = ("Center Loss", "Siamese", "Triplet")
penguin_means = {
    'Self-checkout': (0.979, 0.981, 0.980),     # data from roc_bymethod_bestdist.png
    'Fruits 360': (1.000, 0.976, 0.997),          # data from roc_bymethod_bestdist_fruits360.png
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    print ("attribute: {}".format(attribute))
    print ("measurement: {}".format(measurement))
    offset = width * multiplier
    print ("offset: {}".format(offset))
    rects = plt.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3, fmt='%.3f')
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('ROC AUC')
ax.set_title ( plot_title )
ax.set_xticks(x + width/2, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0.95, 1.01)

#y_ticks=["0.950","1.000"]
ax.set_yticks([.95, .96, .97, .98, .99, 1.], ["0.950","0.960","0.970","0.980","0.990","1.000"])


plt.tight_layout()
plt.savefig ( os.path.join(r"A:\IsKnown_Results\Dists",dest_filename ) )
plt.close()