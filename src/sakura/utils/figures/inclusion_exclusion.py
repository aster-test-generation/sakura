import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data: each criterion is a key, values = [Low, Medium, High]
# True = included, False = excluded
# -----------------------------
data_incl = {
    "One paragraph desc": [True, True, True],
    "Imperative and instructional tone": [True, True, True],
    "Code quoting": [False, False, False],
    "Variable names": [False, False, False],
    "Testing framework": [True, True, True],
    "Mention exact \n identifier names": [True, True, False],
    "Mention exact \n external resources": [True, True, False],
    "Exact test value or range": [True, False, False],
    "Provide example\n test values": [False, True, False],
    "Mention all test annotation": [True, False, False],
    "Mention behavior\n changing test annotation": [True, True, False],
    "Test intent": [True, True, True],
    "Detailed description\n of focal methods": [True, True, False],
    "High level description\n of the user story": [False, False, True],
    "Exact assertion details": [True, True, False],
    "High level assertion details": [False, False, True],
    "Mention exception classes": [True, True, False],
    "Only mention if \nexception occurs or not.": [False, False, True],
    "Detailed description\n of setup methods": [True, True, False],
    "High-level description\n of setup methods": [False, False, True],
    "Specific details \nof the teardown methods": [True, True, False],
    "Specific mention \nof the helper methods": [True, True, False],
    "Include description \nof helper methods \nas part of the test methods": [False, False, True]

}

# -----------------------------
# Setup
# -----------------------------
properties = list(data_incl.keys())
num_vars = len(properties)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

colors = {"Low": "#1f77b4", "Medium": "#ff7f0e", "High": "#2ca02c"}
rings = {"Low": 1, "Medium": 2, "High": 3}

# -----------------------------
# Plot inclusion chart
# -----------------------------
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))

# Draw concentric reference circles
for r in rings.values():
    ax.plot(np.linspace(0,2*np.pi,200), [r]*200, linestyle="dotted", color="gray", alpha=0.5)
ax.set_ylim(0,3.5)

# Fill regions where inclusion=True
for level, r in rings.items():
    for i, prop in enumerate(properties):
        if data_incl[prop][r-1]:  # True = fill
            ax.bar(angles[i], 0.8, width=2*np.pi/num_vars, bottom=r-0.4,
                   color=colors[level], alpha=0.7, edgecolor="k", linewidth=0.3,
                   label=level if i==0 else "")

# Axis labels
ax.set_xticks(angles)
ax.set_xticklabels(properties, fontsize=6)
ax.set_yticks([1,2,3])
ax.set_yticklabels(["Low","Medium","High"], fontsize=9)
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

plt.tight_layout()
plt.savefig("abstraction_inclusion_exclusion.pdf", dpi=300, bbox_inches="tight")
