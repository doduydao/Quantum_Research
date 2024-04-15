import matplotlib.pyplot as plt

x = [i for i in range(0, 1000+50, 50)]
energys = [70.25, 69.72, 69.08, 68.12, 67.09, 65.88, 64.46, 63.18, 61.37, 59.47, 57.32, 55.17, 52.55, 50.18, 47.0, 43.59, 40.03, 35.66, 29.47, 16.17, 5.28 ]
plt.plot(x, energys)
# Loop through data and add text labels with a small offset
# for i, v in enumerate(energys):
#     plt.text(energys[i], v + 0.2, f"{v}", ha="center")  # ha for horizontal alignment
plt.xlabel("Iters")
plt.ylabel("Estimation of average cost")
# plt.legend()

plt.savefig("line_plot_2.png")