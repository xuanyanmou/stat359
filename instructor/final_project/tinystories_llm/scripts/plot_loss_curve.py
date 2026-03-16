import matplotlib.pyplot as plt

epochs = range(1, 16)

# cautious saver losses
train_loss_saver = [
4.3929, 2.8611, 1.1690, 0.3966, 0.2350,
0.1893, 0.1726, 0.1648, 0.1602, 0.1572,
0.1550, 0.1533, 0.1535, 0.1514, 0.1512
]

val_loss_saver = [
3.4123, 1.6407, 0.4495, 0.2070, 0.1573,
0.1458, 0.1409, 0.1383, 0.1365, 0.1354,
0.1348, 0.1332, 0.1335, 0.1330, 0.1327
]

# bold gambler losses
train_loss_gambler = [
4.3301, 2.8345, 1.1445, 0.3990, 0.2304,
0.1853, 0.1704, 0.1621, 0.1567, 0.1549,
0.1516, 0.1505, 0.1503, 0.1483, 0.1475
]

val_loss_gambler = [
3.3644, 1.6208, 0.4668, 0.2066, 0.1560,
0.1462, 0.1401, 0.1381, 0.1357, 0.1348,
0.1346, 0.1340, 0.1332, 0.1332, 0.1330
]


plt.figure(figsize=(8,5))

plt.plot(epochs, train_loss_saver, 'o-', label="Train loss")
plt.plot(epochs, val_loss_saver, 'o-', label="Val loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Cautious Saver: Train vs Val Loss")
plt.legend()

plt.savefig("cautious_saver_loss.png")
plt.close()


plt.figure(figsize=(8,5))

plt.plot(epochs, train_loss_gambler, 'o-', label="Train loss")
plt.plot(epochs, val_loss_gambler, 'o-', label="Val loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Bold Gambler: Train vs Val Loss")
plt.legend()

plt.savefig("bold_gambler_loss.png")
plt.close()