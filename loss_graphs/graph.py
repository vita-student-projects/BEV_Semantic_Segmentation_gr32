import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re

plt.style.use('seaborn-darkgrid')

def parse_loss_values(file_path):
    loss_values = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        iterations = len(lines) // 200  # Number of hundred-iteration groups
        for i in range(iterations):
            start = i * 200
            end = (i + 1) * 200
            group_lines = lines[start:end]
            loss_sum = 0
            count = 0
            for line in group_lines:
                match = re.search(r'\ssem_loss=([\d.]+)', line)
                if match:
                    loss_sum += float(match.group(1))
                    count += 1
            if count > 0:
                mean_loss = loss_sum / count
                loss_values.append(mean_loss)
    return loss_values

def plot_loss_graph(loss_values, label):
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, label=label)

file_path1 = 'train_without_instance.txt'
file_path2 = 'train_with_instance.txt'

loss_values1 = parse_loss_values(file_path1)
loss_values2 = parse_loss_values(file_path2)

# Set y-axis to logarithmic scale
plt.yscale('log')

plot_loss_graph(loss_values1, label='Train without instance')
plot_loss_graph(loss_values2, label='Train')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch', fontsize=16)

# Change format of y-axis label
# Change format of y-axis label to display only integers
def formatter(x, pos):
    return "{:.0f}".format(x)

# Change format of y-axis label
def divide_by_10x(x, pos):
    return "{:.0f}".format((x + 10) / 10)

formattery = ticker.FuncFormatter(formatter)
formatterx = ticker.FuncFormatter(divide_by_10x)

# Set the y-axis formatter to display integers divided by 10
plt.gca().yaxis.set_major_formatter(formattery)
plt.gca().yaxis.set_minor_formatter(formattery)
plt.gca().xaxis.set_major_formatter(formatterx)
plt.gca().xaxis.set_minor_formatter(formatterx)

plt.legend()
plt.show()
