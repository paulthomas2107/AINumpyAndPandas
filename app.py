from functions import *

bias = 0.5
l_rate = 0.01
epochs = 20

data, weights = generate_data(50, 3)
train_model(data, weights, bias, l_rate, epochs)

# Plot to graph
df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind="line", grid=True).get_figure()
df_plot.savefig("AI_Graph.pdf")


