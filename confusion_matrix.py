import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import wandb

CELL_LABEL_X = "actual_X"
CELL_LABEL_Y = "actual_Y"
CELL_PREDICTED_LABEL = "predicted_Y"
FONT_FAMILY = "MANGAL.TTF"
# Define the actual and predicted words
table_cells =  [CELL_LABEL_X, CELL_LABEL_Y,CELL_PREDICTED_LABEL]
data = data = pd.read_csv('predictions_attention.csv',sep=",",names=table_cells)
pred_letters,actual_letters = data[CELL_LABEL_Y].tolist(),data[CELL_LABEL_X].tolist()

# padding words
for i in range(len(actual_letters)):
    actual_n = len(actual_letters[i])
    predicted_n = len(pred_letters[i])
    max_len = max(actual_n,predicted_n)
    actual_letters[i] = actual_letters[i]+' ' * (max_len - actual_n)
    pred_letters[i] = pred_letters[i]+' ' * (max_len - predicted_n)

# Create the confusion matrix character-wise
predicted_alpha = list([char for word_i in pred_letters for char in word_i])
actual_alpha = list([char for word_i in actual_letters for char in word_i])

# Pad the lists if they have different lengths
actual_n = len(actual_alpha)
predict_n = len(predicted_alpha)
max_length = max(actual_n, predict_n)
actual_alpha = [''] * (max_length - actual_n)
predicted_alpha += [''] * (max_length - predict_n)

# Get unique characters as labels
unique_chars = sorted(set(actual_alpha + predicted_alpha))


# Initialize confusion matrix
confusion_mat = [[0] * len(unique_chars) for _ in range(len(unique_chars))]

# Update confusion matrix
for actual, predicted in zip(actual_alpha, predicted_alpha):
    # print(actual_idx,predicted_idx)
    confusion_mat[unique_chars.index(actual)][unique_chars.index(predicted)] += 1

# Set a font that supports Devanagari characters
font_fma = fm.FontProperties(fname=FONT_FAMILY)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(12, 10))
ax = sns.heatmap(confusion_mat, annot=False, cmap='viridis', cbar=False, square=True)

tick_locations = list(range(len(unique_chars)))
ax.set_xticks(tick_locations)
ax.set_xticklabels(unique_chars, rotation=90, fontproperties=font_fma, fontsize=8)

ax.set_yticks(tick_locations)
ax.set_yticklabels(unique_chars, rotation=0, fontproperties=font_fma, fontsize=8)

plt.title('Char-wise Confusion Matrix', fontsize=14)
plt.xlabel('Predi Character', fontsize=12)
plt.ylabel('Actual Character', fontsize=12)

wandb.init(project="dl-assignment-3", entity="cs23m007",name="Confusion Matrix Attention")
# Get the wandb log directory
log_dir = wandb.run.dir

# Save the figure to the log directory
figure_path = f"{log_dir}/confusion_matrix.png"
plt.savefig(figure_path)

# Log the figure in wandb
wandb.log({"Confusion matrix": wandb.Image(figure_path)})
wandb.finish()

# Display the figure
plt.show()
