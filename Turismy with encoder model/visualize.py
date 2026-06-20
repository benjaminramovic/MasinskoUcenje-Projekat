import json
import matplotlib.pyplot as plt

with open("results/checkpoint-5250/trainer_state.json") as f:
    state = json.load(f)

evals = [x for x in state["log_history"] if "eval_f1_micro" in x]

epochs = [x["epoch"] for x in evals]
f1 = [x["eval_f1_micro"] for x in evals]

plt.plot(epochs, f1, marker="o")
plt.title("F1-score kroz epohe")
plt.xlabel("Epoch")
plt.ylabel("F1 Micro")
plt.show()