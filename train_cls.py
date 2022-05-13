import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models import LSTMClsModel
from models import RNNClsModel

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


def train_rnn_cls(args, vocab, train_dataloader, valid_dataloader, weight_matrix=None, model_type="rnn"):
    if model_type == "rnn":
        model = RNNClsModel(weight_matrix=weight_matrix,
                            vocab_size=len(vocab),
                            embed_dim=args["embed_dim"],
                            hidden_dim=args["hidden_dim"])
    elif model_type == "lstm":
        model = LSTMClsModel(weight_matrix=weight_matrix,
                             vocab_size=len(vocab),
                             embed_size=args["embed_dim"],
                             lstm_size=args["hidden_dim"],
                             dense_size=args["dense_dim"],  # optional: more linear layers
                             output_size=args["output_dim"],
                             lstm_layers=args["lstm"],  # number of layers
                             dropout=args["dropout"])

    print(args)
    print(model)
    pad_idx = vocab.index_of("<pad>")

    # ------------------
    # 1. Define loss_func (Binary cross entropy loss)
    # 2. Define optimizer (Recommend to use Adam)

    loss_fn = nn.BCELoss(reduction='sum')  # Cross Entropy(BCE) loss: pred is prob [0,1], target is binary 0/1
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])

    # ------------------

    loss_list = []
    acc_list = []
    for epoch in range(args["epochs"]):
        total_loss = 0.

        # training part
        all_preds = []
        all_targets = []
        model.train()
        for _, (sent, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch:02d}", leave=False):

            # ------------------
            # The shape of sent is [batch_size, fix_length_of_sequence]
            # The shape of targets is [batch_size, 1]
            # The procedure of this part:
            # 1. Forward
            # 2. Compute loss
            # 3. Zero gradients
            # 4. Backward
            # 5. Update network parameters

            fit = model(sent)
            loss = loss_fn(fit, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ------------------
            preds = torch.round(fit)
            all_preds.append(preds.cpu().data.numpy())
            all_targets.append(targets.cpu().data.numpy())

            total_loss += loss.item()

        train_preds = np.vstack(all_preds)
        train_targets = np.vstack(all_targets)
        train_acc = np.mean(train_preds.squeeze() == train_targets.squeeze())  # train accuracy

        # validation part
        all_preds = []
        all_targets = []
        model.eval()
        with torch.no_grad():
            for _, (val_sent, val_y) in enumerate(valid_dataloader):
                preds = torch.round(model(val_sent))  # binary prediction: >0.5 -> 1, <0.5 -> 0
                all_preds.append(preds.cpu().data.numpy())
                all_targets.append(val_y.cpu().data.numpy())

        val_preds = np.vstack(all_preds)
        val_targets = np.vstack(all_targets)
        val_acc = np.mean(val_preds.squeeze() == val_targets.squeeze())  # valid accuracy score

        avg_loss = total_loss / len(train_dataloader)
        loss_list.append(avg_loss)
        acc_list.append(val_acc)
        print(
            f"Epoch: {epoch:02d}\tTrain Loss: {avg_loss:.4f}\tTrain acc: {train_acc:.4f}\t"
            f"Val acc: {val_acc:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    ax1.plot(loss_list)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax2.plot(acc_list)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("valid Accuracy")
    plt.savefig("rnn_cls.jpg")
    plt.show()

    return model


def evaluate_your_model(model, vocab, valid_dataloader):
    pad_idx = vocab.index_of("<pad>")

    model.eval()
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for _, (val_sent, val_y) in enumerate(valid_dataloader):
            preds = torch.round(model(val_sent))
            all_preds.append(preds.cpu().data.numpy())
            all_targets.append(val_y.cpu().data.numpy())

    val_preds = np.vstack(all_preds)
    val_targets = np.vstack(all_targets)

    val_acc = np.mean(val_preds.squeeze() == val_targets.squeeze())

    print("-"*40)
    print(f"Valid Accuracy: {val_acc:.4f}")
    print("-"*40)

    return val_preds, val_targets
