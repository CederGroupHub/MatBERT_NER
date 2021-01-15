import os
import json
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, AutoConfig, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils.metrics import accuracy
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod

class NERModel(ABC):
    """
    A wrapper class for transformers models, implementing train, predict, and evaluate methods
    """

    def __init__(self, model="allenai/scibert_scivocab_cased", classes = ["O"], device="cpu", lr=5e-5):
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.classes = classes
        self.config = AutoConfig.from_pretrained(model)
        self.config.num_labels = len(self.classes)
        self.lr = lr
        self.device = device
        self.model = self.initialize_model()


    def train(self, train_dataloader, n_epochs, val_dataloader=None, save_dir=None):
        """
        Train the model
        Inputs:
            dataloader :: dataloader with training data
            n_epochs :: number of epochs to train
            val_dataloader :: dataloader with validation data - if provided the model with the best performance on the validation set will be saved
            save_dir :: directory to save models
        """

        val_loss_best = 1e10


        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer, n_epochs, train_dataloader)

        for epoch in range(n_epochs):
            print("\n\n\nEpoch: " + str(epoch + 1))
            self.model.train()

            for i, batch in enumerate(tqdm(train_dataloader)):
                
                inputs = {"input_ids": batch[0].to(self.device, non_blocking=True),
                          "attention_mask": batch[1].to(self.device, non_blocking=True),
                          "valid_mask": batch[2].to(self.device, non_blocking=True),
                          "labels": batch[4].to(self.device, non_blocking=True)}

                optimizer.zero_grad()
                loss, predicted = self.model.forward(**inputs)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if i%100 == 0:
                    labels = inputs['labels']

                    acc = accuracy(predicted, labels)

                    print("loss: {}, acc: {}".format(torch.mean(loss).item(),acc.item()))

            save_path = os.path.join(save_dir, "epoch_{}.pt".format(epoch))

            torch.save(self.model.state_dict(), save_path)

            if val_dataloader is not None:
                self.model.eval()
                val_loss = []
                val_pred = []
                val_label = []
                with torch.no_grad():
                    for batch in val_dataloader:

                        inputs = {"input_ids": batch[0].to(self.device),
                                  "attention_mask": batch[1].to(self.device),
                                  "valid_mask": batch[2].to(self.device),
                                  "labels": batch[4].to(self.device)}

                        loss, pred = self.model.forward(**inputs)
                        val_loss.append(loss)
                        val_pred.append(pred)
                        val_label.append(inputs['labels'])
                    val_loss = torch.stack(val_loss)
                    val_pred = torch.cat(val_pred, dim=0)
                    val_label = torch.cat(val_label, dim=0)
                    val_acc = accuracy(val_pred, val_label)

                if torch.mean(val_loss).item() < val_loss_best:
                    save_path = os.path.join(save_dir, "best.pt".format(epoch))

                    torch.save(self.model.state_dict(), save_path)

                print("val loss: {}, val acc: {}".format(torch.mean(val_loss).item(),val_acc.item()))

        if val_dataloader is not None:
            # Restore weights of best model after training if we can

            save_path = os.path.join(save_dir, "best.pt".format(epoch))
            self.model.load_state_dict(torch.load(save_path))


    def predict(self,dataloader):
        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in dataloader:

                inputs = {"input_ids": batch[0].to(device),
                          "attention_mask": batch[1].to(device),
                          "valid_mask": batch[2].to(device)}

                _, pred = ner_model.forward(**inputs)
                preds += pred

        return preds

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def create_optimizer(self):
        pass

    @abstractmethod
    def create_scheduler(self, optimizer, n_epochs, train_dataloader):
        pass
