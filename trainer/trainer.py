import os
import math
import json
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from models.bert import VanillaBert
from utils.data_utils import CLIRDataset, collate_fn
from utils.preprocessor import Preprocessor
from utils.metric import NDCG

class Trainer():
    def __init__(self, config):
        self.config = config
        
        # device config
        if self.config.use_gpu:
            self.config.device = torch.device("cuda")
        else:
            self.config.device = torch.device("cpu")
        
        # logger config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("train.log", mode="w"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # prepare data loader and target documents
        train_dataset = CLIRDataset(config.train_datapath)
        val_dataset = CLIRDataset(config.val_datapath)
        test_dataset = CLIRDataset(config.test_datapath)
        
        self.logger.info(f"Training set length: {len(train_dataset)}")
        self.logger.info(f"Validation set length: {len(val_dataset)}")
        self.logger.info(f"Test set length: {len(test_dataset)}")
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 8,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )
        id2doc = {}
        doc_path = os.path.join("data", f"{config.target_language}.tsv")
        with open(doc_path, "r", encoding="utf-8") as fin:
            for line in fin.readlines():
                l = line.strip().split("\t")
                id2doc[l[0]] = l[1]
                
        self.logger.info("Data init done.")
        
        # prepare model, preprocessor, loss, optimizer and scheduler
        self.model = VanillaBert(self.config)
        self.preprocessor = Preprocessor(self.config, id2doc)
        # self.loss = HingeLoss(task="binary")
        # self.loss = nn.HingeEmbeddingLoss()
        self.loss = nn.MarginRankingLoss()
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )
        
        if config.resume_path is None:
            self.logger.warning("No checkpoint given!")
            self.best_val_loss = 10000
            self.best_f1 = 0
            self.last_epoch = -1
            num_training_steps = self.config.num_epochs * len(self.train_dataloader)
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=self.last_epoch
            )
        else:
            self.logger.info(f"Loading model from checkpoint: {self.config.resume_path}.")
            checkpoint = torch.load(self.config.resume_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.best_val_loss = checkpoint["best_val_loss"]
            self.best_f1 = 0
            self.last_epoch = -1
            num_training_steps = self.config.num_epochs * len(self.train_dataloader)
            self.optimizer.load_state_dict(checkpoint["optimizier"])
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=self.last_epoch
            )
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.model.to(self.config.device)
        
        self.logger.info("Trainer init done.")

    def train_one_epoch(self):
        total_loss = 0
        for step, batch in tqdm(enumerate(self.train_dataloader)):
            _, queries, targets = batch
            encoded_positive_inputs, encoded_negative_inputs = self.preprocessor.process(queries, targets)
            positive_logits = self.model(*encoded_positive_inputs)
            negative_logits = self.model(*encoded_negative_inputs)
            # labels = [1] * self.config.batch_size + [0] * self.config.batch_size
            # labels = torch.tensor([1] * self.config.batch_size + [0] * self.config.batch_size).float().to(self.config.device)
            # loss = self.loss(torch.cat((positive_logits, negative_logits)), labels)
            labels = torch.tensor([1] * self.config.batch_size).float().to(self.config.device)
            loss = self.loss(positive_logits, negative_logits, labels)
            total_loss += loss.detach().cpu().item()
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        return total_loss / step
          
    def train(self):
        val_loss = self.validate()
        self.logger.info(f"Before training validation loss: {val_loss}")
        for epoch in range(self.config.num_epochs):
            self.logger.info("========================")
            epoch_loss = self.train_one_epoch()
            self.logger.info(f"Epoch {epoch} training loss: {epoch_loss}")
            val_loss = self.validate()
            self.logger.info(f"Epoch {epoch} validation loss: {val_loss}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizier": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "last_epoch": self.last_epoch + epoch + 1,
                    "best_val_loss": self.best_val_loss,
                    "config": self.config,
                }
                save_path = os.path.join(self.config.save_dir, "best.pt")
                self.logger.info(f"Saving best checkpoint to {save_path}")
                torch.save(checkpoint, save_path)
    
    @torch.no_grad()
    def validate(self):
        total_loss = 0
        for step, batch in tqdm(enumerate(self.val_dataloader)):
            _, queries, targets = batch
            encoded_positive_inputs, encoded_negative_inputs = self.preprocessor.process(queries, targets)
            positive_logits = self.model(*encoded_positive_inputs)
            negative_logits = self.model(*encoded_negative_inputs)
            # labels = torch.tensor([1] * self.config.batch_size + [0] * self.config.batch_size).float().to(self.config.device)
            # loss = self.loss(torch.cat((positive_logits, negative_logits)), labels)
            labels = torch.tensor([1] * self.config.batch_size).float().to(self.config.device)
            loss = self.loss(positive_logits, negative_logits, labels)
            total_loss += loss.detach().cpu().item()
        return total_loss / step
    
    @torch.no_grad()
    def eval(self, test_dataloader=None):
        self.logger.info("Start evaluation")
        metric = NDCG()
        if test_dataloader is None:
            test_dataloader = self.test_dataloader
        
        total_ndcg1, total_ndcg5, total_ndcg10 = 0, 0, 0
        src_ids, queries, targets, preds = [], [], [], []
        for step, batch in tqdm(enumerate(test_dataloader)):
            src_id, query, target = batch
            src_id = src_id[0]
            query = query[0]
            target = target[0]
            doc_id_chuncks, tensor_chuncks = self.preprocessor.process_all(query)
            doc_ids, logits = [], []
            for doc_id_chunck, tensor_chunck in zip(doc_id_chuncks, tensor_chuncks):
                logit_chunck = self.model(*tensor_chunck)
                for i in range(logit_chunck.size(0)):
                    doc_ids.append(doc_id_chunck[i])
                    logits.append(logit_chunck[i].item())
            ranking = np.argsort(logits)
            ranked_doc_ids = [doc_ids[i] for i in ranking]
            ranked_doc_ids = ranked_doc_ids[:10]
            src_ids.append(src_id)
            queries.append(query)
            targets.append(target)
            preds.append(ranked_doc_ids)
            total_ndcg1 += metric.ndcg_at_k(ranked_doc_ids, target, 1)
            total_ndcg5 += metric.ndcg_at_k(ranked_doc_ids, target, 5)
            total_ndcg10 += metric.ndcg_at_k(ranked_doc_ids, target, 10)
        
        self.logger.info(f"NDCG@1: {total_ndcg1 / (step + 1)}")
        self.logger.info(f"NDCG@5: {total_ndcg5 / (step + 1)}")
        self.logger.info(f"NDCG@10: {total_ndcg10 / (step + 1)}")
        
    
    @torch.no_grad()
    def eval_partial(self, test_dataloader=None):
        self.logger.info("Start evaluation")
        metric = NDCG()
        if test_dataloader is None:
            test_dataloader = self.test_dataloader
        
        total_ndcg1, total_ndcg5, total_ndcg10 = 0, 0, 0
        for step, batch in tqdm(enumerate(test_dataloader)):
            src_id, query, target = batch
            src_id = src_id[0]
            query = query[0]
            target = target[0]
            tensors = self.preprocessor.process_partial(query, target)
            logits = self.model(*tensors).squeeze()    
            ranking = np.argsort(-logits.cpu()) # descending argsort
            pred_relevance_scores = [target[i][1] for i in ranking]
            total_ndcg1 += metric.ndcg_at_k(pred_relevance_scores, target, 1)
            total_ndcg5 += metric.ndcg_at_k(pred_relevance_scores, target, 5)
            total_ndcg10 += metric.ndcg_at_k(pred_relevance_scores, target, 10)
        
        self.logger.info(f"NDCG@1: {total_ndcg1 / (step + 1)}")
        self.logger.info(f"NDCG@5: {total_ndcg5 / (step + 1)}")
        self.logger.info(f"NDCG@10: {total_ndcg10 / (step + 1)}")