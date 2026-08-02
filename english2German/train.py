import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import math
import time
from pathlib import Path


class TransformerTrainer:
    """
    class for training the English to German Transformer
    Setup for CrossEntropyLoss
    Utilizes "Attention is All You Need" lr scheduler
    """

    model: nn.Module
    train_loader: DataLoader
    validation_loader: DataLoader
    optimizer: optim.Optimizer
    criterion: nn.Module
    pad_id: int
    warmup_steps: int
    best_validation_loss: float

    # future to do: have a custom lr choice. base lr w/ warmup/day
    def __init__(
        self, 
        model, 
        train_loader, 
        validation_loader, 
        pad_id, 
        step_count: int=0,
        warmup_steps: int=1000
    ) -> None:
        """
        Training Module for Transformer Constructor

        Args:
            model:          <model transformer>
            train_loader:   <torch.utils.data.DataLoader>
            lr:             <float>
            warmup_steps:   <int>
        """
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.warmup_steps = warmup_steps
        self.step_count = step_count
        self.pad_id = pad_id

        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.0, 
            betas=(0.9, 0.98), 
            eps=1e-9
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id
        )

        self.best_validation_loss = float("inf")


    def get_next_lr(self) -> float:
        """
        Learning Rate Schedule w/ Warmup then decay

        Return:
            <float>
        """
        d_model = self.model.d_model
        step_count = self.step_count + 1

        warmup_lr_increase = step_count * (self.warmup_steps**(-1.5))
        regular_lr_decay = step_count**(-0.5)

        return d_model**(-0.5) * min(warmup_lr_increase, regular_lr_decay)


    def update_lr(self) -> None:
        """Update Learning Rate based on schedule"""
        lr_scaled = self.get_next_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_scaled


    def evaluate_epoch(
        self,
        src_pad_id: int,
        tgt_pad_id: int
    ) -> float:
        """
        Evaluate the model over the validation set.

        Returns:
            Average validation loss per batch.
        """

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.inference_mode():
            for batch in self.validation_loader:

                bos_src_eos_batch = batch['bos_src_eos']
                bos_tgt_batch = batch['bos_tgt']
                tgt_eos_batch = batch['tgt_eos']
                bos_tgt_eos_batch = batch['bos_tgt_eos']
                english_text_batch = batch['en']
                german_text_batch = batch['de']

                src_padding_mask = self.model.create_padding_mask(bos_src_eos_batch, src_pad_id)
                tgt_padding_mask = self.model.create_padding_mask(bos_tgt_batch, tgt_pad_id)
                tgt_causal_mask = self.model.create_causal_mask(bos_tgt_batch.size(1))
                tgt_mask = tgt_causal_mask & tgt_padding_mask

                output = self.model(bos_src_eos_batch, bos_tgt_batch, src_padding_mask, tgt_mask)

                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_eos_batch.reshape(-1))
                total_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            raise ValueError("Validation loader contained no batches.")

        return total_loss / num_batches


    def train_epoch(
        self, 
        epoch: int,
        src_pad_id: int,
        tgt_pad_id: int
    ) -> tuple[float, float]:
        """
        Train for one Epoch
        
        Return:
            <float>
        """
        start = time.perf_counter()

        self.model.train()
        total_loss = 0
        num_batches = 0

        for iBatch, batch in enumerate(self.train_loader):
            self.step_count+= 1
            if self.warmup_steps > 0:
                self.update_lr()

            # Parse batch
            bos_src_eos_batch = batch['bos_src_eos']
            bos_tgt_batch = batch['bos_tgt']
            tgt_eos_batch = batch['tgt_eos']
            bos_tgt_eos_batch = batch['bos_tgt_eos']
            english_text_batch = batch['en']
            german_text_batch = batch['de']

            # Create masks (part of training / data prep technique. basically a techinique that helps to optimize result from training)
            src_padding_mask = self.model.create_padding_mask(bos_src_eos_batch, src_pad_id)
            tgt_padding_mask = self.model.create_padding_mask(bos_tgt_batch, tgt_pad_id)
            tgt_causal_mask = self.model.create_causal_mask(bos_tgt_batch.size(1))
            # Combine masks: both must be True for attention to be allowed
            # Broadcasting can handle shape diff
            tgt_mask = tgt_causal_mask & tgt_padding_mask

            # Forward w/ Teacher Forcing
            self.optimizer.zero_grad()
            output = self.model(bos_src_eos_batch, bos_tgt_batch, src_padding_mask, tgt_mask)

            # Gradient
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_eos_batch.reshape(-1))

            # Gradient Descent / update parameters
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Step {self.step_count}, Loss: {avg_loss:.4f}, lr: {lr:.6f}")



        training_loss = total_loss / num_batches

        # Run validation only after the training portion is complete.
        validation_loss = self.evaluate_epoch(src_pad_id, tgt_pad_id)
        elapsed = time.perf_counter() - start
        seconds_per_batch = elapsed / num_batches

        print(
            f"Epoch {epoch} | "
            f"Training loss: {training_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}"
        )

        print(f"Seconds per batch: {seconds_per_batch:.3f}")
        print(f"Actual epoch time: {elapsed / 60:.1f} minutes")


        checkpoint_path = Path("english2German/checkpoints/english2German_transformer.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "source_tokenizer_path": "english2German/checkpoints/tokenizers/english_bpe.json",
            "target_tokenizer_path": "english2German/checkpoints/tokenizers/german_bpe.json",
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        }

        torch.save(checkpoint, checkpoint_path)

         # Save a separate checkpoint when validation improves.
        if validation_loss < self.best_validation_loss:

            self.best_validation_loss = validation_loss
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved new best checkpoint with validation loss: {validation_loss:.4f}")


        return training_loss, validation_loss