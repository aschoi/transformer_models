import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import math
import time
from pathlib import Path
import re
import sacrebleu


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
        self.best_bleu_score = float('inf')


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

    def greedy_decode_batch(
        self,
        # model: nn.Module,
        source: torch.Tensor,
        src_pad_id: int,
        tgt_pad_id: int,
        tgt_bos_id: int,
        tgt_eos_id: int,
        max_output_length: int = 100
    ) -> torch.Tensor:
        """
        Translate a batch of source sequences using greedy decoding.

        Args:
            model:      Encoder-decoder Transformer.
            source:     Source token IDs shaped: [batch_size, source_length]
            src_pad_id: Padding-token ID used by the source tokenizer
            tgt_pad_id: Padding-token ID used by the target tokenizer.
            tgt_bos_id: Beginning-of-sequence ID used by the target tokenizer.
            tgt_eos_id: End-of-sequence ID used by the target tokenizer.
            max_output_length:Maximum generated sequence length, including BOS.

        Returns:
            Generated target IDs shaped: [batch_size, generated_length]
        """
        device = source.device

        if max_output_length < 2:
            raise ValueError("max_output_length must be at least 2 to allow BOS and one generated token.")

        self.model.eval()

        batch_size = source.size(0)

        # Every target sequence begins with the target-language BOS token.
        generated = torch.full(size=(batch_size, 1), fill_value=tgt_bos_id, dtype=torch.long, device=device)

        # Tracks which sequences have already generated EOS.
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # The source does not change during decoding, so create this once.
        src_mask = self.model.create_padding_mask(source, src_pad_id)

        with torch.inference_mode():
            # BOS already occupies one position.

            # The REPETITIVE re-feeding of the sequential generation of the model's predictions
            # The generative output gets built, one token at a time. 
            for _ in range(max_output_length - 1):

                # sending generated in as causal_mask parameter merely to use its size. 
                # sending genereated so that when mask is created, it creates a tensor specific to the
                # device being used, instead of having to pass in that info as an argument explicitly
                tgt_causal_mask = self.model.create_causal_mask(generated)
                tgt_padding_mask = self.model.create_padding_mask(generated, tgt_pad_id)

                # Both masks use:
                # True  = attention allowed
                # False = attention blocked
                tgt_mask = tgt_causal_mask & tgt_padding_mask
                logits = self.model(source, generated, src_mask, tgt_mask)

                # Expected logits shape:
                # [batch_size, target_length, target_vocab_size]
                if logits.ndim != 3:
                    raise RuntimeError(f"Expected model output with 3 dimensions [batch, target_length, vocabulary_size], but received shape {tuple(logits.shape)}.")

                # Only use the prediction at the newest target position.
                next_token_logits = logits[:, -1, :]

                # Greedy decoding chooses the highest-logit token.
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                # Sequences that previously produced EOS receive PAD from
                # this point onward while the other sequences continue.
                next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, tgt_pad_id), next_token)

                generated = torch.cat([generated, next_token], dim=1)

                # Mark newly completed sequences.
                finished |= next_token.squeeze(1).eq(tgt_eos_id)

                if finished.all():
                    break

        return generated

    def clean_decoded_text(self, text: str) -> str:
        # Remove spaces before common punctuation.
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        # Normalize repeated whitespace.
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def evaluate_epoch(
        self,
        src_pad_id: int,
        tgt_pad_id: int,
        device: torch.device,

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

                bos_src_eos_batch = batch['bos_src_eos'].to(device, non_blocking=True)
                bos_tgt_batch = batch['bos_tgt'].to(device, non_blocking=True)
                tgt_eos_batch = batch['tgt_eos'].to(device, non_blocking=True)
                # bos_tgt_eos_batch = batch['bos_tgt_eos'].to(device, non_blocking=True)
                english_text_batch = batch['en']
                german_text_batch = batch['de']

                src_padding_mask = self.model.create_padding_mask(bos_src_eos_batch, src_pad_id)
                tgt_padding_mask = self.model.create_padding_mask(bos_tgt_batch, tgt_pad_id)
                tgt_causal_mask = self.model.create_causal_mask(bos_tgt_batch)
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
        tgt_pad_id: int,
        tgt_bos_id: int,
        tgt_eos_id: int,
        src_tokenizer,
        tgt_tokenizer,
        device: torch.device,
        all_model_paramters: dict,
        valid_patience_count: int
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

        epoch_info = {
            "epoch": epoch,
            "training_loss": float,
            "validation_loss": float,
            "batch_info": []
        }

        for iBatch, batch in enumerate(self.train_loader):
            self.step_count+= 1
            if self.warmup_steps > 0:
                self.update_lr()

            # Parse batch
            bos_src_eos_batch = batch['bos_src_eos'].to(device, non_blocking=True)
            bos_tgt_batch = batch['bos_tgt'].to(device, non_blocking=True)
            tgt_eos_batch = batch['tgt_eos'].to(device, non_blocking=True)
            # bos_tgt_eos_batch = batch['bos_tgt_eos'].to(device, non_blocking=True)
            english_text_batch = batch['en']
            german_text_batch = batch['de']

            # Create masks (part of training / data prep technique. basically a techinique that helps to optimize result from training)
            src_padding_mask = self.model.create_padding_mask(bos_src_eos_batch, src_pad_id)
            tgt_padding_mask = self.model.create_padding_mask(bos_tgt_batch, tgt_pad_id)
            tgt_causal_mask = self.model.create_causal_mask(bos_tgt_batch)
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
                epoch_info["batch_info"].append(f"Step {self.step_count}, Loss: {avg_loss:.4f}, lr: {lr:.6f}")


        training_loss = total_loss / num_batches

        # Run validation only after the training portion is complete.
        validation_loss = self.evaluate_epoch(src_pad_id, tgt_pad_id, device)


        # predictions = []
        # references = []
        # gref = []
        
        # self.model.eval()
        # with torch.inference_mode():
    
        #     for batch in self.validation_loader:
        #         # Parse Batch Data
        #         bos_src_eos_batch = batch['bos_src_eos'].to(device, non_blocking=True)
        #         bos_tgt_batch = batch['bos_tgt'].to(device, non_blocking=True)
        #         tgt_eos_batch = batch['tgt_eos'].to(device, non_blocking=True)
        #         bos_tgt_eos_batch = batch['bos_tgt_eos'].to(device, non_blocking=True)
        #         english_text_batch = batch['en']
        #         german_text_batch = batch['de']
    
        #         src_padding_mask = self.model.create_padding_mask(bos_src_eos_batch, src_pad_id)
        #         tgt_padding_mask = self.model.create_padding_mask(bos_tgt_batch, tgt_pad_id)
        #         tgt_causal_mask = self.model.create_causal_mask(bos_tgt_batch)
    
        #         tgt_mask = tgt_causal_mask & tgt_padding_mask
    
        #         # output = model(bos_src_eos_batch, bos_tgt_batch, src_padding_mask, tgt_mask)
        #         # loss = criterion(output.reshape(-1, output.size(-1)), tgt_eos_batch.reshape(-1))
        #         # total_loss += loss.item()
        #         # num_batches += 1
    
        #         generated = self.greedy_decode_batch(
        #             source=bos_src_eos_batch,
        #             src_pad_id=src_pad_id,
        #             tgt_pad_id=tgt_pad_id,
        #             tgt_bos_id=tgt_eos_id,
        #             tgt_eos_id=tgt_eos_id,
        #             max_output_length=100
        #         )
    
        #         i = 0
        #         for predicted_ids, reference_ids in zip(generated.tolist(), bos_tgt_eos_batch.tolist()):
    
        #             predicted_text = src_tokenizer.decode(predicted_ids, skip_special_tokens=True)
        #             predicted_text = self.clean_decoded_text(predicted_text)
    
        #             reference_text = tgt_tokenizer.decode(reference_ids, skip_special_tokens=True)
        #             reference_text = self.clean_decoded_text(reference_text)
    
        #             predictions.append(predicted_text)
        #             references.append(reference_text)
        #             gref.append(german_text_batch[i])
        #             i += 1
    
    
        #         print("Generated IDs:", generated[-1].tolist())
        #         print("Translation:", predictions[-1])
        #         print(f"German Text: {gref[-1]}")
        #         print()

        # bleu_result = sacrebleu.corpus_bleu(predictions, [references])
        # bleu_result2 = sacrebleu.corpus_bleu(predictions, [gref])
        # print(f'\nbleu:  {bleu_result.score}')
        # print(f'bleu2:  {bleu_result2.score}\n')

        elapsed = time.perf_counter() - start
        seconds_per_batch = elapsed / num_batches

        print(
            f"Epoch {epoch} | "
            f"Training loss: {training_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}"
        )

        print(f"Seconds per batch: {seconds_per_batch:.3f}")
        print(f"Actual epoch time: {elapsed / 60:.1f} minutes")

        epoch_info["training_loss"] = training_loss
        epoch_info["validation_loss"] = validation_loss
        all_model_paramters['history'].append(epoch_info)

        
        # torch.save(checkpoint, checkpoint_path)
        
        # Save a separate checkpoint when validation improves.

        if validation_loss < self.best_validation_loss:

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
            
            self.best_validation_loss = validation_loss
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved new best checkpoint with validation loss: {validation_loss:.4f}")
            valid_patience_count = 0

        else:
            valid_patience_count += 1

        # if bleu_result2.score > self.best_bleu_score:

        #     checkpoint_path = Path("english2German/checkpoints/english2German_transformer_bestBleu.pt")
        #     checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        #     checkpoint = {
        #         "epoch": epoch,
        #         "step_count": self.step_count,
        #         "warmup_steps": self.warmup_steps,
        #         "model_state_dict": self.model.state_dict(),
        #         "optimizer_state_dict": self.optimizer.state_dict(),
        #         "source_tokenizer_path": "english2German/checkpoints/tokenizers/english_bpe.json",
        #         "target_tokenizer_path": "english2German/checkpoints/tokenizers/german_bpe.json",
        #         "training_loss": training_loss,
        #         "validation_loss": validation_loss,
        #         "bleu": bleu_result2.score
        #     }
            
        #     self.best_bleu_score = bleu_result2.score
        #     torch.save(checkpoint, checkpoint_path)
        #     print(f"Saved new best checkpoint with bleu score: {bleu_result2.score}")
        #     bleu_patience_count = 0
        # else:
        #     bleu_patience_count += 1


        return training_loss, validation_loss, valid_patience_count