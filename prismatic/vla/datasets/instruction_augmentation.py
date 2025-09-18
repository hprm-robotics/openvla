"""
instruction_augmentation.py

Utilities for augmenting language instructions during training using pre-computed instruction mappings.
"""

import json
import random
import re
from typing import Any, Dict, List, Optional

from torch.utils.data import IterableDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def normalize_instruction(instr):
    """
    Normalize instruction by removing trailing punctuation and converting to lowercase.
    Returns None if instruction is empty after normalization.
    """
    if not instr or not isinstance(instr, str):
        return None
    
    instr = instr.strip().lower()
    # Remove any trailing punctuation (., !, ?)
    instr = re.sub(r'[.?!]+$', '', instr).strip()
    
    # Return None if instruction is empty after normalization
    if not instr:
        return None
    
    return instr


class InstructionAugmenter:
    """Handles instruction augmentation using a pre-computed mapping of original instructions to rephrases."""
    
    def __init__(self, instruction_mapping_file: Optional[str] = None, max_rephrases: int = 5):
        """
        Initialize the instruction augmenter.
        
        Args:
            instruction_mapping_file: Path to JSON file containing instruction mappings
            max_rephrases: Maximum number of rephrases to consider per instruction
        """
        self.instruction_mapping = {}
        self.max_rephrases = max_rephrases
        
        if instruction_mapping_file:
            self.load_instruction_mapping(instruction_mapping_file)
    
    def load_instruction_mapping(self, instruction_mapping_file: str) -> None:
        """Load instruction mapping from JSON file."""
        try:
            with open(instruction_mapping_file, 'r') as f:
                raw_mapping = json.load(f)
            
            # Convert to a more efficient lookup structure
            # The JSON has structure: {"0": {"original": "...", "rephrases": [...]}, ...}
            for entry in raw_mapping.values():
                # Normalize the original instruction
                original = normalize_instruction(entry["original"])
                if original is None:
                    continue
                    
                # Normalize and filter rephrases
                normalized_rephrases = []
                for rephrase in entry["rephrases"][:self.max_rephrases]:
                    normalized_rephrase = normalize_instruction(rephrase)
                    if normalized_rephrase is not None and normalized_rephrase != original:
                        normalized_rephrases.append(normalized_rephrase)
                
                self.instruction_mapping[original] = normalized_rephrases
                
            print(f"Loaded instruction mapping with {len(self.instruction_mapping)} unique instructions")
            
        except Exception as e:
            print(f"Warning: Could not load instruction mapping from {instruction_mapping_file}: {e}")
            self.instruction_mapping = {}
    
    def get_all_instruction_variants(self, original_instruction: str) -> List[str]:
        """
        Get all instruction variants including the original and available rephrases.
        
        Args:
            original_instruction: The original instruction text
            
        Returns:
            List of all instruction variants (original + rephrases)
        """
        if not self.instruction_mapping:
            return [original_instruction]
            
        # Normalize the instruction for lookup
        normalized_instruction = normalize_instruction(original_instruction)
        if normalized_instruction is None:
            return [original_instruction]
        
        # Start with the original instruction (normalized)
        variants = [normalized_instruction]
        
        # Add rephrases if available
        if normalized_instruction in self.instruction_mapping:
            rephrases = self.instruction_mapping[normalized_instruction]
            variants.extend(rephrases)
        
        return variants
    
    def augment_instruction(self, original_instruction: str) -> str:
        """
        Augment a single instruction by randomly selecting from available rephrases.
        This method is kept for backward compatibility.
        
        Args:
            original_instruction: The original instruction text
            
        Returns:
            Either the original instruction or a randomly selected rephrase
        """
        variants = self.get_all_instruction_variants(original_instruction)
        return random.choice(variants)


class AugmentedRLDSDataset(IterableDataset):
    """
    Wrapper around RLDSDataset that replicates episodes for each instruction variant.
    This creates multiple training samples from each episode - one for each instruction rephrase.
    """
    
    def __init__(self, base_dataset, instruction_augmenter: Optional[InstructionAugmenter] = None):
        """
        Initialize the augmented dataset.
        
        Args:
            base_dataset: The base RLDSDataset instance
            instruction_augmenter: Optional instruction augmenter for episode replication
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.instruction_augmenter = instruction_augmenter
        self.dataset_statistics = base_dataset.dataset_statistics
        
        # Statistics for logging
        self.episode_count = 0
        self.augmented_episode_count = 0
        self.total_variants = 0
        self.instructions_with_rephrases = 0
        self.log_interval = 100  # Log every 100 original episodes
        self.wandb_log_interval = 100  # Log to wandb every 500 original episodes
    
    def log_augmentation_summary(self):
        """Log a comprehensive summary of augmentation statistics."""
        if self.episode_count == 0:
            return
            
        avg_augmentation = self.total_variants / self.episode_count
        rephrase_coverage = (self.instructions_with_rephrases / self.episode_count) * 100
        
        print(f"\n" + "="*60)
        print(f"INSTRUCTION AUGMENTATION SUMMARY")
        print(f"="*60)
        print(f"Original episodes processed: {self.episode_count:,}")
        print(f"Total training samples created: {self.total_variants:,}")
        print(f"Average augmentation factor: {avg_augmentation:.2f}x")
        print(f"Instructions with rephrases: {self.instructions_with_rephrases:,} ({rephrase_coverage:.1f}%)")
        print(f"Data expansion: {((self.total_variants / self.episode_count) - 1) * 100:.1f}% more training data")
        print(f"="*60 + "\n")
        
        # Log final summary to wandb
        if WANDB_AVAILABLE:
            wandb.log({
                "augmentation_summary/total_original_episodes": self.episode_count,
                "augmentation_summary/total_training_samples": self.total_variants,
                "augmentation_summary/final_augmentation_factor": avg_augmentation,
                "augmentation_summary/final_rephrase_coverage": rephrase_coverage,
                "augmentation_summary/data_expansion_percent": ((avg_augmentation - 1) * 100)
            })
    
    def __iter__(self):
        """Iterate over the dataset with instruction augmentation."""
        for batch in self.base_dataset:
            self.episode_count += 1
            
            if self.instruction_augmenter and "task" in batch and "language_instruction" in batch["task"]:
                # Decode the original instruction
                original_instruction = batch["task"]["language_instruction"].decode()
                
                # Get all instruction variants
                instruction_variants = self.instruction_augmenter.get_all_instruction_variants(original_instruction)
                
                # Update statistics
                self.total_variants += len(instruction_variants)
                if len(instruction_variants) > 1:  # More than just the original
                    self.instructions_with_rephrases += 1
                
                # Log augmentation info periodically
                if self.episode_count % self.log_interval == 0:
                    avg_augmentation = self.total_variants / self.episode_count if self.episode_count > 0 else 0
                    rephrase_coverage = (self.instructions_with_rephrases / self.episode_count) * 100
                    print(f"[Augmentation] Episode {self.episode_count}: "
                          f"'{original_instruction[:50]}...' -> {len(instruction_variants)} variants "
                          f"(avg {avg_augmentation:.1f}x, {rephrase_coverage:.1f}% have rephrases)")
                
                # Log to wandb periodically
                if WANDB_AVAILABLE and self.episode_count % self.wandb_log_interval == 0:
                    avg_augmentation = self.total_variants / self.episode_count
                    rephrase_coverage = (self.instructions_with_rephrases / self.episode_count) * 100
                    
                    wandb.log({
                        "augmentation/episodes_processed": self.episode_count,
                        "augmentation/total_training_samples": self.total_variants,
                        "augmentation/avg_augmentation_factor": avg_augmentation,
                        "augmentation/rephrase_coverage_percent": rephrase_coverage,
                        "augmentation/instructions_with_rephrases": self.instructions_with_rephrases,
                        "augmentation/current_variants": len(instruction_variants)
                    })
                
                # Yield one batch for each instruction variant
                for i, variant in enumerate(instruction_variants):
                    # Create a copy of the batch with the variant instruction
                    augmented_batch = batch.copy()
                    augmented_batch["task"] = batch["task"].copy()
                    augmented_batch["task"]["language_instruction"] = variant.encode()
                    
                    self.augmented_episode_count += 1
                    
                    # Log first few variants for verification
                    if self.episode_count <= 5:
                        print(f"  Variant {i+1}/{len(instruction_variants)}: '{variant}'")
                    
                    yield augmented_batch
            else:
                # No augmentation, yield original batch
                self.augmented_episode_count += 1
                self.total_variants += 1  # Count original as 1 variant
                yield batch


class AugmentedRLDSBatchTransform:
    """Extended RLDSBatchTransform that works with augmented datasets."""
    
    def __init__(self, action_tokenizer, base_tokenizer, image_transform, prompt_builder_fn, 
                 predict_stop_token: bool = True):
        """
        Initialize the batch transform.
        
        Args:
            action_tokenizer: Tokenizer for actions
            base_tokenizer: Base tokenizer for text
            image_transform: Transform for images
            prompt_builder_fn: Function to build prompts
            predict_stop_token: Whether to predict stop tokens
        """
        # Import here to avoid circular imports
        from prismatic.vla.datasets import RLDSBatchTransform
        
        # Create the base transform
        self.base_transform = RLDSBatchTransform(
            action_tokenizer=action_tokenizer,
            base_tokenizer=base_tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=prompt_builder_fn,
            predict_stop_token=predict_stop_token
        )
    
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform RLDS batch (instructions are already augmented at dataset level).
        
        Args:
            rlds_batch: Raw RLDS batch with potentially augmented instruction
            
        Returns:
            Transformed batch ready for model training
        """
        # Apply the base transform (instructions are already processed by AugmentedRLDSDataset)
        return self.base_transform(rlds_batch)
