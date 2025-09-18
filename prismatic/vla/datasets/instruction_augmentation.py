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
        self.debug_call_count = 0  # Track calls for debug limiting
        
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
        self.debug_call_count += 1
        show_debug = self.debug_call_count <= 5  # Only show debug for first 5 calls
        
        if not self.instruction_mapping:
            if show_debug:
                print(f"[DEBUG] No instruction mapping available")
            return [original_instruction]
            
        # Normalize the instruction for lookup
        normalized_instruction = normalize_instruction(original_instruction)
        if normalized_instruction is None:
            if show_debug:
                print(f"[DEBUG] Instruction normalization failed for: '{original_instruction}'")
            return [original_instruction]
        
        # Start with the original instruction (normalized)
        variants = [normalized_instruction]
        
        # Add rephrases if available
        if normalized_instruction in self.instruction_mapping:
            rephrases = self.instruction_mapping[normalized_instruction]
            variants.extend(rephrases)
            if show_debug:
                print(f"[DEBUG] Found {len(rephrases)} rephrases for '{normalized_instruction}'")
        else:
            if show_debug:
                print(f"[DEBUG] No rephrases found for '{normalized_instruction}'")
        
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
    This intercepts the raw RLDS data before batch transformation.
    """
    
    def __init__(self, data_root_dir, data_mix, batch_transform, instruction_augmenter, 
                 resize_resolution, shuffle_buffer_size=256_000, train=True, image_aug=False):
        """
        Initialize the augmented RLDS dataset.
        
        Args:
            data_root_dir: Path to RLDS data
            data_mix: Dataset mixture name
            batch_transform: Batch transformation function
            instruction_augmenter: Instruction augmenter instance
            resize_resolution: Image resize resolution
            shuffle_buffer_size: Shuffle buffer size
            train: Whether training mode
            image_aug: Whether to use image augmentation
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from prismatic.vla.datasets import RLDSDataset
        
        # Create base dataset but we'll intercept its iteration
        self.base_dataset = RLDSDataset(
            data_root_dir=data_root_dir,
            data_mix=data_mix, 
            batch_transform=batch_transform,
            resize_resolution=resize_resolution,
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            image_aug=image_aug
        )
        
        self.instruction_augmenter = instruction_augmenter
        self.dataset_statistics = self.base_dataset.dataset_statistics
        
        # Statistics for logging
        self.episode_count = 0
        self.augmented_episode_count = 0
        self.total_variants = 0
        self.instructions_with_rephrases = 0
        self.skipped_empty_instructions = 0
        self.log_interval = 100  # Log every 100 original episodes
        self.wandb_log_interval = 100  # Log to wandb every 100 original episodes
    
    def log_augmentation_summary(self):
        """Log a comprehensive summary of augmentation statistics."""
        if self.episode_count == 0:
            return
            
        avg_augmentation = self.total_variants / self.episode_count
        rephrase_coverage = (self.instructions_with_rephrases / self.episode_count) * 100
        
        print(f"\n" + "="*60)
        print(f"INSTRUCTION AUGMENTATION SUMMARY")
        print(f"="*60)
        processed_episodes = self.episode_count - self.skipped_empty_instructions
        print(f"Total episodes encountered: {self.episode_count:,}")
        print(f"Episodes with empty instructions (skipped): {self.skipped_empty_instructions:,}")
        print(f"Episodes processed: {processed_episodes:,}")
        print(f"Total training samples created: {self.total_variants:,}")
        print(f"Average augmentation factor: {avg_augmentation:.2f}x")
        print(f"Instructions with rephrases: {self.instructions_with_rephrases:,} ({rephrase_coverage:.1f}%)")
        print(f"Data expansion: {((avg_augmentation) - 1) * 100:.1f}% more training data")
        print(f"="*60 + "\n")
        
        # Log final summary to wandb
        if WANDB_AVAILABLE:
            wandb.log({
                "augmentation_summary/total_episodes_encountered": self.episode_count,
                "augmentation_summary/episodes_skipped_empty": self.skipped_empty_instructions,
                "augmentation_summary/episodes_processed": processed_episodes,
                "augmentation_summary/total_training_samples": self.total_variants,
                "augmentation_summary/final_augmentation_factor": avg_augmentation,
                "augmentation_summary/final_rephrase_coverage": rephrase_coverage,
                "augmentation_summary/data_expansion_percent": ((avg_augmentation - 1) * 100)
            })
    
    def __iter__(self):
        """Iterate over raw RLDS data and apply instruction augmentation before batch transform."""
        print(f"[DEBUG] Starting augmented dataset iteration")
        
        # Iterate over raw RLDS data (before batch transform)
        for rlds_batch in self.base_dataset.dataset.as_numpy_iterator():
            self.episode_count += 1
            
            # Check if we can apply augmentation
            if (self.instruction_augmenter and 
                "task" in rlds_batch and 
                "language_instruction" in rlds_batch["task"]):
                
                # Get original instruction
                original_instruction = rlds_batch["task"]["language_instruction"].decode().strip()
                
                # Skip episodes with empty instructions
                if not original_instruction:
                    self.skipped_empty_instructions += 1
                    if self.episode_count <= 10:
                        print(f"[DEBUG] Episode {self.episode_count}: Skipping empty instruction")
                    continue
                
                # Debug output for first few batches
                if self.episode_count <= 3:
                    print(f"[DEBUG] Episode {self.episode_count}: Processing instruction: '{original_instruction}'")
                
                # Get all instruction variants
                instruction_variants = self.instruction_augmenter.get_all_instruction_variants(original_instruction)
                
                if self.episode_count <= 3:
                    print(f"[DEBUG] Episode {self.episode_count}: Generated {len(instruction_variants)} variants")
                    for i, variant in enumerate(instruction_variants):
                        print(f"[DEBUG]   Variant {i+1}: '{variant}'")
                
                # Update statistics
                self.total_variants += len(instruction_variants)
                if len(instruction_variants) > 1:
                    self.instructions_with_rephrases += 1
                
                # Log periodically
                if self.episode_count % self.log_interval == 0:
                    processed_episodes = self.episode_count - self.skipped_empty_instructions
                    avg_augmentation = self.total_variants / processed_episodes if processed_episodes > 0 else 0
                    rephrase_coverage = (self.instructions_with_rephrases / processed_episodes) * 100 if processed_episodes > 0 else 0
                    print(f"[Augmentation] Episode {self.episode_count}: "
                          f"'{original_instruction[:50]}...' -> {len(instruction_variants)} variants "
                          f"(avg {avg_augmentation:.1f}x, {rephrase_coverage:.1f}% have rephrases, "
                          f"skipped {self.skipped_empty_instructions} empty)")
                
                # Create one processed batch for each variant
                for variant in instruction_variants:
                    # Create a copy of the raw RLDS batch with variant instruction
                    augmented_rlds_batch = rlds_batch.copy()
                    augmented_rlds_batch["task"] = rlds_batch["task"].copy()
                    augmented_rlds_batch["task"]["language_instruction"] = variant.encode()
                    
                    # Apply batch transform to get processed batch
                    processed_batch = self.base_dataset.batch_transform(augmented_rlds_batch)
                    yield processed_batch
                    
                    self.augmented_episode_count += 1
            else:
                # Check if instruction exists and is not empty (for non-augmented case)
                if ("task" in rlds_batch and 
                    "language_instruction" in rlds_batch["task"]):
                    instruction = rlds_batch["task"]["language_instruction"].decode().strip()
                    if not instruction:
                        self.skipped_empty_instructions += 1
                        if self.episode_count <= 10:
                            print(f"[DEBUG] Episode {self.episode_count}: Skipping empty instruction (no augmentation)")
                        continue
                
                # No augmentation, just apply batch transform
                if self.episode_count <= 3:
                    print(f"[DEBUG] Episode {self.episode_count}: No augmentation - applying base transform")
                    print(f"[DEBUG] Episode {self.episode_count}: RLDS keys = {list(rlds_batch.keys())}")
                    if "task" in rlds_batch:
                        print(f"[DEBUG] Episode {self.episode_count}: Task keys = {list(rlds_batch['task'].keys())}")
                
                processed_batch = self.base_dataset.batch_transform(rlds_batch)
                yield processed_batch
                
                self.augmented_episode_count += 1
                self.total_variants += 1


class AugmentedRLDSBatchTransform:
    """RLDSBatchTransform that applies instruction augmentation to raw RLDS data."""
    
    def __init__(self, action_tokenizer, base_tokenizer, image_transform, prompt_builder_fn, 
                 instruction_augmenter: Optional[InstructionAugmenter] = None, predict_stop_token: bool = True):
        """
        Initialize the augmented batch transform.
        
        Args:
            action_tokenizer: Tokenizer for actions
            base_tokenizer: Base tokenizer for text
            image_transform: Transform for images
            prompt_builder_fn: Function to build prompts
            instruction_augmenter: Optional instruction augmenter
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
        self.instruction_augmenter = instruction_augmenter
        self.batch_count = 0
    
    def __call__(self, rlds_batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transform RLDS batch with instruction augmentation.
        
        Args:
            rlds_batch: Raw RLDS batch
            
        Returns:
            List of transformed batches (one for each instruction variant)
        """
        self.batch_count += 1
        
        # Check if we have instruction augmentation and the required structure
        if (self.instruction_augmenter and 
            "task" in rlds_batch and 
            "language_instruction" in rlds_batch["task"]):
            
            # Get original instruction
            original_instruction = rlds_batch["task"]["language_instruction"].decode()
            
            # Debug output for first few batches
            if self.batch_count <= 3:
                print(f"[DEBUG] Batch {self.batch_count}: Processing instruction: '{original_instruction}'")
            
            # Get all instruction variants
            instruction_variants = self.instruction_augmenter.get_all_instruction_variants(original_instruction)
            
            if self.batch_count <= 3:
                print(f"[DEBUG] Batch {self.batch_count}: Generated {len(instruction_variants)} variants")
                for i, variant in enumerate(instruction_variants):
                    print(f"[DEBUG]   Variant {i+1}: '{variant}'")
            
            # Create one batch for each variant
            augmented_batches = []
            for variant in instruction_variants:
                # Create a copy of the batch with the variant instruction
                augmented_batch = rlds_batch.copy()
                augmented_batch["task"] = rlds_batch["task"].copy()
                augmented_batch["task"]["language_instruction"] = variant.encode()
                
                # Apply base transform to get the final processed batch
                processed_batch = self.base_transform(augmented_batch)
                augmented_batches.append(processed_batch)
            
            return augmented_batches
        else:
            # No augmentation, just apply base transform
            if self.batch_count <= 3:
                print(f"[DEBUG] Batch {self.batch_count}: No augmentation applied")
                print(f"[DEBUG] Batch {self.batch_count}: Keys = {list(rlds_batch.keys())}")
                if "task" in rlds_batch:
                    print(f"[DEBUG] Batch {self.batch_count}: Task keys = {list(rlds_batch['task'].keys())}")
            
            return [self.base_transform(rlds_batch)]
