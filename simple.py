import torch
from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D, Trainer
from torch.utils.data import Dataset

class CopyDataset(Dataset):
    """Simple task: copy input sequence to output"""
    def __init__(self, size=50, seq_len=5, num_tokens=5):
        self.size = size
        self.seq_len = seq_len
        self.num_tokens = num_tokens
        # Pre-generate all data
        self.data = [torch.randint(0, num_tokens, (seq_len,)) for _ in range(size)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        return sequence, sequence  # Input = Output (copy task)

print("="*60)
print("SIMPLE COPY TASK")
print("="*60)
print("Task: Copy a sequence of 5 tokens")
print("Vocabulary: {0, 1, 2, 3, 4}")
print("="*60)

# Create model with appropriate size
trm = TinyRecursiveModel(
    dim=128,
    num_tokens=5,
    network=MLPMixer1D(dim=128, depth=2, seq_len=5),
)

# Create datasets
train_data = CopyDataset(size=50, seq_len=5, num_tokens=5)
test_data = CopyDataset(size=20, seq_len=5, num_tokens=5)

print(f"\nTraining examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")

# Train with corrected hyperparameters
print("\n" + "="*60)
print("TRAINING")
print("="*60)

trainer = Trainer(
    trm,
    train_data,
    learning_rate=0.01,        # Higher LR (gets divided by batch_size * steps)
    epochs=50,                  # 50 epochs should be enough
    batch_size=8,              # Smaller batch size
    max_recurrent_steps=6,     # Fewer recurrent steps
    halt_prob_thres=0.5,       # Halting threshold
    cpu=True                   # Set to False if you have GPU
)

trainer()

# Evaluate with per-token accuracy
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

perfect_sequences = 0
total_tokens_correct = 0
total_tokens = 0

trm.eval()  # Put model in evaluation mode
with torch.no_grad():  # No gradients needed for evaluation
    
    # First, show detailed results for first 5 examples
    print("\nDetailed Results (First 5 Examples):")
    print("-" * 60)
    
    for i in range(min(5, len(test_data))):
        inp, out = test_data[i]
        pred, exit_idx = trm.predict(
            inp.unsqueeze(0),
            max_deep_refinement_steps=12,
            halt_prob_thres=0.5
        )
        
        pred_seq = pred.squeeze()
        
        # Count correct tokens
        correct_tokens = (pred_seq == out).sum().item()
        is_perfect = torch.equal(pred_seq, out)
        
        print(f"\nExample {i+1}:")
        print(f"  Input:     {inp.tolist()}")
        print(f"  Expected:  {out.tolist()}")
        print(f"  Predicted: {pred_seq.tolist()}")
        print(f"  Correct tokens: {correct_tokens}/{len(out)} ({100*correct_tokens/len(out):.1f}%)")
        print(f"  Perfect match: {'✓ YES' if is_perfect else '✗ NO'}")
        print(f"  Exit step: {exit_idx.item()}")
    
    # Now calculate overall statistics for all test data
    print("\n" + "-" * 60)
    print("Computing overall statistics...")
    print("-" * 60)
    
    for inp, out in test_data:
        pred, exit_idx = trm.predict(
            inp.unsqueeze(0),
            max_deep_refinement_steps=12,
            halt_prob_thres=0.5
        )
        
        pred_seq = pred.squeeze()
        
        # Count per-token accuracy
        correct_tokens = (pred_seq == out).sum().item()
        total_tokens_correct += correct_tokens
        total_tokens += len(out)
        
        # Count perfect sequences
        if torch.equal(pred_seq, out):
            perfect_sequences += 1

# Calculate accuracies
sequence_accuracy = 100 * perfect_sequences / len(test_data)
token_accuracy = 100 * total_tokens_correct / total_tokens

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Per-Token Accuracy:     {token_accuracy:.1f}%")
print(f"Perfect Sequence Accuracy: {sequence_accuracy:.1f}%")
print(f"")
print(f"Correct tokens:         {total_tokens_correct}/{total_tokens}")
print(f"Perfect sequences:      {perfect_sequences}/{len(test_data)}")
print("="*60)

# Success criteria
print("\n" + "="*60)
print("SUCCESS CRITERIA")
print("="*60)
if sequence_accuracy >= 90:
    print("✓ EXCELLENT: Model learned to copy perfectly!")
elif token_accuracy >= 90:
    print("✓ GOOD: Model learned most tokens (need more training for perfection)")
elif token_accuracy >= 70:
    print("⚠ DECENT: Model is learning but needs more training")
elif token_accuracy >= 40:
    print("⚠ POOR: Model is learning slowly - try simpler task or more epochs")
else:
    print("✗ FAILED: Model not learning - check configuration")

print("\nTarget: 90%+ token accuracy, 80%+ sequence accuracy")
print("="*60)dir
