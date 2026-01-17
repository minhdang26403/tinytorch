import numpy as np

from tinytorch.nn import Linear, MSELoss
from tinytorch.optim import SGD
from tinytorch.optim.lr_scheduler import CosineSchedule
from tinytorch.tensor import Tensor
from tinytorch.training import Trainer


def test_unit_trainer():
    """🔬 Test Trainer implementation."""
    print("🔬 Unit Test: Trainer...")

    # Use REAL components from previous modules (already imported at module level)

    # Create a simple model using REAL Linear layer
    class SimpleModel:
        def __init__(self):
            self.layer = Linear(2, 1)  # Real Linear from Module 03
            self.training = True

        def forward(self, x):
            return self.layer.forward(x)

        def parameters(self):
            return self.layer.parameters()

    # Create trainer with REAL components
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)  # Real SGD from Module 07
    loss_fn = MSELoss()  # Real MSELoss from Module 04
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)

    trainer = Trainer(model, optimizer, loss_fn, scheduler, grad_clip_norm=1.0)

    # Test training
    print("Testing training epoch...")
    # Use real Tensors for data
    dataloader = [
        (Tensor([[1.0, 0.5]]), Tensor([[2.0]])),
        (Tensor([[0.5, 1.0]]), Tensor([[1.5]])),
    ]

    loss = trainer.train_epoch(dataloader)
    assert isinstance(loss, (float, np.floating)), (
        f"Expected float loss, got {type(loss)}"
    )
    assert trainer.epoch == 1, f"Expected epoch 1, got {trainer.epoch}"

    # Test evaluation
    print("Testing evaluation...")
    eval_loss, accuracy = trainer.evaluate(dataloader)
    assert isinstance(eval_loss, (float, np.floating)), (
        f"Expected float eval_loss, got {type(eval_loss)}"
    )
    assert isinstance(accuracy, (float, np.floating)), (
        f"Expected float accuracy, got {type(accuracy)}"
    )

    # Test checkpointing
    print("Testing checkpointing...")
    checkpoint_path = "/tmp/test_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    # Modify trainer state
    original_epoch = trainer.epoch
    trainer.epoch = 999

    # Load checkpoint
    trainer.load_checkpoint(checkpoint_path)
    assert trainer.epoch == original_epoch, "Checkpoint didn't restore epoch correctly"

    # Clean up
    import os

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"✅ Trainer works correctly! Final loss: {loss:.4f}")


def test_complete_training_pipeline():
    """
    Complete end-to-end training example using all components.

    This demonstrates how Trainer, scheduler, gradient clipping,
    and checkpointing work together in a real training scenario.
    """
    print("🏗️ Building Complete Training Pipeline...")
    print("=" * 60)

    # Step 1: Create model using REAL Linear layer
    class SimpleNN:
        def __init__(self):
            self.layer1 = Linear(3, 5)
            self.layer2 = Linear(5, 2)
            self.training = True

        def forward(self, x):
            x = self.layer1.forward(x)
            # Simple ReLU-like activation (max with 0)
            x = Tensor(np.maximum(0, x.data))
            x = self.layer2.forward(x)
            return x

        def parameters(self):
            return self.layer1.parameters() + self.layer2.parameters()

    print("✓ Model created: 3 → 5 → 2 network")

    # Step 2: Create optimizer
    model = SimpleNN()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    print("✓ Optimizer: SGD with momentum")

    # Step 3: Create loss function
    loss_fn = MSELoss()
    print("✓ Loss function: MSE")

    # Step 4: Create scheduler
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.001, total_epochs=5)
    print("✓ Scheduler: Cosine annealing (0.1 → 0.001)")

    # Step 5: Create trainer with gradient clipping
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        grad_clip_norm=1.0,
    )
    print("✓ Trainer initialized with gradient clipping")

    # Step 6: Create synthetic training data
    train_data = [
        (Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(4, 2))),
        (Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(4, 2))),
        (Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(4, 2))),
    ]
    print("✓ Training data: 3 batches of 4 samples")

    # Step 7: Train for multiple epochs
    print("\n🚀 Starting Training...")
    print("-" * 60)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Learning Rate':<15}")
    print("-" * 60)

    for epoch in range(3):
        loss = trainer.train_epoch(train_data)
        lr = scheduler.get_lr(epoch)
        print(f"{epoch:<8} {loss:<12.6f} {lr:<15.6f}")

    # Step 8: Save checkpoint
    checkpoint_path = "/tmp/training_example_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)
    print(f"\n✓ Checkpoint saved: {checkpoint_path}")

    # Step 9: Evaluate
    eval_loss, accuracy = trainer.evaluate(train_data)
    print(f"✓ Evaluation - Loss: {eval_loss:.6f}, Accuracy: {accuracy:.6f}")

    # Clean up
    import os

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n" + "=" * 60)
    print("✅ Complete training pipeline executed successfully!")
    print("\n💡 This pipeline demonstrates:")
    print("   • Model → Optimizer → Loss → Scheduler → Trainer integration")
    print("   • Training loop with scheduling and gradient clipping")
    print("   • Checkpointing for training persistence")
    print("   • Evaluation mode for model assessment")
