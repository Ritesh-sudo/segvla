# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

LeRobot is a PyTorch-based robotics library providing state-of-the-art machine learning models, datasets, and tools for real-world robotics. The library focuses on imitation learning and reinforcement learning, with support for simulation environments and real-world robot control.

## Architecture

### High-Level Organization

The codebase follows a modular architecture with clear separation between:

- **Policies**: Neural network models for robot control (ACT, Diffusion, TDMPC, VQBeT, etc.)
- **Environments**: Simulation environments (ALOHA, PushT) via gymnasium interfaces
- **Datasets**: Dataset loading, processing, and management using LeRobotDataset format
- **Robots**: Real-world robot implementations (Koch, ALOHA, SO-100, SO-101, etc.)
- **Motors**: Motor control interfaces (Dynamixel, Feetech)
- **Cameras**: Camera interfaces (OpenCV, Intel RealSense)
- **Teleoperators**: Teleoperation interfaces (gamepad, keyboard, phone, etc.)

### Directory Structure

```
src/lerobot/
├── policies/          # Policy implementations (act, diffusion, tdmpc, vqbet, groot, smolvla, etc.)
├── datasets/          # Dataset loading and processing (LeRobotDataset)
├── envs/              # Environment configurations and wrappers
├── robots/            # Robot implementations (koch, aloha, so100, so101, hope_jr, lekiwi, etc.)
├── motors/            # Motor bus abstractions (dynamixel, feetech)
├── cameras/           # Camera interfaces (opencv, intelrealsense)
├── teleoperators/     # Teleoperation devices (gamepad, keyboard, phone, homunculus, etc.)
├── configs/           # Configuration dataclasses using draccus
├── scripts/           # CLI entry points (lerobot-train, lerobot-eval, etc.)
├── model/             # Shared model components
├── processor/         # Data preprocessing
├── optim/             # Optimizers and schedulers
├── rl/                # Reinforcement learning utilities
├── utils/             # Common utilities
├── async_inference/   # Async policy inference (client-server architecture)
└── transport/         # gRPC-based communication
```

### Configuration System

The library uses **draccus** for configuration management, which provides dataclass-based configs with CLI overrides:

- Policy configs: `src/lerobot/configs/policies.py` and policy-specific files (e.g., `configuration_act.py`)
- Training configs: `src/lerobot/configs/train.py`
- Eval configs: `src/lerobot/configs/eval.py`
- Default configs: `src/lerobot/configs/default.py`

CLI arguments use dot notation for nested configs: `--policy.type=act --policy.dim_model=64`

### Policy Architecture

Each policy follows a standard structure:

1. **Configuration class**: Defines policy hyperparameters (e.g., `ACTConfig`)
2. **Modeling class**: Implements the policy neural network (e.g., `ACTPolicy`)
3. **Processor class**: Handles input preprocessing (optional)

Policies inherit from `nn.Module` and implement:

- `forward()`: Training forward pass
- `select_action()`: Inference action selection
- `reset()`: Reset policy state (for stateful policies)

### Dataset Format: LeRobotDataset

LeRobotDataset is the standard dataset format, built on:

- **Hugging Face Datasets**: Arrow/Parquet backend for metadata and frame info
- **MP4 videos**: Efficient video storage
- **JSON metadata**: Episode info, statistics, tasks

Key features:

- Temporal indexing with `delta_timestamps` for multi-frame observations
- Automatic download from Hugging Face Hub
- Support for both local and remote datasets
- Built-in statistics and normalization

Datasets are located at: `~/.cache/huggingface/lerobot/` by default

### Adding New Components

**New Policy**:

1. Create directory: `src/lerobot/policies/my_policy/`
2. Add files: `configuration_my_policy.py`, `modeling_my_policy.py`
3. Update `lerobot/__init__.py`: Add to `available_policies` and `available_policies_per_env`
4. Update `tests/test_available.py`: Import your policy class

**New Environment**:

1. Create gymnasium environment in separate repo (e.g., `gym-myenv`)
2. Update `lerobot/__init__.py`: Add to `available_tasks_per_env` and `available_datasets_per_env`
3. Add as optional dependency in `pyproject.toml`

**New Dataset**:

1. Convert to LeRobotDataset format using `src/lerobot/datasets/push_dataset_to_hub/`
2. Upload to Hugging Face Hub
3. Update `lerobot/__init__.py`: Add to `available_datasets_per_env`

**New Robot**:

1. Create directory: `src/lerobot/robots/my_robot/`
2. Implement robot class following the Robot interface
3. Update `lerobot/__init__.py`: Add to `available_robots`

### Important Patterns

**Configuration Loading**:

- Configs can be loaded from pretrained hub paths: `PreTrainedConfig.from_pretrained("lerobot/diffusion_pusht")`
- CLI overrides are applied on top of loaded configs
- Training resumes reuse the saved `train_config.json`

**Hub Integration**:

- Models and datasets use Hugging Face Hub for storage/retrieval
- Upload checkpoints: `huggingface-cli upload user/repo-name path/to/pretrained_model`
- Checkpoints contain: `config.json`, `model.safetensors`, `train_config.json`

**Device Management**:

- Policies support device specification: `--policy.device=cuda`
- Use `DEVICE` environment variable for testing: `make test-act-ete-train DEVICE=cuda`

**Video Processing**:

- Uses `torchcodec` for fast video decoding (when available)
- Falls back to `av` (PyAV) for compatibility
- Videos encoded with `libsvtav1` codec
- Requires ffmpeg 7.x with `libsvtav1` support

## Special Considerations

### Python and Dependencies

- Minimum Python version: 3.10
- PyTorch 2.2+ required
- Install `ffmpeg` via conda: `conda install ffmpeg -c conda-forge`
- Use `poetry` or `uv` for dependency management (not bare `pip`)

### Testing with Simulation Environments

Tests that require simulation environments (aloha, pusht) are skipped locally if not installed but run in CI. Install with `--all-extras` to run them locally.

### Git LFS

Test artifacts are stored with Git LFS. Install and pull:

```bash
brew install git-lfs  # or sudo apt-get install git-lfs
git lfs install
git lfs pull
```

### Pre-commit Hooks

The repository uses extensive pre-commit hooks:

- **ruff**: Formatting and linting
- **mypy**: Type checking (being gradually rolled out)
- **typos**: Spell checking
- **bandit**: Security checks
- **gitleaks**: Secret detection
- **prettier**: Markdown formatting

Run `pre-commit` before committing or use `pre-commit run --all-files` to check everything.

### Code Style

- Line length: 110 characters
- Use double quotes
- Google-style docstrings (when docstrings are added)
- Import order: Hugging Face, core dependencies, lerobot modules


# Important User Notes

We will be adding a new policy called 'segvla' which is an exact copy of smolvla except it has a segmentation model (facebook/mask2former-swin-large-ade-semantic). 
This process will take time and has many components that must be completed properly or else it will fail. 

We DO NOT need to write production ready code. All we need is a trainable model that can be evaluated against the original smolvla. The performance of the model (in terms of metrics like accuracy etc) does not really matter. The code does not need to be clean, it needs to work. 

I have already copied the smolvla/ policy dir to segvla/ and renamed the files appropriately (except smolvlm_with_expert.py since that will remain the same i believe).
Inside these files, rename the classes to `SegVLAConfig` and `SegVLAPolicy` and so on.

Part 1: Implementation Plan
1. Configuration (SmolVLAConfig) We will add two new fields to the configuration object:

  * use_segmentation_adapter: Boolean flag to enable the module.
  * num_segmentation_classes: Integer (default 150) to define the input channels.

2. The Adapter Module (SegmentationFusionAdapter) We will create a custom nn.Module.

  * Architecture: A simple 1x1 Conv2d.
    * Input Channels: 3 (RGB) + 150 (Logits) = 153.
    * Output Channels: 3 (Fused RGB).
    * Kernel Size: 1 (preserving spatial alignment pixel-for-pixel).
  * Initialization (Zero-Init):
    * The weights corresponding to the RGB input channels will be set to the Identity matrix.
    * The weights corresponding to the 150 Logit channels will be set to Zero.
    * Biases will be set to Zero.
    * Result: At step 0, the output is mathematically identical to the original image.

3. Data Preprocessing (prepare_images) This is the most critical logic change. Currently, prepare_images resizes, pads, and normalizes the RGB image. We must inject the fusion step in the middle:

  * Resize & Pad: Apply resize_with_pad to both the RGB image and the Segmentation Logits independently but with identical parameters.
  * Concatenate: Stack them along the channel dimension.
  * Forward Pass: Pass the stack through the SegmentationFusionAdapter.
  * Normalize: Only after the adapter outputs the new 3-channel image do we apply the SigLIP normalization (scaling to [-1, 1]).


Everything else is on you, good luck!