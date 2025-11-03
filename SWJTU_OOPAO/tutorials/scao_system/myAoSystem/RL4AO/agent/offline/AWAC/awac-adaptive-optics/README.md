# AWAC Adaptive Optics Project

This project implements the AWAC (Actor with Advantage-weighted Regression) algorithm for adaptive optics using reinforcement learning. The goal is to optimize the performance of adaptive optics systems through both offline and online training strategies.

## Project Structure

The project is organized into the following directories and files:

```
awac-adaptive-optics
├── agent
│   └── awac.py                  # Implements the AWAC algorithm for training the agent.
├── envs
│   └── adaptive_optics_env.py    # Contains the adaptive optics reinforcement learning environment.
├── train
│   └── train_awac.py             # Orchestrates the training process using the AWAC algorithm.
├── utils
│   ├── replay_buffer.py           # Defines a ReplayBuffer class for storing experiences.
│   └── helpers.py                 # Contains utility functions for data preprocessing and logging.
├── data
│   └── offline_dataset.npz        # NumPy archive containing the offline dataset for training.
├── configs
│   └── params.py                  # Configuration parameters for training and environment settings.
├── requirements.txt               # Lists the dependencies required to run the project.
└── README.md                      # Documentation for the project.
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd awac-adaptive-optics
pip install -r requirements.txt
```

## Usage

1. **Environment Setup**: The adaptive optics environment is defined in `envs/adaptive_optics_env.py`. You can customize the environment parameters in `configs/params.py`.

2. **Training the Agent**: To train the AWAC agent, run the following command:

   ```bash
   python train/train_awac.py --paramFile <your_param_file>
   ```

   Replace `<your_param_file>` with the name of your parameter file.

3. **Evaluating the Agent**: After training, you can evaluate the agent's performance using the evaluation functions included in the training script.

## Algorithms

The AWAC algorithm combines the benefits of offline and online reinforcement learning. It utilizes a replay buffer to store experiences and employs advantage-weighted regression to update the policy based on the collected data.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.