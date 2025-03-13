
### `Ising/ising_observables.jl`

- **Purpose**: Simulates a 2D antiferromagnetic Ising model on a square lattice, optionally with vacancies.  
- **Key Functions**:  
  - `initialize_lattice(...)`: Builds a lattice with a specified vacancy fraction.  
  - `monte_carlo_step!(...)`: Performs one Metropolis sweep at temperature `T`.  
  - `calculate_properties(...)`: Measures magnetization, energy, etc.  
  - `compute_structure_factor(...)`: Computes the spin structure factor `S(Q)`.  
  - Several helper functions to measure quantities like the correlation length, specific heat, transition temperature, etc.  
- **Usage**: Run in Julia (e.g., `julia ising_observables.jl`). The script will generate  data and plots (stored in a `results/` folder).

### `Models/`

All Python scripts here use PyTorch to train generative models on spin configurations. They assume you have generated or loaded data (e.g., `.npz` files containing Ising snapshots). **A GPU** is expected by most scripts.

1. **`train_adbm.py`**  
   - Trains an *advanced DBM* (with optional parallel tempering or persistent CD) on Ising-like data.
   - Supports occupant clamping (meaning vacancy bits are fixed and only spin sign bits are learned).

2. **`train_clamped_adbm.py`**  
   - Similar to `train_adbm.py`, but specifically focuses on occupant‐clamped scenarios for advanced DBM architectures.

3. **`train_clamped.py`**  
   - Shows how to clamp the vacancy/occupant bits in simpler RBMs, DBMs, DBNs, or DRBNs, so that only the spin sign bits are trained.

4. **`train_drbn.py`**  
   - Demonstrates a two-layer “Deep RBM” approach for occupant+sign data. 
   - Can be run in either “unclamped” or “clamped” mode depending on the script calls.

5. **`train_general.py`**  
   - A general script covering RBM/DBM/DBN/DRBN (modified) training. 
   - Has both unclamped (learning occupant + sign) and occupant-clamped modes.



**All Python scripts**:
- **Input**: Typically expect a `.npz` dataset containing spins `[-1, 0, +1]` or occupant+sign encoding.  
- **Output**: Saved model parameters (`.pt` files) in subfolders like `clamped_data/` or `normal/`.  
- **Dependencies**: 
  - `numpy`
  - `PyTorch`
  - `torchvision` (sometimes used)
  - `matplotlib` (for any plotting, if applicable)

---

## Getting Started

1. **Install Dependencies**:
   - **Julia**: Ensure you have packages like `FFTW`, `LsqFit`, `Plots`, `StatsBase`, etc.
   - **Python**: Install `torch`, `numpy`, `matplotlib`, etc.
2. **Run Julia Simulation**:  
   - Inside `Ising/`, run `julia ising_observables.jl` to generate and save Ising snapshots or to analyze a set of temperatures and vacancy fractions.
3. **Train Models**:  
   - Move to `Models/`, pick a script (`train_adbm.py`, `train_drbn.py`, etc.), then run `python train_adbm.py` (for example).  
   - The script will look for a dataset (e.g., `data/disordered_snapshots_16.npz`) and then train a model, saving the results to a `.pt` file.
4. **Adjust Parameters**:  
   - Each Python file has parameters like `EPOCHS_DEFAULT`, `BATCH_SIZE_DEFAULT`, or a range of `temperature_offsets`. Modify as needed.

---

# IsingDL
# IsingDLmod
# IsingDLmod
# IsingDL1
