# TopMT: Structure-Based 3D Generative Models for Ligand Design with Enhanced Efficiency and Diversity 

![License](https://img.shields.io/github/license/aeghnnsw/TopMT)
![Issues](https://img.shields.io/github/issues/aeghnnsw/TopMT)
![Stars](https://img.shields.io/github/stars/aeghnnsw/TopMT)
![Forks](https://img.shields.io/github/forks/aeghnnsw/TopMT)

## Table of Contents
- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## About the Project

Recent advancements in 3D structure-based molecular generative models have shown promise in expediting the hit discovery process in drug design. Despite their potential, efficiently generating a focused library of candidate molecules that exhibit both effective interactions and structural diversity remains a significant challenge. 

Here we introduce Topology Molecular Type assignment (TopMT), a novel approach that first utilizes a Generative Adversarial Network (GAN) to construct 3D molecular topologies inside a protein pocket, which is then followed by atom and bond type assignment to those valid molecular topologies using a second GAN. This two-step architecture enables TopMT to efficiently generate diverse and potent ligands with precise 3D poses for specific protein pockets. 


<!-- ![Project Screenshot](path/to/screenshot.png) -->

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have the following prerequisites installed:

- Python 3.7

### Installation

1. Clone the repo
```
   git clone git@github.com:aeghnnsw/TopMT.git
```
2. Navigate to the project directory
```
   cd TopMT
```
3. Install through poetry
```
   pip install poetry
   poetry install
```
## Usage
The examples in the `examples` folder demonstrate how to use the project. For this demo, we'll use the folder `examples/7d3i`.

1. Prepare the target protein file. Follow Vina docking instructions for receptor preparation. After this step, you will have a receptor pdbqt file for Vina docking. For example, in our example, the output file is at `structures/7d3i_p.pdbqt`.

2. Prepare the shape for ligand generation. There are two options:
    - Scaffold-hopping mode: Use known ligands, if they exist. Get the sdf or mol2 file for the ligand. For example, use `structures/7d3i_l.sdf`.
    - Pocket-mapping mode: Use an atom probe to depict the shape of the desired pocket and store the atom positions in a pdb file. For example, use `structures/pocket.pdb`.
3. Generate topologies with the topology graph translation generator. Use the script `scripts/sample_tops.py`. An example command is listed in `01_sample_tops_pd.sh`:
    - `mode`: Determines the mode of generation, use 'pocket' or 'scaffold-hopping'. In this example, `pocket` mode is used.
    - `DGT_ckpt_path`: Path to the checkpoint directory for the deep generative topology model.
    - `n_tops`: Number of topologies to generate.

    The output of this step is a `tops.pkl` file, which contains the generated topologies with 3D coordinates.

4. Perform molecule assignment using the molecular type assignment GAN. Similar to step 3, prepare a script to run this step using `scripts/assign_from_tops.py`. An example is listed in `02_assign_mols.py`:

    Note: Ensure the `--wkdir` parameter is the same as the one used in step 3.

    Expected output: After this step, in the `wkdir`, there will be a `final` directory containing all molecules with a threshold better than the one defined in `assign_from_tops`. These are the generated molecules.

5. Evaluate the generated molecules using any suitable methods. You can choose various computational chemistry tools and techniques to assess the properties, binding affinities, or other relevant metrics of the generated molecules. Ensure that the evaluation criteria align with your specific research goals and requirements.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this project better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Shen Wang - [wang.12218@osu.edu](mailto:your.email@example.com)

Project Link: [git@github.com:aeghnnsw/TopMT.git](git@github.com:aeghnnsw/TopMT.git)