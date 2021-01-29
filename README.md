# Stochastic Fractal Search In Python
Stochastic Fractal Search (SFS) is a meta-heuristic optimization algorithm inspired from the concept of fractals, SFS was originally created by Dr Hamid Salimi in his 
article, named [Stochastic Fractal Search: A powerful metaheuristic algorithm](https://www.sciencedirect.com/science/article/abs/pii/S0950705114002822)

Fractals are impressive recursive structures and a major interest of computer graphics curriculum.

## Sample Fractals
<p align="center">
<img src="https://i.pinimg.com/originals/12/27/1a/12271a8f5a1157cd194cec0e2e5d0757.gif" alt="Sierpinski triangle" />
<img src="https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/37813/versions/1/screenshot.gif" alt="Mandelbrot Set" />
</p>

## Project Structure
```
.
├── LICENSE 
├── README.md
├── sfs.py (The python implementation of stochastic fractal search)
└── walks
    ├── random walk.py (random gaussaing walk demo)
    └── self-avoiding-walk.py (self avoiding gaussaing walk demo)
```
## Features
1. The algorithm is implemented in *sfs.py*, where benchmark functions could be tested effortlessly using [opteval](https://github.com/keit0222/optimization-evaluation) package
2. You may also view some random walk demos under walks directory.

<p align="center">
  <img src="https://media.giphy.com/media/Iok6UIB10yEKchtzEW/giphy.gif"  alt="demo" />
</p>

## Original Matlab implementation
The original creator of SFS has published the Matlab code of the algorithm via [MATLAB Central File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/47565-stochastic-fractal-search-sfs).

## gifski
I used gifski open-source tool for generating high-quality gifs 🎊🎊
