# START HERE: Dimensionality Reduction Workflow Assignment 2 Part 2

Welcome to the assignment project. I have prepared this guide to help you get started with the dimensionality reduction codebase.

## Quick Start (5 Minutes)

1. **Install the required dependencies:**
   ```bash
   cd dimensionality-reduction-workflow-assignment-2-part-2
   pip install -r requirements.txt
   ```

2. **Run the example:**
   ```bash
   python examples/run_dimensionality_reduction.py
   ```

3. **Verify the outputs were generated:**
   ```bash
   ls outputs/dimensionality_reduction/
   ```

If these steps completed successfully, you have executed a complete dimensionality reduction workflow that compares three techniques.

---

## Documentation Structure

I recommend reading the documentation in this order:

1. **QUICKSTART.md** (approximately 5 minutes): Run the example and understand the output
2. **ASSIGNMENT_REPORT.md** (approximately 10 minutes): Technique choices, results, and conclusions
3. **STUDENT_GUIDE.md** (approximately 15 minutes): Week-by-week learning plan
4. **README.md** (approximately 20 minutes): Complete documentation of modules, techniques, and metrics
5. **PROJECT_SUMMARY.md** (approximately 10 minutes): Technical architecture overview

---

## Project Overview

This is an educational machine learning codebase focused on one task:

**Dimensionality Reduction**: Compressing 64-pixel images of handwritten digits into a few informative features.

It compares three techniques:
- **PCA** (from the original project): Unsupervised and linear; keeps the directions of largest variance
- **LDA** (added): Supervised and linear; keeps the directions that best separate the digits
- **Isomap** (added): Unsupervised and non-linear; preserves distances along the curved structure of the data

---

## Project Structure

```
dimensionality-reduction-workflow-assignment-2-part-2/
│
├── Documentation
│   ├── 00_START_HERE.md          (you are currently reading this file)
│   ├── QUICKSTART.md
│   ├── ASSIGNMENT_REPORT.md      (results and conclusions)
│   ├── STUDENT_GUIDE.md
│   ├── README.md
│   └── PROJECT_SUMMARY.md
│
├── Runnable Example
│   ├── examples/run_dimensionality_reduction.py
│   └── main.py                    (runs the full workflow)
│
├── Practice Exercise
│   └── exercises/exercise_01_dimensionality_reduction.py
│
├── Source Code
│   ├── config/                    (configuration management)
│   ├── utils/                     (utility functions)
│   ├── data_loaders/              (dataset loading)
│   └── workflows/                 (dimensionality reduction workflow)
│
└── Output Directories
    ├── outputs/                   (generated plots and results)
    └── visuals/                   (optional: add your own diagrams)
```

---

## Frequently Asked Questions

**Q: Which technique is best?**  
A: It depends on the goal. Isomap is best for 2D visualization, LDA is best for compact features that feed a classifier, and PCA is best for fast, invertible compression without labels. See ASSIGNMENT_REPORT.md.

**Q: Why is there a train/test split in a dimensionality reduction project?**  
A: LDA uses the labels. Fitting every technique on training data and evaluating on test data keeps the comparison fair.

**Q: What should I do if I encounter import errors?**  
A: Run every command from the project root (the folder containing `main.py`) and install the dependencies with `pip install -r requirements.txt`.

---

Remember that learning is an iterative process. Take your time to understand each concept thoroughly, experiment with the code, learn from mistakes, and build your understanding incrementally.
