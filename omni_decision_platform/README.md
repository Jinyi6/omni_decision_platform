# **1. Overview**

This is an open-source project dedicated to integrating a variety of cutting-edge decision intelligence technologies to build a unified, efficient, and scalable algorithm library. It aims to provide academic researchers and industry developers with a powerful and flexible toolset to solve complex real-world decision-making problems.

# **2. Project Objectives**

  * **Integration:** Consolidate decision-making algorithms from multiple domains into a single, unified framework, making it easy for users to call and combine them.
  * **Cutting-edge:** Actively track and implement the latest algorithms from various fields to maintain the project's technological leadership.
  * **Practicality:** Focus on the real-world application of algorithms by providing a rich collection of examples and use cases from actual scenarios.

# **3. Contribution Guidelines (Contribution Workflow)**

We use the standard Fork & Pull Request workflow to ensure code quality and organized project management. All code contributions must go through this process.

The complete workflow is as follows:

**Step 1: Fork the Project Repository**
In the top-right corner of the project's GitHub page, click the "Fork" button. This will create a complete copy of the project under your own GitHub account.

**Step 2: Clone Your Fork to Your Local Machine**
Open your terminal and use the `git clone` command to clone your forked repository to your local computer.

**Step 3: Create a New Working Branch**
Before you start writing code, create a new branch with a descriptive name based on the `main` branch. Do not make changes directly on the `main` branch.

Branch Naming Convention: `feature/your-feature-name` or `fix/bug-description`. For example:

```bash
# Create a new branch from the latest main branch
git checkout main
git pull origin main
git checkout -b feature/add-topic-1
```

**Step 4: Code & Commit**
Add or modify code in your chosen topic directory (e.g., `topics/topic_2/`). Once your development is complete, commit your changes.

```bash
# Add the files you have modified
git add .

# Commit the changes with a clear commit message
git commit -m "feat: ..."
```

**Step 5: Push Your Branch to Your Remote Repository**
Push your local branch to your forked repository on GitHub.

```bash
git push origin feature/add-topic-1
```

**Step 6: Create a Pull Request (PR)**
Return to your repository page on GitHub. You will see a prompt to create a Pull Request from the branch you just pushed. Click that button, and then:

  * **Check the details:** Ensure the base repository is `main` and the head repository is your own branch.
  * **Fill in the title and description:** Write a clear title and a detailed description.
  * **Submit the PR:** Click the "Create Pull Request" button.

Once your PR is approved, your code will be merged into the main project's `main` branch. 

### **4. Repository Directory Structure**

```
omi_decision_platform/
├── .gitignore                # Git ignore file configuration
├── LICENSE                   # Project license
├── README.md                 # The file you are currently reading
└── src/                      # Directory for core algorithm code
    ├── topic_1
    │   └── README.md
    ├── topic_2
    │   └── README.md
    ├── topic_3
    │   └── README.md
    ├── topic_4
    │   └── README.md
    └── topic_5
        └── README.md
```