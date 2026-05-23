
# 🎌 Anime Hybrid Recommender System

A production-grade MLOps pipeline for anime recommendations, combining collaborative filtering and content-based approaches into a hybrid model. The system is fully versioned, experiment-tracked, containerized, and deployed on Google Kubernetes Engine via a Jenkins CI/CD pipeline.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Database Setup (GCP Buckets)](#database-setup-gcp-buckets)
- [Data Ingestion with GCP](#data-ingestion-with-gcp)
- [Jupyter Notebook Testing](#jupyter-notebook-testing)
- [Data Processing](#data-processing)
- [Model Architecture and Training](#model-architecture-and-training)
- [Experiment Tracking with Comet ML](#experiment-tracking-with-comet-ml)
- [Training Pipeline](#training-pipeline)
- [Data Versioning with DVC and GitHub](#data-versioning-with-dvc-and-github)
- [Prediction Helper Functions](#prediction-helper-functions)
- [Flask Web Application](#flask-web-application)
- [CI/CD with Jenkins and Kubernetes](#cicd-with-jenkins-and-kubernetes)
  - [Jenkins Container Setup](#1-jenkins-container-setup)
  - [GitHub Integration](#2-github-integration)
  - [Dockerization](#3-dockerization-of-the-project)
  - [Install GCloud and kubectl on Jenkins](#4-install-google-cloud-and-kubernetes-cli-on-jenkins)
  - [Build and Push Image to GCR](#5-build-and-push-image-to-gcr)
  - [Docker Permissions for Jenkins](#6-give-docker-permissions-to-jenkins)
  - [Kubernetes Deployment](#7-kubernetes-deployment)
- [Contributing](#contributing)

---

## Overview

The **Anime Hybrid Recommender System (AHRS)** is an end-to-end machine learning project that recommends anime titles to users based on their watch history and content features. It integrates:

- A **hybrid recommendation model** (collaborative + content-based filtering)
- A **Flask web app** powered by ChatGPT for natural language interaction
- A full **MLOps pipeline** with DVC, Comet ML, Jenkins, Docker, and GKE

---

## Architecture

```
Raw Data (GCP Bucket)
        │
        ▼
  Data Ingestion
        │
        ▼
  Data Processing
        │
        ▼
 Model Training ──────► Comet ML (Experiment Tracking)
        │
        ▼
  Artifacts (model, weights, checkpoints)
        │
        ▼
  DVC (Data & Model Versioning) ──► GCP Bucket (DVC Remote)
        │
        ▼
  Flask App (Prediction API + ChatGPT UI)
        │
        ▼
  Docker Image ──► Google Container Registry (GCR)
        │
        ▼
  Jenkins CI/CD Pipeline
        │
        ▼
  Google Kubernetes Engine (GKE)
```
---

## Tech Stack

| Category              | Tools / Services                                      |
|-----------------------|-------------------------------------------------------|
| Cloud Storage         | Google Cloud Storage (GCP Buckets)                    |
| Data Versioning       | DVC + GCP Bucket remote                               |
| Code Versioning       | Git + GitHub                                          |
| Experiment Tracking   | Comet ML                                              |
| Model Framework       | Python (PyTorch / TensorFlow / Scikit-learn)          |
| Web Framework         | Flask                                                 |
| AI Integration        | OpenAI ChatGPT API                                    |
| Containerization      | Docker                                                |
| CI/CD                 | Jenkins (Docker-in-Docker)                            |
| Container Registry    | Google Container Registry (GCR)                       |
| Orchestration         | Google Kubernetes Engine (GKE)                        |


---

## Getting Started

### Prerequisites

Make sure the following are installed on your machine:

- Python 3.8+
- Git
- Docker Desktop
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
- A GCP account with billing enabled
- A GitHub account
- A [Comet ML](https://www.comet.com/) account

### Installation

Clone the repository and install the package in editable mode:

bash
git clone https://github.com/<your-username>/anime-hybrid-recommender-system.git
cd anime-hybrid-recommender-system
pip install -e .

---

## Database Setup (GCP Buckets)

This project uses a GCP Bucket as the primary data store.

1. Go to **GCP Console** → search **Buckets** → **Create**
2. Set the bucket name: `anime-hybrid-recommender-system`
3. **Uncheck** "Enforce public access prevention on this bucket"
4. Click **Create**
5. Inside the bucket, click **Upload** and upload the following datasets:
   - `anime`
   - `animelist`
   - `anime_with_synopsis`

---

## Data Ingestion with GCP

### 1. Verify Google Cloud CLI

bash
gcloud --version

### 2. Create a Service Account

1. Go to **GCP** → **IAM & Admin** → **Service Accounts** → **Create Service Account**
2. Name: `anime-hybrid-recommender-system` *(must be globally unique)*
3. Click **Create and Continue**
4. Assign the following roles:
   - `Storage Admin`
   - `Storage Object Viewer`
   - `Owner`
5. Click **Continue** → **Done**

### 3. Generate a Service Account Key

1. In **Service Accounts**, click the **⋮ menu** of your service account
2. Go to **Manage Keys** → **Add Key** → **New Key** → **JSON**
3. A JSON key file will be downloaded — keep it safe

### 4. Grant Bucket Access to the Service Account

1. In your bucket, click the **⋮ menu** → **Edit Access**
2. Click **Add Principal** → select your service account
3. Assign roles: `Storage Object Admin`, `Storage Object Viewer`

### 5. Set Credentials and Install Dependencies

bash
# Windows
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\key.json

# macOS/Linux
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/key.json

pip install -e .

---

## Jupyter Notebook Testing

1. Install the **Jupyter** extension in VS Code
2. Install dependencies:

bash
pip install -e .

3. Open notebooks in the `notebooks/` directory to explore data ingestion, processing, and model prototyping interactively.

---

## Data Processing

The data processing step handles:

- Merging and cleaning raw anime datasets
- Encoding categorical features
- Normalizing ratings
- Splitting data into train/validation/test sets

All processed artifacts are saved to `artifacts/processed/`.

---

## Model Architecture and Training

The hybrid recommender combines:

- **Collaborative Filtering** — learns user-item interaction patterns from the `animelist` dataset
- **Content-Based Filtering** — leverages anime metadata and synopsis embeddings

Training is handled in `src/model_training.py`. Trained artifacts (model, weights, checkpoints) are saved under `artifacts/`.

---

## Experiment Tracking with Comet ML

### Setup

bash
pip install -e .

### Create a Comet ML Project

1. Go to [comet.com](https://www.comet.com/) → **Try Comet for Free** → Log in
2. Click **New Project** → name: `anime-hybrid-recommender-system` → **Public** → **Create**
3. Click your **profile picture** → **API Keys** → copy your API key

### Configure Tracking

In `src/model_training.py`, set the following in the `ModelTraining` initializer:

python
api_key   = "YOUR_COMET_API_KEY"
workspace = "YOUR_COMET_WORKSPACE"

After running `model_training.py`, your experiment metrics, parameters, and charts will appear in the Comet ML dashboard.

---

## Training Pipeline

The full training pipeline (`src/pipeline.py`) orchestrates:

1. Data ingestion from GCP
2. Data processing
3. Model training with experiment tracking
4. Artifact saving

Run the pipeline end-to-end:

bash
python src/pipeline.py

---

## Data Versioning with DVC and GitHub

### 1. Create a DVC Remote Bucket

1. Go to **GCP** → **Buckets** → **Create**
2. Name: `my-dvc-bucket` *(must be globally unique)*
3. **Uncheck** "Enforce public access prevention on this bucket"
4. Click **Create**
5. In the bucket's **⋮ menu** → **Data Access** → **Add Principal**
   - Select your service account
   - Roles: `Storage Object Admin`, `Storage Object Viewer`
6. Upload the raw datasets: `anime`, `animelist`, `anime_with_synopsis`

### 2. Initialize DVC and Track Artifacts

bash
pip install -e .
dvc init .

# Track artifact directories
dvc add artifacts/raw
dvc add artifacts/processed
dvc add artifacts/model
dvc add artifacts/model_checkpoint
dvc add artifacts/weights

# Verify tracked files
dvc status

### 3. Commit DVC Files to Git

bash
git add .
git commit -m "Add dvc files"

### 4. Configure DVC Remote and Push

bash
dvc remote add -d myremote gs://my-dvc-bucket/
dvc push

### 5. Pull Data from Remote

bash
dvc pull

### 6. Push to GitHub

Create a new GitHub repository and push your local repo:

bash
git remote add origin https://github.com/<your-username>/anime-hybrid-recommender-system.git
git push -u origin main

---

## Prediction Helper Functions

The `src/prediction.py` module provides helper functions for:

- Loading trained model weights
- Preprocessing user input
- Generating top-N anime recommendations
- Formatting output for the Flask API

---

## Flask Web Application

The app (`app/app.py`) provides a web interface powered by Flask and ChatGPT for natural language anime recommendations.

bash
pip install -e .
python app/app.py

Navigate to `http://localhost:5000` to use the app.

---

## CI/CD with Jenkins and Kubernetes

The full deployment pipeline automates: build → test → dockerize → push to GCR → deploy to GKE.

### 1. Jenkins Container Setup

Build and run a custom Jenkins image with Docker-in-Docker (DinD) support:

bash
cd custom_jenkins
docker build -t jenkins-dind-AHRS .
docker images

docker run -d --name jenkins-dind-AHRS \
  --privileged \
  -p 8080:8080 -p 50000:50000 \
  -v //var/run/docker.sock:/var/run/docker.sock \
  -v jenkins_home:/var/jenkins_home \
  jenkins-dind-AHRS

> **Windows users:** Use `^` instead of `\` for line continuation in CMD.

Verify the container is running:

bash
docker ps
docker logs jenkins-dind-AHRS

**Initial Jenkins Setup:**

1. Open `http://localhost:8080`
2. Copy the admin password from the logs and paste it
3. Select **Install suggested plugins**
4. Create your first admin user
5. Click **Start using Jenkins**

**Install Python inside the Jenkins container:**

bash
docker exec -u root -it jenkins-dind-AHRS bash

apt update -y
apt install -y python3
python3 --version
ln -s /usr/bin/python3 /usr/bin/python
python --version
apt install -y python3-pip
apt install -y python3-venv

exit
docker restart jenkins-dind-AHRS

---

### 2. GitHub Integration

**Generate a GitHub Personal Access Token:**

1. GitHub → **Settings** → **Developer Settings** → **Personal access tokens** → **Tokens (classic)**
2. Click **Generate new token (classic)**
3. Note: `AHRS-token`
4. Permissions: `repo`, `admin:repo_hook`
5. Click **Generate token** → copy the token

**Add Credentials to Jenkins:**

1. Jenkins → **Manage Jenkins** → **Credentials** → **global** → **Add Credentials**
   - Kind: `Username with password`
   - Scope: `Global`
   - Username: your GitHub username
   - Password: your access token
   - ID: `AHRS-token`
   - Description: `AHRS-token`
   - Click **Create**

**Create the Jenkins Pipeline:**

1. Jenkins → **New Item** → Name: `AHRS-MLOPS` → **Pipeline** → OK
2. Under **Pipeline**:
   - Definition: `Pipeline script from SCM`
   - SCM: `Git`
   - Repository URL: your GitHub repo URL
   - Credentials: select `AHRS-token`
   - Branch Specifier: `*/main`
3. Click **Apply** → **Save**

**Generate the checkout pipeline script:**

1. Jenkins → **AHRS-MLOPS** → **Pipeline Syntax**
2. Sample Step: `checkout: Check out from version control`
3. Fill in your repo URL, credentials, and branch
4. Click **Generate Pipeline Script**
5. Paste the generated script into the first stage of your `Jenkinsfile`

Push your local repo to GitHub, then trigger a build:


Jenkins → AHRS-MLOPS → Build Now

---

### 3. Dockerization of the Project

**Add GCP Service Account Key as a Jenkins Secret:**

1. Jenkins → **Manage Jenkins** → **Credentials** → **global** → **Add Credentials**
   - Kind: `Secret file`
   - Scope: `Global`
   - Upload your GCP service account JSON key file
   - ID: `gcp-key`
   - Description: `gcp-key`
   - Click **Create**

Trigger a build to verify Dockerization:


Jenkins → AHRS-MLOPS → Build Now

---

### 4. Install Google Cloud and Kubernetes CLI on Jenkins

bash
docker exec -u root -it jenkins-dind-AHRS bash

apt-get update
apt-get install -y curl apt-transport-https ca-certificates gnupg

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | \
  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

apt-get update && apt-get install -y google-cloud-sdk
apt-get update && apt-get install -y kubectl
apt-get install -y google-cloud-sdk-gke-gcloud-auth-plugin

kubectl version --client
gcloud --version

exit
docker restart jenkins-dind-AHRS

---

### 5. Build and Push Image to GCR

In your `Jenkinsfile`, set the GCP project ID:

groovy
environment {
    GCP_PROJECT = 'your-gcp-project-id'
}

The pipeline will build the Docker image and push it to Google Container Registry automatically.

**Enable required GCP APIs:**

GCP → **API & Services** → **Library** → enable:
- `Google Container Registry API`
- `Artifact Registry API`
- `Kubernetes Engine API`

---

### 6. Give Docker Permissions to Jenkins

bash
docker exec -u root -it jenkins-dind-AHRS bash

groupadd docker
usermod -aG docker jenkins
usermod -aG root jenkins

exit
docker restart jenkins-dind-AHRS

**Create a GKE Cluster:**

1. GCP → search **Kubernetes Engine** → **Clusters** → **Create Cluster**
2. Name: `anime-hybrid-recommender-system-cluster`
3. Check **Access using DNS**
4. Check **Access using IPv4 addresses**
5. Click **Create Cluster**

---

### 7. Kubernetes Deployment

**Configure the deployment manifest:**

In `deployment.yml`, set the container image:

yaml
containers:
  - name: ahrs-app
    image: gcr.io/<GCP_PROJECT>/anime-hybrid-recommender-system:<TAG>

**Configure the Jenkinsfile deploy stage:**

In the Kubernetes deployment stage of your `Jenkinsfile`, set the GKE cluster region to match your cluster's details in GCP.

**Deploy:**

bash
git add .
git commit -m "Configure Kubernetes deployment"
git push origin main

Then in Jenkins:


Jenkins → AHRS-MLOPS → Build Now

**Verify the deployment:**

- GCP → **Container Registry** — confirm the image was pushed
- GCP → **Kubernetes Engine** → **Clusters** — confirm the cluster is active
- GCP → **Kubernetes Engine** → **Workloads** — confirm the pod is running

Navigate to the external endpoint shown in the workload details to access the live application.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---
