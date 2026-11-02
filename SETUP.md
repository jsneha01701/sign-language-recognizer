# Complete Setup Guide - Sign Language Recognizer

This guide covers all steps to set up the project on a fresh Windows laptop.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Verification](#verification)
4. [Troubleshooting](#troubleshooting)

---

## System Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **Java** | 21 LTS | Spring Boot backend |
| **Python** | 3.12.4 | ML/MediaPipe scripts |
| **Maven** | 3.9+ | Build tool (included via mvnw) |
| **Git** | Latest | Version control |
| **FastAPI** | Latest | Python web framework |
| **PyTorch** | 2.0+ | ML training & inference |
| **MediaPipe** | 0.10.35 Tasks API | Hand/pose detection |
| **OpenCV** | Latest | Video processing |
| **ONNX Runtime** | 1.17.1 (Java) | Model inference |

The Python extractor uses MediaPipe's Tasks Holistic Landmarker. It returns the same
258 pose-and-hand values per frame expected by the existing training and prediction code.
The landmark model differs from the legacy API, so re-extract existing videos and retrain
if combining new results with data created by the old extractor.
The live web demo assembles recognized starter-vocabulary signs in order as an English
gloss sentence. It is not a general ISL-to-English translator; accurate translation of
continuous ISL requires a much larger, sentence-labeled ISL video dataset.

---

## Installation Steps

### STEP 1: Install Java 21

#### Option A: Using Windows Package Manager (Recommended)
```powershell
winget install Oracle.JDK.21
``` 

#### Option B: Manual Installation
1. Download from: https://www.oracle.com/java/technologies/downloads/
2. Choose "Windows x64 Installer"
3. Run installer and select "Add to PATH"
4. Restart PowerShell

**Verify Installation:**
```powershell
java -version
# Expected output: java version "21" or similar
```

---

### STEP 2: Install Python 3.12.4

#### Option A: Using Windows Package Manager (Recommended)
```powershell
winget install Python.Python.3.12
```

#### Option B: Manual Installation
1. Download from: https://www.python.org/downloads/release/python-3124/
2. Choose the Windows installer
3. **Important:** Check "Add Python to PATH" during installation
4. Restart PowerShell

**Verify Installation:**
```powershell
python --version
# Expected output: Python 3.12.4

pip --version
# Expected output: pip version information
```

---

### STEP 3: Clone/Navigate to Project

```powershell
# Navigate to your project directory
cd C:\Users\lavib\Downloads\sign-language-recognizer

# Verify you're in the right location
Get-Location
# Should show: C:\Users\lavib\Downloads\sign-language-recognizer

# List contents to verify
Get-ChildItem
# Should see: pom.xml, mvnw, mvnw.cmd, mediapipe_service, src, plan.md, HELP.md
```

---

### STEP 4: Create Python Virtual Environment

```powershell
# Navigate to project root
cd C:\Users\lavib\Downloads\sign-language-recognizer

# Create virtual environment using Python 3.12.4
python -m venv .\mediapipe_service\.venv312

# Activate virtual environment (Windows PowerShell)
.\mediapipe_service\.venv312\Scripts\Activate.ps1

# If you get execution policy error, run this once:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verify activation - prompt should show (.venv312) prefix
# Example: (.venv312) PS C:\Users\lavib\Downloads\sign-language-recognizer>
```

---

### STEP 5: Upgrade pip and Install Python Dependencies

```powershell
# Make sure virtual environment is activated
# (you should see (.venv312) in prompt)

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install all Python dependencies
pip install -r .\mediapipe_service\requirements.txt

# Download the MediaPipe Holistic Landmarker model
New-Item -ItemType Directory -Force .\mediapipe_service\models | Out-Null
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task" `
  -OutFile .\mediapipe_service\models\holistic_landmarker.task

# Verify installations
pip list
# Should show: fastapi, uvicorn, mediapipe, opencv-contrib-python, numpy, torch, torchvision, onnx, python-multipart
```

---

### STEP 6: Verify Maven Setup

```powershell
# Deactivate Python virtual environment (optional)
deactivate

# Navigate to project root (if not already there)
cd C:\Users\lavib\Downloads\sign-language-recognizer

# Test Maven wrapper
.\mvnw --version
# Expected output: Apache Maven 3.9.x or similar
```

---

### STEP 7: Build Java Project

```powershell
# Full build with tests
.\mvnw clean install

# OR skip tests for faster build (first time)
.\mvnw clean install -DskipTests

# Wait 2-5 minutes for dependencies to download
```

**Expected Output:**
```
[INFO] BUILD SUCCESS
[INFO] Total time: X.XXs
[INFO] Finished at: YYYY-MM-DD'T'HH:MM:SS...
```

---

### STEP 8: (Optional) Install Git

```powershell
winget install Git.Git
```

Verify:
```powershell
git --version
```

---

## Verification

### Complete Verification Script

Run this PowerShell script to verify everything is installed correctly:

```powershell
# Create a verification script
@"
Write-Host "=== JAVA VERIFICATION ===" -ForegroundColor Green
java -version 2>&1

Write-Host "`n=== PYTHON VERIFICATION ===" -ForegroundColor Green
python --version

Write-Host "`n=== PIP PACKAGES ===" -ForegroundColor Green
pip list | findstr "fastapi|uvicorn|mediapipe|opencv|numpy|torch"

Write-Host "`n=== MAVEN VERIFICATION ===" -ForegroundColor Green
.\mvnw --version

Write-Host "`n=== PROJECT STRUCTURE ===" -ForegroundColor Green
Get-ChildItem -Path . | Select-Object Name

Write-Host "`n=== PYTHON VENV ===" -ForegroundColor Green
if (Test-Path ".\mediapipe_service\.venv312") {
    Write-Host ".venv312 folder exists: YES" -ForegroundColor Green
} else {
    Write-Host ".venv312 folder exists: NO" -ForegroundColor Red
}

Write-Host "`n=== BUILD VERIFICATION ===" -ForegroundColor Green
if (Test-Path ".\target\classes") {
    Write-Host "Java build successful: YES" -ForegroundColor Green
} else {
    Write-Host "Java build successful: NO (run mvnw clean install)" -ForegroundColor Yellow
}

Write-Host "`n=== ALL CHECKS COMPLETE ===" -ForegroundColor Cyan
"@ | Out-File -FilePath verify_setup.ps1 -Encoding UTF8

# Run the verification script
.\verify_setup.ps1
```

---

## Running the Project

### Start Python MediaPipe Service

```powershell
# Run from mediapipe_service after collecting webcam samples and training the model
cd C:\Users\lavib\Downloads\sign-language-recognizer\mediapipe_service
.\.venv312\Scripts\Activate.ps1

# Start the API and live-video web app
uvicorn extractor:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser and allow camera access. The page shows the
recognized English gloss sentence. Show one sign at a time and briefly lower your hands
between signs; the `rest` class separates consecutive signs. Use **Clear sentence** to
start a new sentence.

### Collect Data and Train the Live Sentence Demo

The demo requires real webcam samples for all nine starter labels. Record varied examples
of each sign, including a neutral `rest` pose. The existing generated samples in
`data\landmarks` are synthetic and are deliberately not used for training this model.
Each labeled sample captures 24 frames; the collector aims for 30 samples per sign.

```powershell
cd C:\Users\lavib\Downloads\sign-language-recognizer\mediapipe_service
.\.venv312\Scripts\Activate.ps1

# Follow the camera prompts; press Space to record each sample.
python collect_training_data.py

# Train the live webcam samples (30 per label are recommended).
python train.py

# Then start the web app using the command above.
```

The starter labels are `hello`, `i`, `need`, `help`, `water`, `thankyou`, `yes`, `no`,
and `rest`. The English sentence is assembled from the recognized signs in order; users
must record examples of the intended ISL signs themselves. This prototype does not yet
translate full ISL grammar or recognize unrestricted continuous signing.

**Java app note:** The existing Spring controller uses a separate pixel-based ONNX path;
the live ISL webcam demo currently runs in the Python/FastAPI app above.

### Start Java Spring Application

```powershell
# In project root (deactivate Python venv if still active)
deactivate

# Run Spring Boot application
.\mvnw spring-boot:run

# You should see:
# ... c.s.s.SignLanguageRecognizerApplication : Started SignLanguageRecognizerApplication in X.XXX seconds
```

Access the application at: `http://localhost:8080`

---

## Quick Reference Commands

### First-Time Setup (Copy-Paste All)
```powershell
# Install Java
winget install Oracle.JDK.21

# Install Python
winget install Python.Python.3.12

# Create venv
cd C:\Users\lavib\Downloads\sign-language-recognizer
python -m venv .\mediapipe_service\.venv312

# Activate venv
.\mediapipe_service\.venv312\Scripts\Activate.ps1

# Install Python packages
pip install --upgrade pip
pip install -r .\mediapipe_service\requirements.txt

# Download the MediaPipe Holistic Landmarker model
New-Item -ItemType Directory -Force .\mediapipe_service\models | Out-Null
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task" `
  -OutFile .\mediapipe_service\models\holistic_landmarker.task

# Verify Maven (needed only for the separate Spring app)
.\mvnw --version

# Build Java project
.\mvnw clean install -DskipTests
```

### Daily Development
```powershell
# Terminal 1: Start Python service
cd C:\Users\lavib\Downloads\sign-language-recognizer
.\mediapipe_service\.venv312\Scripts\Activate.ps1
cd mediapipe_service
uvicorn extractor:app --host 127.0.0.1 --port 8000

# Terminal 2: Start Java service
cd C:\Users\lavib\Downloads\sign-language-recognizer
deactivate
.\mvnw spring-boot:run
```

---

## Troubleshooting

### Issue: "Python not found" or "python is not recognized"
**Solution:**
```powershell
# Restart PowerShell after installation
# OR add Python to PATH manually:
# 1. Search "Environment Variables" in Windows
# 2. Click "Edit the system environment variables"
# 3. Click "Environment Variables" button
# 4. Under "User variables", click "New"
# 5. Variable name: PATH, Variable value: C:\Users\{YourUsername}\AppData\Local\Programs\Python\Python312
# 6. Click OK and restart PowerShell
```

### Issue: "Java not found" or "java is not recognized"
**Solution:**
```powershell
# Restart PowerShell after installation
# Verify installation path:
dir "C:\Program Files\Java"
```

### Issue: Execution Policy Error for `.venv312\Scripts\Activate.ps1`
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Type 'Y' and press Enter
# Then try activating venv again
.\mediapipe_service\.venv312\Scripts\Activate.ps1
```

### Issue: Maven Build Fails ("BUILD FAILURE")
**Solution:**
```powershell
# Try skipping tests first
.\mvnw clean install -DskipTests

# If still fails, check Java version
java -version
# Should be 21 or higher

# Clear Maven cache and rebuild
.\mvnw clean
.\mvnw install -DskipTests
```

### Issue: Port 8000 or 8080 Already in Use
**Solution:**
```powershell
# Find process using port 8000 (Python)
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /PID {PID} /F

# For Java port 8080
netstat -ano | findstr :8080
taskkill /PID {PID} /F
```

### Issue: "pip install" is very slow
**Solution:**
```powershell
# Use a faster PyPI mirror
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r .\mediapipe_service\requirements.txt

# Or upgrade pip first
python -m pip install --upgrade pip
```

### Issue: CUDA/GPU not working with PyTorch
**Solution:**
```powershell
# For CPU-only (recommended for testing):
# Already installed correctly

# For NVIDIA GPU support:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## System Specifications Verified On

- **OS:** Windows 10/11 (64-bit)
- **RAM:** 8GB minimum (16GB recommended)
- **Disk Space:** 20GB available
- **GPU:** Optional (NVIDIA CUDA 11.8+ for GPU support)

---

## Next Steps After Installation

1. ✅ Complete setup using this guide
2. Collect training data (record videos of sign language)
3. Process videos through MediaPipe extractor
4. Train the model using `train.py`
5. Test predictions with `quick_predict.py`
6. Export model to ONNX for Java integration

See `plan.md` for detailed development roadmap.

---

## Support

If you encounter issues:
1. Check troubleshooting section above
2. Verify all prerequisites are installed: `verify_setup.ps1`
3. Check application logs in PowerShell console
4. Review `plan.md` for detailed step-by-step guide
