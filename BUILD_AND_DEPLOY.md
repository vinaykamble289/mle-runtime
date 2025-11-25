# MLE Runtime - Build & Deploy Guide

Complete guide for building all SDKs and deploying to package registries.

## Quick Start

### Windows

```powershell
# Full build with all SDKs
.\setup.ps1

# Build with release packages
.\setup.ps1 -PrepareRelease

# Build without tests (faster)
.\setup.ps1 -SkipTests

# Verbose output
.\setup.ps1 -Verbose
```

### Linux/macOS

```bash
# Make script executable
chmod +x setup.sh

# Full build with all SDKs
./setup.sh

# Build with release packages
./setup.sh --prepare-release

# Build without tests (faster)
./setup.sh --skip-tests

# Verbose output
./setup.sh --verbose
```

---

## What Gets Built

The setup script builds:

1. **C++ Core** (`cpp_core/`)
   - Inference engine
   - Memory-mapped loader
   - CPU/CUDA operators
   - Tests (optional)

2. **Python SDK** (`sdk/python/`)
   - Python bindings
   - pip package
   - Tests
   - Distribution packages (with `-PrepareRelease`)

3. **Node.js SDK** (`sdk/nodejs/`)
   - TypeScript compilation
   - Native addon
   - npm package
   - Tests

4. **Java SDK** (`sdk/java/`)
   - JNI bindings
   - Maven artifacts
   - Tests
   - JAR files

5. **C++ SDK** (`sdk/cpp/`)
   - Header-only library
   - Examples (optional)

---

## Build Output

### Directory Structure After Build

```
project/
├── cpp_core/
│   └── build/              # C++ core binaries
│       ├── libmle_core.so  # Linux
│       ├── libmle_core.dylib  # macOS
│       └── mle_core.dll    # Windows
│
├── sdk/
│   ├── python/
│   │   ├── build/          # Python build artifacts
│   │   └── dist/           # Distribution packages (with -PrepareRelease)
│   │       ├── mle_runtime-1.0.0.tar.gz
│   │       └── mle_runtime-1.0.0-*.whl
│   │
│   ├── nodejs/
│   │   ├── dist/           # Compiled TypeScript
│   │   ├── build/          # Native addon
│   │   └── mle-runtime-*.tgz  # npm package (with -PrepareRelease)
│   │
│   ├── java/
│   │   └── target/         # Maven artifacts
│   │       ├── mle-runtime-1.0.0.jar
│   │       └── mle-runtime-1.0.0-sources.jar
│   │
│   └── cpp/
│       ├── build/          # Examples
│       └── include/        # Headers (ready to use)
```

---

## Individual SDK Builds

### Python SDK Only

```bash
cd sdk/python

# Install dependencies
pip install -r requirements.txt

# Build
python setup.py build

# Install in development mode
pip install -e .

# Create distribution
python setup.py sdist bdist_wheel
```

### Node.js SDK Only

```bash
cd sdk/nodejs

# Install dependencies
npm install

# Build TypeScript
npm run build

# Create package
npm pack
```

### Java SDK Only

```bash
cd sdk/java

# Build
mvn clean install

# Run tests
mvn test

# Create package
mvn package
```

### C++ SDK Only

```bash
cd sdk/cpp

# Configure
mkdir build && cd build
cmake -DBUILD_EXAMPLES=ON ..

# Build
make -j$(nproc)
```

---

## Deployment Scripts

Each SDK has deployment scripts for easy publishing:

### Python

```bash
# Linux/macOS
cd sdk/python
./deploy.sh

# Windows
cd sdk\python
.\deploy.ps1
```

### Node.js

```bash
# Linux/macOS
cd sdk/nodejs
./deploy.sh

# Windows
cd sdk\nodejs
.\deploy.ps1
```

### Java

```bash
# Linux/macOS
cd sdk/java
./deploy.sh

# Windows
cd sdk\java
.\deploy.ps1
```

---

## Testing

### Run All Tests

```bash
# Windows
.\setup.ps1

# Linux/macOS
./setup.sh
```

### Run Individual SDK Tests

```bash
# Python
cd sdk/python
pytest tests/

# Node.js
cd sdk/nodejs
npm test

# Java
cd sdk/java
mvn test

# C++
cd cpp_core/build
ctest
```

---

## Verification

### Verify Python SDK

```bash
python -c "import mle_runtime; print('OK')"
```

### Verify Node.js SDK

```bash
node -e "const mle = require('./sdk/nodejs/dist/index'); console.log('OK');"
```

### Verify Java SDK

```bash
java -cp sdk/java/target/mle-runtime-1.0.0.jar com.mle.runtime.MLEEngine
```

### Verify C++ SDK

```bash
# Check header exists
ls sdk/cpp/include/mle_client.h
```

---

## Deployment to Package Registries

### Prerequisites

1. **npm** - Create account at https://www.npmjs.com/
2. **PyPI** - Create account at https://pypi.org/
3. **Maven Central** - Create account at https://oss.sonatype.org/

### Deploy Python to PyPI

```bash
cd sdk/python

# Install twine
pip install twine

# Build packages
python setup.py sdist bdist_wheel

# Upload to Test PyPI (recommended first)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

### Deploy Node.js to npm

```bash
cd sdk/nodejs

# Login to npm
npm login

# Publish
npm publish
```

### Deploy Java to Maven Central

```bash
cd sdk/java

# Configure ~/.m2/settings.xml with credentials
# Then deploy
mvn clean deploy
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

---

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/build.yml`:

```yaml
name: Build All SDKs

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Setup Java
        uses: actions/setup-java@v3
        with:
          java-version: '17'
          distribution: 'temurin'
      
      - name: Install CMake
        uses: lukka/get-cmake@latest
      
      - name: Build (Windows)
        if: runner.os == 'Windows'
        run: .\setup.ps1 -SkipExamples
      
      - name: Build (Linux/macOS)
        if: runner.os != 'Windows'
        run: |
          chmod +x setup.sh
          ./setup.sh --skip-examples
```

---

## Troubleshooting

### Build Fails on Windows

```powershell
# Ensure Visual Studio C++ tools are installed
# Install: Visual Studio 2019+ with C++ workload

# Check CMake
cmake --version

# Check Python
python --version

# Run with verbose output
.\setup.ps1 -Verbose
```

### Build Fails on Linux

```bash
# Install build essentials
sudo apt-get update
sudo apt-get install build-essential cmake python3-dev

# Run with verbose output
./setup.sh --verbose
```

### Build Fails on macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew packages
brew install cmake python node maven

# Run with verbose output
./setup.sh --verbose
```

### Python SDK Import Error

```bash
# Reinstall in development mode
cd sdk/python
pip install -e . --force-reinstall
```

### Node.js Native Addon Error

```bash
# Rebuild native addon
cd sdk/nodejs
npm rebuild
```

### Java Build Error

```bash
# Clean Maven cache
cd sdk/java
mvn clean
rm -rf ~/.m2/repository/com/mle

# Rebuild
mvn clean install
```

---

## Performance Benchmarks

After building, run benchmarks:

```bash
# MLE vs joblib comparison
python tools/benchmarks/mle_vs_joblib.py

# Export sklearn model
python tools/exporter/sklearn_to_mle.py --demo

# Run complete workflow
python examples/complete_workflow.py
```

---

## Documentation

- **[SDK_OVERVIEW.md](SDK_OVERVIEW.md)** - Complete SDK overview
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Detailed deployment guide
- **[sdk/QUICKSTART.md](sdk/QUICKSTART.md)** - Quick start guide
- **[sdk/INSTALLATION.md](sdk/INSTALLATION.md)** - Installation guide

---

## Support

For build issues:
- 📖 Check documentation above
- 🐛 Open GitHub Issue
- 💬 GitHub Discussions
- 📧 Email: support@mle.dev

---

**Ready to build? Run the setup script!** 🚀

```bash
# Windows
.\setup.ps1

# Linux/macOS
./setup.sh
```
