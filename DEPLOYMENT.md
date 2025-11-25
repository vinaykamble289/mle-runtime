# MLE Runtime - Deployment Guide

Complete guide for deploying all SDKs to their respective package registries.

## Prerequisites

### All Platforms
- Git (for version tagging)
- Accounts on package registries:
  - npm account (for Node.js)
  - PyPI account (for Python)
  - Sonatype OSSRH account (for Maven Central)

### Tools
- **Node.js**: npm CLI
- **Python**: twine (`pip install twine`)
- **Java**: Maven with GPG setup

---

## Quick Deploy (All SDKs)

### Windows
```powershell
# Build everything
.\setup.ps1 -PrepareRelease

# Deploy individually (see sections below)
```

### Linux/macOS
```bash
# Build everything
./setup.sh --prepare-release

# Deploy individually (see sections below)
```

---

## Node.js SDK Deployment

### 1. Prepare Package

```bash
cd sdk/nodejs

# Install dependencies
npm install

# Build TypeScript
npm run build

# Test locally
npm test

# Create package
npm pack
```

### 2. Test Package Locally

```bash
# Install locally
npm install -g mle-runtime-1.0.0.tgz

# Test
node -e "const mle = require('@mle/runtime'); console.log('OK');"
```

### 3. Publish to npm

```bash
# Login to npm
npm login

# Dry run (test)
npm publish --dry-run

# Publish to npm
npm publish

# Or publish with tag
npm publish --tag beta
```

### 4. Verify

```bash
# Install from npm
npm install @mle/runtime

# Check version
npm view @mle/runtime version
```

### Automated Script

**Windows:**
```powershell
cd sdk\nodejs
.\deploy.ps1
```

**Linux/macOS:**
```bash
cd sdk/nodejs
./deploy.sh
```

---

## Python SDK Deployment

### 1. Prepare Package

```bash
cd sdk/python

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build distributions
python setup.py sdist bdist_wheel

# Check packages
twine check dist/*
```

### 2. Test Package Locally

```bash
# Install locally
pip install dist/mle_runtime-1.0.0-*.whl

# Test
python -c "import mle_runtime; print('OK')"
```

### 3. Upload to PyPI

```bash
# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ mle-runtime

# Upload to production PyPI
twine upload dist/*
```

### 4. Verify

```bash
# Install from PyPI
pip install mle-runtime

# Check version
pip show mle-runtime
```

### Automated Script

**Windows:**
```powershell
cd sdk\python
.\deploy.ps1
```

**Linux/macOS:**
```bash
cd sdk/python
./deploy.sh
```

---

## Java SDK Deployment

### 1. Configure Maven Settings

Create/edit `~/.m2/settings.xml`:

```xml
<settings>
  <servers>
    <server>
      <id>ossrh</id>
      <username>YOUR_SONATYPE_USERNAME</username>
      <password>YOUR_SONATYPE_PASSWORD</password>
    </server>
  </servers>
  
  <profiles>
    <profile>
      <id>ossrh</id>
      <activation>
        <activeByDefault>true</activeByDefault>
      </activation>
      <properties>
        <gpg.executable>gpg</gpg.executable>
        <gpg.passphrase>YOUR_GPG_PASSPHRASE</gpg.passphrase>
      </properties>
    </profile>
  </profiles>
</settings>
```

### 2. Prepare Package

```bash
cd sdk/java

# Clean and build
mvn clean install

# Run tests
mvn test

# Package
mvn package
```

### 3. Test Package Locally

```bash
# Install to local Maven repository
mvn install

# Test
java -cp target/mle-runtime-1.0.0.jar com.mle.runtime.MLEEngine
```

### 4. Deploy to Maven Central

```bash
# Deploy to OSSRH (staging)
mvn clean deploy

# Or deploy with release profile
mvn clean deploy -P release

# Then login to https://oss.sonatype.org/
# 1. Find your staging repository
# 2. Close the repository
# 3. Release the repository
```

### 5. Verify

```bash
# Wait for sync (can take 2-4 hours)
# Then check: https://search.maven.org/artifact/com.mle/mle-runtime

# Use in project
# Add to pom.xml:
# <dependency>
#   <groupId>com.mle</groupId>
#   <artifactId>mle-runtime</artifactId>
#   <version>1.0.0</version>
# </dependency>
```

### Automated Script

**Windows:**
```powershell
cd sdk\java
.\deploy.ps1
```

**Linux/macOS:**
```bash
cd sdk/java
./deploy.sh
```

---

## C++ SDK Deployment

The C++ SDK is header-only and doesn't require package registry deployment.

### Distribution Options

#### 1. GitHub Releases

```bash
# Create release archive
cd sdk/cpp
tar -czf mle-cpp-sdk-1.0.0.tar.gz include/ examples/ CMakeLists.txt README.md

# Upload to GitHub releases
```

#### 2. vcpkg

Create `vcpkg.json`:

```json
{
  "name": "mle-runtime",
  "version": "1.0.0",
  "description": "Fast ML inference runtime",
  "homepage": "https://github.com/mle/mle-runtime",
  "dependencies": []
}
```

Submit to vcpkg repository.

#### 3. Conan

Create `conanfile.py`:

```python
from conan import ConanFile

class MLERuntimeConan(ConanFile):
    name = "mle-runtime"
    version = "1.0.0"
    description = "Fast ML inference runtime"
    url = "https://github.com/mle/mle-runtime"
    license = "MIT"
    
    def package(self):
        self.copy("*.h", dst="include", src="include")
```

Upload to Conan Center.

---

## Version Management

### Update Version Numbers

Before deploying, update version in:

1. **Node.js**: `sdk/nodejs/package.json`
   ```json
   "version": "1.0.0"
   ```

2. **Python**: `sdk/python/setup.py` and `pyproject.toml`
   ```python
   version = "1.0.0"
   ```

3. **Java**: `sdk/java/pom.xml`
   ```xml
   <version>1.0.0</version>
   ```

### Git Tagging

```bash
# Create tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tag
git push origin v1.0.0

# Create GitHub release
# Go to: https://github.com/your-repo/releases/new
```

---

## CI/CD Automation

### GitHub Actions Example

Create `.github/workflows/publish.yml`:

```yaml
name: Publish SDKs

on:
  release:
    types: [published]

jobs:
  publish-npm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          registry-url: 'https://registry.npmjs.org'
      - run: cd sdk/nodejs && npm install && npm run build
      - run: cd sdk/nodejs && npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}

  publish-pypi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install build twine
      - run: cd sdk/python && python -m build
      - run: cd sdk/python && twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}

  publish-maven:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-java@v3
        with:
          java-version: '17'
          distribution: 'temurin'
      - run: cd sdk/java && mvn clean deploy
        env:
          MAVEN_USERNAME: ${{ secrets.OSSRH_USERNAME }}
          MAVEN_PASSWORD: ${{ secrets.OSSRH_PASSWORD }}
          GPG_PASSPHRASE: ${{ secrets.GPG_PASSPHRASE }}
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version numbers updated
- [ ] CHANGELOG.md updated
- [ ] Git tag created
- [ ] GitHub release created

### Node.js

- [ ] Package built (`npm run build`)
- [ ] Tests passing (`npm test`)
- [ ] Package created (`npm pack`)
- [ ] Tested locally
- [ ] Published to npm (`npm publish`)
- [ ] Verified on npm registry

### Python

- [ ] Distributions built (`python setup.py sdist bdist_wheel`)
- [ ] Packages checked (`twine check dist/*`)
- [ ] Tested locally
- [ ] Uploaded to Test PyPI
- [ ] Tested from Test PyPI
- [ ] Uploaded to PyPI (`twine upload dist/*`)
- [ ] Verified on PyPI

### Java

- [ ] Package built (`mvn clean install`)
- [ ] Tests passing (`mvn test`)
- [ ] Tested locally
- [ ] Deployed to OSSRH (`mvn deploy`)
- [ ] Staging repository closed
- [ ] Staging repository released
- [ ] Verified on Maven Central (after sync)

### C++

- [ ] Header files packaged
- [ ] Examples included
- [ ] Documentation updated
- [ ] Archive created
- [ ] Uploaded to GitHub releases

---

## Post-Deployment

### Verify Installations

```bash
# Node.js
npm install @mle/runtime
node -e "const mle = require('@mle/runtime'); console.log('OK');"

# Python
pip install mle-runtime
python -c "import mle_runtime; print('OK')"

# Java
# Add to pom.xml and run: mvn install
```

### Update Documentation

- [ ] Update README with new version
- [ ] Update installation instructions
- [ ] Announce on social media/blog
- [ ] Update project website

### Monitor

- [ ] Check download statistics
- [ ] Monitor issue tracker
- [ ] Respond to user feedback
- [ ] Plan next release

---

## Troubleshooting

### npm publish fails

```bash
# Check authentication
npm whoami

# Re-login
npm login

# Check package name availability
npm view @mle/runtime
```

### PyPI upload fails

```bash
# Check credentials
cat ~/.pypirc

# Use token authentication
twine upload --username __token__ --password YOUR_TOKEN dist/*
```

### Maven deploy fails

```bash
# Check settings.xml
cat ~/.m2/settings.xml

# Verify GPG key
gpg --list-keys

# Test GPG signing
gpg --sign test.txt
```

---

## Support

For deployment issues:
- 📧 Email: support@mle.dev
- 💬 GitHub Discussions
- 🐛 GitHub Issues

---

**Ready to deploy? Follow the checklist and scripts above!** 🚀
