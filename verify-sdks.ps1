# MLE Runtime - SDK Structure Verification Script
# Verifies all SDK files are in place and ready for development

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "     MLE Runtime - SDK Structure Verification          " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check Node.js SDK
Write-Host "[1/4] Verifying Node.js SDK..." -ForegroundColor Yellow
$nodejsFiles = @(
    "sdk\nodejs\package.json",
    "sdk\nodejs\tsconfig.json",
    "sdk\nodejs\binding.gyp",
    "sdk\nodejs\src\index.ts",
    "sdk\nodejs\examples\simple_inference.js",
    "sdk\nodejs\examples\express_server.js",
    "sdk\nodejs\README.md"
)

$nodejsOk = $true
foreach ($file in $nodejsFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [X] Missing: $file" -ForegroundColor Red
        $nodejsOk = $false
        $allGood = $false
    }
}

if ($nodejsOk) {
    Write-Host "  Node.js SDK: READY" -ForegroundColor Green
}

# Check Java SDK
Write-Host ""
Write-Host "[2/4] Verifying Java SDK..." -ForegroundColor Yellow
$javaFiles = @(
    "sdk\java\pom.xml",
    "sdk\java\src\main\java\com\mle\runtime\MLEEngine.java",
    "sdk\java\src\main\java\com\mle\runtime\Device.java",
    "sdk\java\src\main\java\com\mle\runtime\ModelMetadata.java",
    "sdk\java\src\main\java\com\mle\runtime\MLEException.java",
    "sdk\java\examples\SimpleInference.java",
    "sdk\java\examples\SpringBootServer.java",
    "sdk\java\README.md"
)

$javaOk = $true
foreach ($file in $javaFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [X] Missing: $file" -ForegroundColor Red
        $javaOk = $false
        $allGood = $false
    }
}

if ($javaOk) {
    Write-Host "  Java SDK: READY" -ForegroundColor Green
}

# Check Python SDK
Write-Host ""
Write-Host "[3/4] Verifying Python SDK..." -ForegroundColor Yellow
$pythonFiles = @(
    "sdk\python\setup.py",
    "sdk\python\pyproject.toml",
    "sdk\python\MANIFEST.in",
    "sdk\python\mle_runtime\__init__.py",
    "sdk\python\tests\test_engine.py",
    "sdk\python\examples\simple_inference.py",
    "sdk\python\examples\flask_server.py",
    "sdk\python\README.md"
)

$pythonOk = $true
foreach ($file in $pythonFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [X] Missing: $file" -ForegroundColor Red
        $pythonOk = $false
        $allGood = $false
    }
}

if ($pythonOk) {
    Write-Host "  Python SDK: READY" -ForegroundColor Green
}

# Check C++ SDK
Write-Host ""
Write-Host "[4/4] Verifying C++ SDK..." -ForegroundColor Yellow
$cppFiles = @(
    "sdk\cpp\CMakeLists.txt",
    "sdk\cpp\include\mle_client.h",
    "sdk\cpp\examples\simple_inference.cpp",
    "sdk\cpp\examples\CMakeLists.txt",
    "sdk\cpp\README.md"
)

$cppOk = $true
foreach ($file in $cppFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [X] Missing: $file" -ForegroundColor Red
        $cppOk = $false
        $allGood = $false
    }
}

if ($cppOk) {
    Write-Host "  C++ SDK: READY" -ForegroundColor Green
}

# Check Documentation
Write-Host ""
Write-Host "[5/5] Verifying Documentation..." -ForegroundColor Yellow
$docFiles = @(
    "README_SDK.md",
    "SDK_OVERVIEW.md",
    "SDK_COMPLETE.md",
    "SDK_CREATION_SUMMARY.md",
    "DEPLOYMENT.md",
    "BUILD_AND_DEPLOY.md",
    "sdk\README.md",
    "sdk\QUICKSTART.md",
    "sdk\INSTALLATION.md"
)

$docOk = $true
foreach ($file in $docFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [X] Missing: $file" -ForegroundColor Red
        $docOk = $false
        $allGood = $false
    }
}

if ($docOk) {
    Write-Host "  Documentation: COMPLETE" -ForegroundColor Green
}

# Check Deployment Scripts
Write-Host ""
Write-Host "[6/6] Verifying Deployment Scripts..." -ForegroundColor Yellow
$deployFiles = @(
    "sdk\nodejs\deploy.ps1",
    "sdk\nodejs\deploy.sh",
    "sdk\python\deploy.ps1",
    "sdk\python\deploy.sh",
    "sdk\java\deploy.ps1",
    "sdk\java\deploy.sh"
)

$deployOk = $true
foreach ($file in $deployFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [X] Missing: $file" -ForegroundColor Red
        $deployOk = $false
        $allGood = $false
    }
}

if ($deployOk) {
    Write-Host "  Deployment Scripts: READY" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Verification Summary" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$summary = @()
if ($nodejsOk) { $summary += "Node.js" }
if ($javaOk) { $summary += "Java" }
if ($pythonOk) { $summary += "Python" }
if ($cppOk) { $summary += "C++" }

Write-Host "SDKs Ready: $($summary.Count)/4" -ForegroundColor $(if ($summary.Count -eq 4) { "Green" } else { "Yellow" })
foreach ($sdk in $summary) {
    Write-Host "  [OK] $sdk SDK" -ForegroundColor Green
}

if ($allGood) {
    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host "              ALL SDKs VERIFIED!                        " -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor White
    Write-Host ""
    Write-Host "1. Build SDKs:" -ForegroundColor Gray
    Write-Host "   .\setup.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "2. Read documentation:" -ForegroundColor Gray
    Write-Host "   type README_SDK.md" -ForegroundColor Cyan
    Write-Host "   type sdk\QUICKSTART.md" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "3. Deploy to registries:" -ForegroundColor Gray
    Write-Host "   type DEPLOYMENT.md" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Some files are missing. Please check the errors above." -ForegroundColor Red
    Write-Host ""
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# File count summary
Write-Host "File Statistics:" -ForegroundColor White
$totalFiles = (Get-ChildItem -Path sdk -Recurse -File | Measure-Object).Count
Write-Host "  Total SDK files: $totalFiles" -ForegroundColor Cyan

$nodeFiles = (Get-ChildItem -Path sdk\nodejs -Recurse -File | Measure-Object).Count
$javaFiles = (Get-ChildItem -Path sdk\java -Recurse -File | Measure-Object).Count
$pythonFiles = (Get-ChildItem -Path sdk\python -Recurse -File | Measure-Object).Count
$cppFiles = (Get-ChildItem -Path sdk\cpp -Recurse -File | Measure-Object).Count

Write-Host "  Node.js: $nodeFiles files" -ForegroundColor Gray
Write-Host "  Java: $javaFiles files" -ForegroundColor Gray
Write-Host "  Python: $pythonFiles files" -ForegroundColor Gray
Write-Host "  C++: $cppFiles files" -ForegroundColor Gray
Write-Host ""
