#!/usr/bin/env python3
"""
Simple MLE Runtime Test
Creates an MLE file in the current directory and tests it by importing and running inference.
"""

import numpy as np
import os
import time

def main():
    print("🧪 Simple MLE Runtime Test")
    print("=" * 40)
    
    # Step 1: Import MLE Runtime
    try:
        import mle_runtime as mle
        print(f"✅ MLE Runtime imported successfully (v{mle.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import MLE Runtime: {e}")
        print("💡 Install with: pip install mle-runtime")
        return False
    
    # Step 2: Check if scikit-learn is available
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        print("✅ Scikit-learn available")
    except ImportError:
        print("❌ Scikit-learn not available")
        print("💡 Install with: pip install scikit-learn")
        return False
    
    # Step 3: Create sample data
    print("\n📊 Creating sample dataset...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_classes=3, 
        n_informative=15, 
        random_state=42
    )
    print(f"✅ Dataset created: {X.shape} features, {len(np.unique(y))} classes")
    
    # Step 4: Train models
    print("\n🔧 Training models...")
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X, y)
    print("✅ LogisticRegression trained")
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X, y)
    print("✅ RandomForestClassifier trained")
    
    # Step 5: Export models to MLE format
    print("\n📦 Exporting models to MLE format...")
    
    models_to_test = [
        (lr_model, "logistic_regression_model.mle", "LogisticRegression"),
        (rf_model, "random_forest_model.mle", "RandomForestClassifier")
    ]
    
    exported_models = []
    
    for model, filename, model_name in models_to_test:
        try:
            start_time = time.time()
            result = mle.export_model(model, filename, input_shape=(1, 20))
            export_time = (time.time() - start_time) * 1000
            
            if result['success']:
                file_size = os.path.getsize(filename)
                print(f"✅ {model_name} exported to {filename}")
                print(f"   📊 Export time: {export_time:.1f}ms")
                print(f"   📦 File size: {file_size} bytes")
                exported_models.append((filename, model_name, X, y))
            else:
                print(f"❌ Failed to export {model_name}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error exporting {model_name}: {e}")
    
    if not exported_models:
        print("❌ No models were successfully exported")
        return False
    
    # Step 6: Test loading and inference
    print(f"\n🔍 Testing {len(exported_models)} exported models...")
    
    test_results = []
    
    for filename, model_name, X_data, y_data in exported_models:
        print(f"\n🧪 Testing {model_name} ({filename})...")
        
        try:
            # Check file exists
            if not os.path.exists(filename):
                print(f"❌ File {filename} does not exist")
                continue
            
            # Load model
            start_time = time.time()
            runtime = mle.load_model(filename)
            load_time = (time.time() - start_time) * 1000
            print(f"✅ Model loaded in {load_time:.1f}ms")
            
            # Get model info
            info = runtime.get_model_info()
            print(f"📋 Model info: {info['metadata']['model_type']}")
            
            # Prepare test data
            X_test = X_data[:10]  # Use first 10 samples for testing
            y_test = y_data[:10]
            
            # Run inference
            start_time = time.time()
            predictions = runtime.run([X_test])
            inference_time = (time.time() - start_time) * 1000
            
            print(f"✅ Inference completed in {inference_time:.1f}ms")
            print(f"📊 Predictions shape: {predictions[0].shape}")
            print(f"🎯 Sample predictions: {predictions[0][:3]}")
            
            # Benchmark performance (suppress warnings during benchmarking)
            print("⚡ Running benchmark...")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                benchmark_results = runtime.benchmark([X_test], num_runs=20)
            print(f"📈 Average inference time: {benchmark_results['mean_time_ms']:.2f}ms")
            print(f"📈 Throughput: {benchmark_results.get('throughput_samples_per_sec', 0):.0f} samples/sec")
            
            test_results.append({
                'model': model_name,
                'file': filename,
                'load_time_ms': load_time,
                'inference_time_ms': inference_time,
                'benchmark_time_ms': benchmark_results['mean_time_ms'],
                'file_size': os.path.getsize(filename),
                'success': True
            })
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            test_results.append({
                'model': model_name,
                'file': filename,
                'success': False,
                'error': str(e)
            })
    
    # Step 7: Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    successful_tests = [r for r in test_results if r['success']]
    failed_tests = [r for r in test_results if not r['success']]
    
    print(f"✅ Successful tests: {len(successful_tests)}")
    print(f"❌ Failed tests: {len(failed_tests)}")
    print(f"📊 Success rate: {len(successful_tests)/len(test_results)*100:.1f}%")
    
    if successful_tests:
        print("\n🎯 Performance Summary:")
        for result in successful_tests:
            print(f"  {result['model']}:")
            print(f"    📦 File size: {result['file_size']} bytes")
            print(f"    ⚡ Load time: {result['load_time_ms']:.1f}ms")
            print(f"    🚀 Inference: {result['benchmark_time_ms']:.2f}ms avg")
    
    if failed_tests:
        print("\n❌ Failed Tests:")
        for result in failed_tests:
            print(f"  {result['model']}: {result['error']}")
    
    # # Step 8: Cleanup
    # print(f"\n🧹 Cleaning up...")
    # for filename, _, _, _ in exported_models:
    #     try:
    #         if os.path.exists(filename):
    #             os.remove(filename)
    #             print(f"🗑️  Removed {filename}")
    #     except Exception as e:
    #         print(f"⚠️  Could not remove {filename}: {e}")
    
    # Final result
    if len(successful_tests) == len(test_results):
        print("\n🎉 All tests passed! MLE Runtime is working correctly.")
        return True
    else:
        print(f"\n⚠️  {len(failed_tests)} test(s) failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)