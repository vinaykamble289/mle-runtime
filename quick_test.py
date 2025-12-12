#!/usr/bin/env python3
"""
Quick MLE Runtime Test
A simple test to verify MLE Runtime is working correctly.
"""

def main():
    print("🚀 Quick MLE Runtime Test")
    print("-" * 30)
    
    try:
        # Import MLE Runtime
        import mle_runtime as mle
        print(f"✅ MLE Runtime v{mle.__version__} imported")
        
        # Import scikit-learn
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        print("✅ Scikit-learn available")
        
        # Create simple data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        print("✅ Test data created")
        
        # Train model
        model = LogisticRegression()
        model.fit(X, y)
        print("✅ Model trained")
        
        # Export to MLE
        result = mle.export_model(model, 'test_model.mle', input_shape=(1, 5))
        if result['success']:
            print(f"✅ Model exported ({result['file_size_bytes']} bytes)")
        else:
            print("❌ Export failed")
            return False
        
        # Load and test
        runtime = mle.load_model('test_model.mle')
        predictions = runtime.run([X[:5]])
        print(f"✅ Inference successful ({predictions[0].shape})")
        
        # Cleanup
        import os
        if os.path.exists('test_model.mle'):
            os.remove('test_model.mle')
            print("✅ Cleanup completed")
        
        print("\n🎉 MLE Runtime is working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)