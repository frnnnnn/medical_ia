import os

def test_repo_has_key_files():
    assert os.path.exists("app/main.py")
    assert os.path.exists("models/train_insurance.py")
    assert os.path.exists("models/train_diabetes.py")
    assert os.path.exists("README.md")
