# GitHub Repository Launch Checklist

Complete this checklist before pushing to GitHub.

---

## Pre-Launch Preparation

### Code Quality
- [ ] Run `black app/` to format code
- [ ] Run `flake8 app/` to check linting
- [ ] Run `pytest tests/ -v` to verify tests pass
- [ ] Check for hardcoded passwords or secrets
- [ ] Remove debug print statements
- [ ] Update docstrings

### Documentation
- [ ] ✅ README.md created and reviewed
- [ ] ✅ LICENSE file present (MIT)
- [ ] ✅ CONTRIBUTING.md created
- [ ] ✅ CHANGELOG.md created
- [ ] ✅ TROUBLESHOOTING.md created
- [ ] ✅ docs/ARCHITECTURE.md created
- [ ] ✅ docs/PARAMETERS.md created
- [ ] ✅ docs/DEVELOPMENT.md created

### Configuration Files
- [ ] ✅ .gitignore configured
- [ ] ✅ requirements.txt up-to-date
- [ ] ✅ setup.py configured
- [ ] ✅ CITATION.cff created
- [ ] ✅ .github/workflows/tests.yml created
- [ ] ✅ .pre-commit-config.yaml created

### Metadata Updates (CRITICAL)

**README.md**:
- [ ] Replace `yourusername` with actual GitHub username (appears in 5+ places)
- [ ] Update email address if different from birrulwaldi@example.com
- [ ] Verify all links are correct

**setup.py**:
- [ ] Update `author_email`
- [ ] Update `url` and `project_urls`
- [ ] Verify classifiers are accurate

**CITATION.cff**:
- [ ] Update author ORCID (optional but recommended)
- [ ] Update GitHub repository URL
- [ ] Verify citation format

**CONTRIBUTING.md**:
- [ ] Update email address for contact

### Final Checks
- [ ] All temporary test files removed
- [ ] No large binary files in repo (.pth files should be in assets/)
- [ ] .gitignore prevents tracking of unnecessary files
- [ ] No environment variables exposed
- [ ] All imports work correctly

---

## GitHub Setup

### Repository Creation

- [ ] Create new repository on GitHub: https://github.com/new
  - Name: `spectroscopic-analyzer`
  - Description: "AI-powered LIBS spectroscopic analysis software"
  - Visibility: Public (for academic publication)
  - Do NOT initialize with README/LICENSE/gitignore

- [ ] Clone and push:
  ```bash
  cd spectroscopic-analyzer
  git remote add origin https://github.com/yourusername/spectroscopic-analyzer.git
  git branch -M main
  git push -u origin main
  ```

### Repository Settings

**General**
- [ ] Repository name: `spectroscopic-analyzer`
- [ ] Description: "AI-powered LIBS spectroscopic analysis software"
- [ ] Add topics: `libs`, `spectroscopy`, `deep-learning`, `python`, `qt`, `pytorch`

**Access**
- [ ] Visibility: Public
- [ ] Default branch: main

**Merge Button**
- [ ] Allow squash merging ✓
- [ ] Allow merge commits ✓
- [ ] Allow rebase merging ✓

**Branch Protection**

- [ ] Go to Settings → Branches
- [ ] Add rule for branch `main`
- [ ] Check: "Require a pull request before merging"
- [ ] Check: "Dismiss stale pull request approvals when new commits are pushed"
- [ ] Check: "Require status checks to pass before merging"
  - Select: "tests"
- [ ] Check: "Require branches to be up to date before merging"

**Secrets** (if using API keys)
- [ ] Settings → Secrets and variables → Actions
- [ ] Add any required tokens

---

## First Release

### Version 1.0.0

```bash
# Create annotated tag
git tag -a v1.0.0 -m "Initial release: AI-powered LIBS analysis software"

# Push tag to GitHub
git push origin v1.0.0
```

### GitHub Release

- [ ] Go to Releases → Create a new release
- [ ] Select tag: v1.0.0
- [ ] Title: "Version 1.0.0 - Initial Release"
- [ ] Description: (copy from CHANGELOG.md)
  ```markdown
  ## Features
  - Core LIBS spectroscopic analysis engine
  - Deep learning element detection (Informer model)
  - Interactive visualization with drag-select regions
  - Baseline correction using Asymmetric Least Squares
  - Publication-ready plot export (300 DPI PNG)
  - Optional Abel deconvolution
  
  ## Installation
  ```bash
  pip install -r requirements.txt
  python app/main.py
  ```
  
  ## Documentation
  See [README.md](README.md) for installation and usage.
  ```
- [ ] Click "Publish release"

---

## Post-Launch Marketing

### Documentation

- [ ] Check GitHub Actions → Tests workflow passes
- [ ] Verify README renders correctly on GitHub
- [ ] Check all links in documentation work

### Badges (Optional, add to README after setup)

```markdown
[![Tests](https://github.com/yourusername/spectroscopic-analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/spectroscopic-analyzer/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
```

### Social Sharing (Optional)

- [ ] Tweet announcement
- [ ] Share in research communities
- [ ] Add to software registries (ASCL, Zenodo, etc.)

---

## Optional: Advanced Setup

### Zenodo Integration (for DOI)

1. Go to https://zenodo.org/
2. Sign in with GitHub
3. Activate repository for archival
4. Create a GitHub release
5. Zenodo automatically creates DOI
6. Add badge to README:
   ```markdown
   [![DOI](https://zenodo.org/badge/xxxxxxxxxxxx.svg)](https://zenodo.org/badge/latestdoi/xxxxxxxxxxxx)
   ```

### PyPI Publication (Optional)

For `pip install spectroscopic-analyzer`:

1. Create account at https://pypi.org
2. Install build tools:
   ```bash
   pip install build twine
   ```
3. Build package:
   ```bash
   python -m build
   ```
4. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

### ReadTheDocs (Optional)

For auto-hosted documentation:

1. Sign up at https://readthedocs.org
2. Import GitHub repository
3. Set documentation root to `/docs`
4. Add badge to README

---

## Ongoing Maintenance

### Monthly
- [ ] Check for dependency updates
- [ ] Review open issues
- [ ] Test on latest Python version

### With Each Release
- [ ] Update CHANGELOG.md
- [ ] Update version number in setup.py
- [ ] Create git tag
- [ ] Create GitHub Release with notes
- [ ] (Optional) Upload to PyPI

### Community
- [ ] Respond to issues within 48 hours
- [ ] Review pull requests promptly
- [ ] Update contributors list

---

## Success Criteria

Your repository is launch-ready when:

✅ All code quality checks pass  
✅ All tests pass  
✅ Documentation is complete and accurate  
✅ Metadata (author, URLs, email) is updated  
✅ LICENSE is present (MIT)  
✅ .gitignore prevents accidental commits  
✅ Branch protection configured  
✅ First release tagged and published  
✅ README badge shows test status

---

## Quick Reference Links

- **GitHub**: https://github.com/yourusername/spectroscopic-analyzer
- **Issues**: https://github.com/yourusername/spectroscopic-analyzer/issues
- **Discussions**: https://github.com/yourusername/spectroscopic-analyzer/discussions
- **Actions**: https://github.com/yourusername/spectroscopic-analyzer/actions
- **Releases**: https://github.com/yourusername/spectroscopic-analyzer/releases

---

## Questions?

Refer to:
- REPOSITORY_SETUP_SUMMARY.md (overview)
- CONTRIBUTING.md (contributor workflow)
- docs/DEVELOPMENT.md (developer setup)
- TROUBLESHOOTING.md (common issues)

---

**Created**: November 29, 2025  
**Status**: Ready for launch  
**Last Updated**: November 29, 2025

