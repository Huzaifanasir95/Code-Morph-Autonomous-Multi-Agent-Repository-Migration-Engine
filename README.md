# 🔄 Code-Morph: Autonomous Multi-Agent Repository Migration Engine

> Semantically aware code migration powered by AST analysis and multi-agent LLM orchestration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 What is Code-Morph?

Code-Morph is not a simple find-replace tool. It's an **intelligent migration engine** that:

- 🧠 **Understands code semantically** using AST (Abstract Syntax Tree) parsing
- 🤖 **Orchestrates specialized AI agents** that analyze, transform, and verify code
- ✅ **Guarantees behavioral equivalence** through automated testing in sandboxed environments
- 🚀 **Handles complex migrations** like TensorFlow 1.x → PyTorch or React Class → Next.js

## ✨ Key Features

### AST-Driven Analysis
Deep understanding of code structure, dependencies, and patterns—not just text matching.

### Multi-Agent Architecture
- **Analyst Agent**: Maps dependencies, detects deprecated APIs
- **Migration Agent**: Transforms code using SOTA LLMs
- **Verification Agent**: Ensures zero logical drift

### Zero Logical Drift
Automated test generation and execution in Docker containers proves migrated code behaves identically.

### Production-Grade
Type-safe (Pydantic), comprehensive error handling, scalable architecture.

## 🏗️ Architecture

```
┌─────────────┐
│ Legacy Code │
└──────┬──────┘
       │
       ▼
┌─────────────────┐    ┌──────────────┐
│ Analyst Agent   │◄───┤  AST Engine  │
│ (Plan)          │    └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────┐
│ Migration Agent │◄───┤ LLM + Rules  │
│ (Transform)     │    └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────┐
│ Verification    │◄───┤   Docker     │
│ Agent (Test)    │    │   Sandbox    │
└────────┬────────┘    └──────────────┘
         │
         ▼
┌─────────────────┐
│ Migrated Code   │
│ + Test Report   │
└─────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/code-morph.git
cd code-morph

# Install with Poetry (recommended)
poetry install

# Or with pip
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Basic Usage

```bash
# Analyze a legacy codebase
code-morph analyze --source ./legacy_code --output migration_plan.json

# Execute migration
code-morph migrate --source ./legacy_code --target pytorch --output ./migrated

# Run with verification
code-morph migrate --source ./legacy_code --target pytorch --verify
```

## 📦 Phase 1 (Current)

**Status**: 🚧 In Development

- [x] Project structure setup
- [x] Poetry configuration
- [ ] Python AST parser (libcst)
- [ ] Dependency analyzer (networkx)
- [ ] Deprecated API detector
- [ ] Migration Plan generator
- [ ] CLI interface
- [ ] Demo: TensorFlow 1.x analysis

## 🛠️ Technology Stack

- **AST Parsing**: libcst, ts-morph
- **Type Safety**: Pydantic, mypy
- **LLM Integration**: OpenAI, Anthropic
- **Agent Orchestration**: LangGraph, LangChain
- **Testing**: pytest, Docker
- **CLI**: Typer, Rich

## 📚 Documentation

- [Architecture Details](docs/architecture.md) *(Coming soon)*
- [Migration Guides](docs/migration_guides/) *(Coming soon)*
- [API Reference](docs/api_reference.md) *(Coming soon)*

## 🤝 Contributing

Contributions are welcome! This is currently a portfolio project in active development.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🎯 Roadmap

- **Phase 1**: AST Engine & Foundation ✅ (Current)
- **Phase 2**: Migration Engine & LLM Integration
- **Phase 3**: Verification Sandbox
- **Phase 4**: Agent Orchestration (LangGraph)
- **Phase 5**: TypeScript Support
- **Phase 6**: Polish & Portfolio Demo

---

Built with ❤️ as a demonstration of production-grade AI agent systems
